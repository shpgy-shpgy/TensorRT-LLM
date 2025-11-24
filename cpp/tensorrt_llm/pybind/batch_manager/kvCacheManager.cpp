/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/pybind/common/bindTypes.h"
#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/torchView.h"

#include <ATen/ATen.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/extension.h>
#include <pybind11/functional.h>



namespace tb = tensorrt_llm::batch_manager;
namespace tbc = tensorrt_llm::batch_manager::kv_connector;
namespace tbk = tensorrt_llm::batch_manager::kv_cache_manager;
namespace tr = tensorrt_llm::runtime;
namespace py = pybind11;
using BlockKey = tbk::BlockKey;
using VecUniqueTokens = tensorrt_llm::runtime::VecUniqueTokens;
using SizeType32 = tensorrt_llm::runtime::SizeType32;
using TokenIdType = tensorrt_llm::runtime::TokenIdType;
using VecTokens = std::vector<TokenIdType>;
using CudaStreamPtr = std::shared_ptr<tensorrt_llm::runtime::CudaStream>;

namespace
{
std::optional<tensorrt_llm::runtime::ITensor::UniquePtr> from_torch(std::optional<at::Tensor> torchPtr)
{
    if (torchPtr)
    {
        return tr::TorchView::of(torchPtr.value());
    }
    return std::nullopt;
}

namespace {

/**
 * Factory: 从 Python shim 创建并返回一个 std::shared_ptr<BlockManager>
 */
std::shared_ptr<tbk::BlockManager> make_block_manager_from_shim(
    py::object py_bm,
    SizeType32 tokens_per_block,
    SizeType32 max_num_sequences = static_cast<SizeType32>(1024),
    SizeType32 max_beam_width = static_cast<SizeType32>(1))
{
    py::gil_scoped_acquire gil;

    if (py_bm.is_none()) {
        throw std::runtime_error("make_block_manager_from_shim: py_bm is None");
    }
    if (!py::hasattr(py_bm, "get_layer_to_pool_mapping") || !py::hasattr(py_bm, "get_pool_primary_base_ptrs")) {
        throw std::runtime_error("make_block_manager_from_shim: py_bm missing required methods");
    }

    // 提取 shim 数据
    at::Tensor layer_map = py_bm.attr("get_layer_to_pool_mapping")().cast<at::Tensor>();
    std::vector<int64_t> prim_ptrs = py_bm.attr("get_pool_primary_base_ptrs")().cast<std::vector<int64_t>>();
    std::vector<int64_t> sec_ptrs;
    if (py::hasattr(py_bm, "get_pool_secondary_base_ptrs")) {
        sec_ptrs = py_bm.attr("get_pool_secondary_base_ptrs")().cast<std::vector<int64_t>>();
    }

    SizeType32 num_layers = static_cast<SizeType32>(layer_map.numel() == 0 ? 0 : layer_map.size(0));
    SizeType32 num_pools = static_cast<SizeType32>(prim_ptrs.size());

    // 构造 BlockManager 所需的参数
    std::vector<SizeType32> numKvHeadsPerLayer((num_layers ? static_cast<size_t>(num_layers) : 1), static_cast<SizeType32>(1));
    SizeType32 sizePerHead = static_cast<SizeType32>(1);
    SizeType32 tokensPerBlock = tokens_per_block;
    std::map<SizeType32, std::tuple<SizeType32, SizeType32>> blocksPerWindow;
    // 使用一个 window 大小 = tokensPerBlock，primary blocks = num_pools，secondary = sec_ptrs.size()
    blocksPerWindow[tokensPerBlock] = { static_cast<SizeType32>(num_pools), static_cast<SizeType32>(sec_ptrs.size()) };

    // 创建合法的 CUDA stream
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> stream;
    try {
        // 使用 default stream handle (nullptr) 来包装一个有效的 CudaStream 对象
        stream = std::make_shared<tensorrt_llm::runtime::CudaStream>(static_cast<cudaStream_t>(nullptr));
    } catch (const std::exception& ex) {
        throw std::runtime_error(std::string("make_block_manager_from_shim: failed to create CudaStream: ") + ex.what());
    }

    SizeType32 maxSequenceLength = tokensPerBlock;
    std::vector<SizeType32> maxAttentionWindowVec = { tokensPerBlock };
    std::optional<tbk::TempAttentionWindowInputs> tempAttentionWindowInputs = std::nullopt;
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
    SizeType32 sinkBubbleLength = 0;
    bool onboardBlocks = false;
    tbk::CacheType cacheType = tbk::CacheType::kSELF;

    // 其它可选参数使用默认/空值
    std::optional<tensorrt_llm::executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt;
    std::shared_ptr<tbk::KVCacheEventManager> eventManager = nullptr;
    bool enablePartialReuse = false;
    bool copyOnPartialReuse = false;
    std::shared_ptr<tbc::KvCacheConnectorManager> kvCacheConnectorManager = nullptr;
    std::optional<tensorrt_llm::executor::kv_cache::BaseAgentConfig> agentConfig = std::nullopt;
    bool enableIndexerKCache = false;
    SizeType32 indexerKCacheQuantBlockSize = 0;
    SizeType32 indexerKCacheIndexHeadDim = 0;

    // 构造 BlockManager 并返回 shared_ptr
    auto bm = std::make_shared<tbk::BlockManager>(
        numKvHeadsPerLayer,
        sizePerHead,
        tokensPerBlock,
        blocksPerWindow,
        max_num_sequences,
        stream,
        maxSequenceLength,
        max_beam_width,
        maxAttentionWindowVec,
        tempAttentionWindowInputs,
        dtype,
        sinkBubbleLength,
        onboardBlocks,
        cacheType,
        secondaryOffloadMinPriority,
        eventManager,
        enablePartialReuse,
        copyOnPartialReuse,
        kvCacheConnectorManager,
        agentConfig,
        enableIndexerKCache,
        indexerKCacheQuantBlockSize,
        indexerKCacheIndexHeadDim
    );

    return bm;
}

static py::object py_make_block_manager_from_shim(
    py::object py_bm,
    SizeType32 tokens_per_block,
    int max_num_sequences = 1024,
    int max_beam_width = 1)
{
    // call real factory and return a pybind-wrapped shared_ptr as a generic py::object
    auto bm = make_block_manager_from_shim(py_bm, static_cast<SizeType32>(tokens_per_block),
                                          static_cast<SizeType32>(max_num_sequences),
                                          static_cast<SizeType32>(max_beam_width));
    return py::cast(bm);
}

/**
 * 将 factory 暴露给 Python 的注册函数。
 */
void register_kv_cache_factory(py::module_ &m)
{
    // bind the Python-friendly wrapper (returns py::object) so pybind11_stubgen won't encounter raw C++ type names
    m.def("make_block_manager_from_shim",
          &py_make_block_manager_from_shim,
          py::arg("py_bm"),
          py::arg("tokens_per_block"),
          py::arg("max_num_sequences") = 1024,
          py::arg("max_beam_width") = 1,
          "Construct a BlockManager from a Python shim and return a shared_ptr wrapped in a Python object. "
          "The shim must implement get_layer_to_pool_mapping() and get_pool_primary_base_ptrs().");
}

} // anonymous namespace

class PyKvCacheManager : public tbk::BaseKVCacheManager
{
public:
    // using BaseKVCacheManager::BaseKVCacheManager; // Inherit constructors
    void allocatePools(bool useUvm = false) override
    {
        PYBIND11_OVERLOAD_PURE(void, tbk::BaseKVCacheManager, allocatePools, useUvm);
    }

    void releasePools() override
    {
        PYBIND11_OVERLOAD_PURE(void, tbk::BaseKVCacheManager, releasePools);
    }

    void startScheduling() override
    {
        PYBIND11_OVERLOAD_PURE(void, tbk::BaseKVCacheManager, startScheduling);
    }

    SizeType32 getTokensPerBlock() const override
    {
        PYBIND11_OVERLOAD_PURE(SizeType32, tbk::BaseKVCacheManager, getTokensPerBlock);
    }

    SizeType32 getMaxNumBlocks() const override
    {
        PYBIND11_OVERLOAD_PURE(SizeType32, tbk::BaseKVCacheManager, getMaxNumBlocks);
    }

    SizeType32 getNumPools() const override
    {
        PYBIND11_OVERLOAD_PURE(SizeType32, tbk::BaseKVCacheManager, getNumPools);
    }

    tbk::KvCacheStats getKvCacheStats() const override
    {
        PYBIND11_OVERLOAD_PURE(tbk::KvCacheStats, tbk::BaseKVCacheManager, getKvCacheStats);
    }

    void addToken(tb::LlmRequest::RequestIdType requestId) override
    {
        PYBIND11_OVERLOAD_PURE(void, tbk::BaseKVCacheManager, addToken, requestId);
    }

    void addSequence(tb::LlmRequest::RequestIdType requestId, SizeType32 inputLength, SizeType32 beamWidth,
        tensorrt_llm::common::OptionalRef<tb::LlmRequest> llmRequest = std::nullopt) override
    {
        PYBIND11_OVERLOAD_PURE(
            void, tbk::BaseKVCacheManager, addSequence, requestId, inputLength, beamWidth, llmRequest);
    }

    std::optional<tbk::KVCacheBlock::IdType> removeSequence(tb::LlmRequest::RequestIdType requestId,
        tensorrt_llm::common::OptionalRef<tb::LlmRequest const> llmRequest = std::nullopt,
        bool pinOnRelease = false) override
    {
        PYBIND11_OVERLOAD_PURE(std::optional<tbk::KVCacheBlock::IdType>, tbk::BaseKVCacheManager, removeSequence,
            requestId, llmRequest, pinOnRelease);
    }

    std::optional<tbk::KVCacheBlock::IdType> storeBlocksForReuse(tb::LlmRequest::RequestIdType requestId,
        tensorrt_llm::common::OptionalRef<tb::LlmRequest const> llmRequest, bool pinBlocks) override
    {
        PYBIND11_OVERLOAD_PURE(std::optional<tbk::KVCacheBlock::IdType>, tbk::BaseKVCacheManager, storeBlocksForReuse,
            requestId, llmRequest, pinBlocks);
    }

    tbk::GenerationRequest const& getSequence(tb::LlmRequest::RequestIdType requestId) const override
    {
        PYBIND11_OVERLOAD_PURE(tbk::GenerationRequest const&, tbk::BaseKVCacheManager, getSequence, requestId);
    }

    void schedulingRemoveSequence(tb::LlmRequest::RequestIdType requestId) override
    {
        PYBIND11_OVERLOAD_PURE(void, tbk::BaseKVCacheManager, schedulingRemoveSequence, requestId);
    }

    tensorrt_llm::runtime::ITensor::SharedPtr getBlockPoolPointers() const override
    {
        PYBIND11_OVERLOAD_PURE(
            tensorrt_llm::runtime::ITensor::UniquePtr, tbk::BaseKVCacheManager, getBlockPoolPointers);
    }

    tensorrt_llm::runtime::ITensor::SharedPtr getLayerToPoolMapping() const override
    {
        PYBIND11_OVERLOAD_PURE(
            tensorrt_llm::runtime::ITensor::UniquePtr, tbk::BaseKVCacheManager, getLayerToPoolMapping);
    }

    void getBlockOffsetsOfBatch(tensorrt_llm::runtime::ITensor& output, SizeType32 firstBatchSlotIdx,
        SizeType32 batchSize, SizeType32 beamWidth) const override
    {
        PYBIND11_OVERLOAD_PURE(
            void, tbk::BaseKVCacheManager, getBlockOffsetsOfBatch, output, firstBatchSlotIdx, batchSize, beamWidth);
    }

    SizeType32 copyBlockOffsets(tensorrt_llm::runtime::ITensor& output, SizeType32 outputSlotOffset,
        tb::LlmRequest::RequestIdType requestId) const override
    {
        PYBIND11_OVERLOAD_PURE(
            SizeType32, tbk::BaseKVCacheManager, copyBlockOffsets, output, outputSlotOffset, requestId);
    }

    bool isEnableBlockReuse() const override
    {
        PYBIND11_OVERLOAD_PURE(bool, tbk::BaseKVCacheManager, isEnableBlockReuse);
    }

    void rewindKVCache(tb::LlmRequest::RequestIdType requestId, SizeType32 rewindLengths) override
    {
        PYBIND11_OVERLOAD_PURE(void, tbk::BaseKVCacheManager, rewindKVCache, requestId, rewindLengths);
    }

    bool isCrossKv() const override
    {
        PYBIND11_OVERLOAD_PURE(bool, tbk::BaseKVCacheManager, isCrossKv);
    }

    std::optional<BlockKey> findNewContextBlock(
        VecUniqueTokens const& uniqueTokens, tb::LlmRequest const& llmRequest) const override
    {
        PYBIND11_OVERLOAD_PURE(
            std::optional<BlockKey>, tbk::BaseKVCacheManager, findNewContextBlock, uniqueTokens, llmRequest);
    }

    void storeContextBlocks(tb::LlmRequest const& llmRequest) override
    {
        PYBIND11_OVERLOAD_PURE(void, tbk::BaseKVCacheManager, storeContextBlocks, llmRequest);
    }

    std::vector<std::vector<SizeType32>> const& getCacheBlockIds(
        tb::LlmRequest::RequestIdType requestId, SizeType32 windowSize) const override
    {
        PYBIND11_OVERLOAD_PURE(std::vector<std::vector<SizeType32>> const&, tbk::BaseKVCacheManager, getCacheBlockIds,
            requestId, windowSize);
    }

    std::vector<std::vector<std::vector<SizeType32>>> getBatchCacheBlockIds(
        std::vector<tb::LlmRequest::RequestIdType> const& requestIds, SizeType32 windowSize) const override
    {
        PYBIND11_OVERLOAD_PURE(std::vector<std::vector<std::vector<SizeType32>>>, tbk::BaseKVCacheManager,
            getBatchCacheBlockIds, requestIds, windowSize);
    }

    SizeType32 getUsedNumBlocks() const override
    {
        PYBIND11_OVERLOAD_PURE(SizeType32, tbk::BaseKVCacheManager, getUsedNumBlocks);
    }

    SizeType32 getNumFreeBlocks() const override
    {
        PYBIND11_OVERLOAD_PURE(SizeType32, tbk::BaseKVCacheManager, getNumFreeBlocks);
    }

    tbk::BlockManager const& getBlockManager() const override
    {
        PYBIND11_OVERLOAD_PURE(tbk::BlockManager const&, tbk::BaseKVCacheManager, getBlockManager);
    }

    std::deque<tensorrt_llm::executor::KVCacheEvent> getLatestEvents(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt) const override
    {
        PYBIND11_OVERLOAD_PURE(
            std::deque<tensorrt_llm::executor::KVCacheEvent>, tbk::BaseKVCacheManager, getLatestEvents, timeout);
    }

    tensorrt_llm::runtime::ITensor::SharedPtr getPrimaryPool(SizeType32 poolIdx) const override
    {
        PYBIND11_OVERLOAD_PURE(
            tensorrt_llm::runtime::ITensor::SharedPtr, tbk::BaseKVCacheManager, getPrimaryPool, poolIdx);
    }

    tensorrt_llm::runtime::ITensor::SharedPtr getIndexerKCachePool() const override
    {
        PYBIND11_OVERLOAD_PURE(
            tensorrt_llm::runtime::ITensor::SharedPtr, tbk::BaseKVCacheManager, getIndexerKCachePool);
    }

    tensorrt_llm::runtime::ITensor::SharedPtr getUniquePrimaryPool() const override
    {
        PYBIND11_OVERLOAD_PURE(
            tensorrt_llm::runtime::ITensor::SharedPtr, tbk::BaseKVCacheManager, getUniquePrimaryPool);
    }

    SizeType32 getPoolLayerIdx(SizeType32 layer_idx) const override
    {
        PYBIND11_OVERLOAD_PURE(SizeType32, tbk::BaseKVCacheManager, getPoolLayerIdx, layer_idx);
    }

    void refreshBlocks() override
    {
        PYBIND11_OVERLOAD_PURE(void, tbk::BaseKVCacheManager, refreshBlocks);
    }

    void flushIterationEvents() override
    {
        PYBIND11_OVERLOAD_PURE(void, tbk::BaseKVCacheManager, flushIterationEvents);
    }
};

// TODO: Deduplicate executor bindings KvCacheStats
class PyBasePeftCacheManager : public tb::BasePeftCacheManager
{
public:
    ~PyBasePeftCacheManager() override = default;

    void addRequestPeft(tb::BasePeftCacheManager::LlmRequestPtr llmRequest, bool tryGpuCache = true) override
    {
        PYBIND11_OVERLOAD_PURE(void, tb::BasePeftCacheManager, addRequestPeft, llmRequest, tryGpuCache);
    }

    tb::BasePeftCacheManager::PeftTable ensureBatch(tb::RequestVector const& contextRequests,
        tb::RequestVector const& generationRequests, bool resetGpuCache = false) override
    {
        PYBIND11_OVERLOAD_PURE(tb::BasePeftCacheManager::PeftTable, tb::BasePeftCacheManager, ensureBatch,
            contextRequests, generationRequests, resetGpuCache);
    }

    void resetDeviceCache() override
    {
        PYBIND11_OVERLOAD_PURE(void, tb::BasePeftCacheManager, resetDeviceCache);
    }

    void markRequestDone(tb::LlmRequest const& llmReq, bool pause = false) override
    {
        PYBIND11_OVERLOAD_PURE(void, tb::BasePeftCacheManager, markRequestDone, llmReq, pause);
    }

    tr::SizeType32 getMaxDevicePages() const override
    {
        PYBIND11_OVERLOAD_PURE(tr::SizeType32, tb::BasePeftCacheManager, getMaxDevicePages);
    }

    tr::SizeType32 getMaxHostPages() const override
    {
        PYBIND11_OVERLOAD_PURE(tr::SizeType32, tb::BasePeftCacheManager, getMaxHostPages);
    }

    tr::SizeType32 determineNumPages(std::shared_ptr<tb::LlmRequest> llmRequest) const override
    {
        PYBIND11_OVERLOAD_PURE(tr::SizeType32, tb::BasePeftCacheManager, determineNumPages, llmRequest);
    }

    bool enabled() const override
    {
        PYBIND11_OVERLOAD_PURE(bool, tb::BasePeftCacheManager, enabled);
    }
};

class PyKVCacheAdapter : public tbk::BaseKVCacheManager {
public:
    explicit PyKVCacheAdapter(py::object py_impl) : _py_impl(std::move(py_impl)) {}

    ~PyKVCacheAdapter() override = default;

    // Forwarding helpers
    bool has_attr(const char* name) const { return py::hasattr(_py_impl, name); }

    // Implement virtuals by forwarding to Python impl (acquire GIL)
    void allocatePools(bool useUvm = false) override {
        py::gil_scoped_acquire gil;
        if (has_attr("allocate_pools")) _py_impl.attr("allocate_pools")(useUvm);
        else throw std::runtime_error("Python impl missing allocate_pools");
    }

    void releasePools() override {
        py::gil_scoped_acquire gil;
        if (has_attr("release_pools")) _py_impl.attr("release_pools")();
        else throw std::runtime_error("Python impl missing release_pools");
    }

    void startScheduling() override {
        py::gil_scoped_acquire gil;
        if (has_attr("start_scheduling")) _py_impl.attr("start_scheduling")();
    }

    void registerKVManager(std::shared_ptr<tbk::BaseKVCacheManager> kv_manager) {
        py::gil_scoped_acquire gil;
        if (has_attr("register_kv_manager")) _py_impl.attr("register_kv_manager")(kv_manager);
    }
    
    void registerKvTensors(py::object kv_tensors) {
        py::gil_scoped_acquire gil;
        if (has_attr("register_kv_tensors")) _py_impl.attr("register_kv_tensors")(kv_tensors);
    }

    tbk::KvCacheStats getKvCacheStats() const override {
        py::gil_scoped_acquire gil;
        tbk::KvCacheStats stats{};
        if (!has_attr("get_kv_cache_stats")) {
            return stats;
        }
        py::object py_stats = _py_impl.attr("get_kv_cache_stats")();
        if (py_stats.is_none()) return stats;
        try {
            if (py::hasattr(py_stats, "max_num_blocks"))
                stats.maxNumBlocks = py_stats.attr("max_num_blocks").cast<SizeType32>();
            if (py::hasattr(py_stats, "free_num_blocks"))
                stats.freeNumBlocks = py_stats.attr("free_num_blocks").cast<SizeType32>();
            if (py::hasattr(py_stats, "used_num_blocks"))
                stats.usedNumBlocks = py_stats.attr("used_num_blocks").cast<SizeType32>();
            if (py::hasattr(py_stats, "tokens_per_block"))
                stats.toksPerBlock = py_stats.attr("tokens_per_block").cast<SizeType32>();
            if (py::hasattr(py_stats, "alloc_total_blocks"))
                stats.allocTotalBlocks = py_stats.attr("alloc_total_blocks").cast<SizeType32>();
            if (py::hasattr(py_stats, "alloc_new_blocks"))
                stats.allocNewBlocks = py_stats.attr("alloc_new_blocks").cast<SizeType32>();
            if (py::hasattr(py_stats, "reused_blocks"))
                stats.reusedBlocks = py_stats.attr("reused_blocks").cast<SizeType32>();
            if (py::hasattr(py_stats, "missed_blocks"))
                stats.missedBlocks = py_stats.attr("missed_blocks").cast<SizeType32>();
            if (py::hasattr(py_stats, "cache_hit_rate"))
                stats.cacheHitRate = py_stats.attr("cache_hit_rate").cast<double>();
            if (py::hasattr(py_stats, "num_free_blocks_per_window_size")) {
                // expect a dict-like mapping window_size -> free_blocks
                py::dict d = py_stats.attr("num_free_blocks_per_window_size");
                for (auto item : d) {
                    int window = item.first.cast<int>();
                    SizeType32 val = item.second.cast<SizeType32>();
                    stats.numFreeBlocksPerWindowSize[window] = val;
                }
            }
            if (py::hasattr(py_stats, "allocated_bytes"))
                stats.allocatedBytes = py_stats.attr("allocated_bytes").cast<size_t>();
        } catch (const std::exception& e) {
            py::print("Warning: failed to convert python KvCacheStats to C++ KvCacheStats:", e.what());
        }
        return stats;
    }

    SizeType32 getTokensPerBlock() const override {
        py::gil_scoped_acquire gil;
        if (has_attr("tokens_per_block")) return py::cast<SizeType32>(_py_impl.attr("tokens_per_block"));
        throw std::runtime_error("Python impl missing tokens_per_block");
    }

    SizeType32 getMaxNumBlocks() const override {
        py::gil_scoped_acquire gil;
        if (has_attr("max_num_blocks")) return py::cast<SizeType32>(_py_impl.attr("max_num_blocks"));
        throw std::runtime_error("Python impl missing max_num_blocks");
    }

    SizeType32 getNumPools() const override {
        py::gil_scoped_acquire gil;
        if (has_attr("num_pools")) return py::cast<SizeType32>(_py_impl.attr("num_pools"));
        throw std::runtime_error("Python impl missing num_pools");
    }

    void addToken(RequestIdType requestId) override {
        py::gil_scoped_acquire gil;
        if (has_attr("add_token")) _py_impl.attr("add_token")(requestId);
    }

    void addSequence(RequestIdType requestId, SizeType32 inputLength, SizeType32 beamWidth,
                     tensorrt_llm::common::OptionalRef<tensorrt_llm::batch_manager::LlmRequest> llmRequest = std::nullopt) override
    {
        py::gil_scoped_acquire gil;
        if (has_attr("add_sequence")) {
            _py_impl.attr("add_sequence")(requestId, inputLength, beamWidth, llmRequest);
        }
    }

    std::optional<tbk::KVCacheBlock::IdType> removeSequence(RequestIdType requestId,
        tensorrt_llm::common::OptionalRef<const tensorrt_llm::batch_manager::LlmRequest> llmRequest = std::nullopt,
        bool pinOnRelease = false) override
    {
        py::gil_scoped_acquire gil;
        if (!has_attr("remove_sequence")) return std::nullopt;
        auto res = _py_impl.attr("remove_sequence")(requestId, pinOnRelease);
        if (res.is_none()) return std::nullopt;
        return py::cast<std::optional<tbk::KVCacheBlock::IdType>>(res);
    }

    std::optional<tbk::KVCacheBlock::IdType> storeBlocksForReuse(RequestIdType requestId,
        tensorrt_llm::common::OptionalRef<const tensorrt_llm::batch_manager::LlmRequest> llmRequest,
        bool pinBlocks) override
    {
        py::gil_scoped_acquire gil;
        if (!has_attr("store_blocks_for_reuse")) return std::nullopt;
        auto res = _py_impl.attr("store_blocks_for_reuse")(requestId, pinBlocks);
        if (res.is_none()) return std::nullopt;
        return py::cast<std::optional<tbk::KVCacheBlock::IdType>>(res);
    }

    tbk::GenerationRequest const& getSequence(RequestIdType requestId) const override {
        py::gil_scoped_acquire gil;
        throw std::runtime_error("getSequence forwarding not implemented in adapter");
    }

    void schedulingRemoveSequence(RequestIdType requestId) override {
        py::gil_scoped_acquire gil;
        if (has_attr("scheduling_remove_sequence")) _py_impl.attr("scheduling_remove_sequence")(requestId);
    }

    tr::ITensor::SharedPtr getBlockPoolPointers() const override {
        py::gil_scoped_acquire gil;
        if (!has_attr("get_block_pool_pointers")){
            py::print("Python impl missing get_block_pool_pointers");
            return nullptr;
        }
        py::object r = _py_impl.attr("get_block_pool_pointers")();
        if (r.is_none()) {
            py::print("get_block_pool_pointers returned None");
            return nullptr;
        }
        at::Tensor t = r.cast<at::Tensor>();
        auto up = tr::TorchView::of(t);
        return tr::ITensor::SharedPtr(std::move(up));
    }

    tr::ITensor::SharedPtr getLayerToPoolMapping() const override {
        py::gil_scoped_acquire gil;
        if (!has_attr("get_layer_to_pool_mapping")) {
            py::print("Python impl missing get_layer_to_pool_mapping");
            return nullptr;
        }
        py::object r = _py_impl.attr("get_layer_to_pool_mapping")();
        if (r.is_none()) {
            py::print("get_layer_to_pool_mapping returned None");
            return nullptr;
        }
        at::Tensor t = r.cast<at::Tensor>();
        auto up = tr::TorchView::of(t);
        return tr::ITensor::SharedPtr(std::move(up));
    }

    void getBlockOffsetsOfBatch(tr::ITensor& output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize,
                                SizeType32 beamWidth) const override
    {
        py::gil_scoped_acquire gil;
        if (!has_attr("get_block_offsets_of_batch")) return;
        auto outputShared = std::shared_ptr<tr::ITensor>(&output, [](tr::ITensor*){});
        auto out_t = tr::Torch::tensor(outputShared);
        _py_impl.attr("get_block_offsets_of_batch")(out_t, firstBatchSlotIdx, batchSize, beamWidth);
    }

    SizeType32 copyBlockOffsets(tr::ITensor& output, SizeType32 outputSlotOffset, RequestIdType requestId) const override {
        py::gil_scoped_acquire gil;
        if (!has_attr("copy_block_offsets")) return 0;
        auto outputShared = std::shared_ptr<tr::ITensor>(&output, [](tr::ITensor*){});
        auto out_t = tr::Torch::tensor(outputShared);
        auto r = _py_impl.attr("copy_block_offsets")(out_t, outputSlotOffset, requestId);
        return py::cast<SizeType32>(r);
    }

    bool isEnableBlockReuse() const override {
        py::gil_scoped_acquire gil;
        if (has_attr("enable_block_reuse")) return py::cast<bool>(_py_impl.attr("enable_block_reuse"));
        return false;
    }

    void rewindKVCache(RequestIdType requestId, SizeType32 rewindLengths) override {
        py::gil_scoped_acquire gil;
        if (has_attr("rewind_kv_cache")) _py_impl.attr("rewind_kv_cache")(requestId, rewindLengths);
    }

    bool isCrossKv() const override {
        py::gil_scoped_acquire gil;
        if (has_attr("cross_kv")) return py::cast<bool>(_py_impl.attr("cross_kv"));
        return false;
    }

    // Many other methods exist on BaseKVCacheManager; implement as needed by forwarding pattern.
    // For brevity, unimplemented methods will throw if called.
    std::optional<tbk::BlockKey> findNewContextBlock(const tensorrt_llm::runtime::VecUniqueTokens& uniqueTokens,
                                                     const tensorrt_llm::batch_manager::LlmRequest& llmRequest) const override
    {
        throw std::runtime_error("findNewContextBlock not implemented in adapter");
    }

    void storeContextBlocks(const tensorrt_llm::batch_manager::LlmRequest& llmRequest) override {
        py::gil_scoped_acquire gil;
        if (has_attr("store_context_blocks")) _py_impl.attr("store_context_blocks")(py::none());
    }

    std::vector<std::vector<SizeType32>> const& getCacheBlockIds(RequestIdType requestId, SizeType32 windowSize) const override {
        throw std::runtime_error("getCacheBlockIds not implemented in adapter");
    }

    std::vector<std::vector<std::vector<SizeType32>>> getBatchCacheBlockIds(
        std::vector<RequestIdType> const& requestIds, SizeType32 windowSize) const override
    {
        throw std::runtime_error("getBatchCacheBlockIds not implemented in adapter");
    }

    SizeType32 getUsedNumBlocks() const override {
        throw std::runtime_error("getUsedNumBlocks not implemented in adapter");
    }

    SizeType32 getNumFreeBlocks() const override {
        throw std::runtime_error("getNumFreeBlocks not implemented in adapter");
    }

    tbk::BlockManager const& getBlockManager() const override
    {
        py::gil_scoped_acquire gil;
        if (has_attr("get_block_manager")) {
            py::object py_bm = _py_impl.attr("get_block_manager")();
            if (py_bm.is_none()) {
                throw std::runtime_error("get_block_manager returned None");
            }
            
            // 如果 Python 返回了 C++ BlockManager，直接转换
            try {
                return py_bm.cast<tbk::BlockManager const&>();
            } catch (const std::exception&) {
                // 尝试转换为 shared_ptr<BlockManager>
                try {
                    auto sp = py_bm.cast<std::shared_ptr<tbk::BlockManager>>();
                    if (!sp) throw std::runtime_error("get_block_manager returned null shared_ptr");
                    _block_manager_holder = sp;
                    return *(_block_manager_holder);
                } catch (const std::exception&) {
                    // 不是 C++ 对象，尝试作为 Python shim 处理
                }
            }

            // 如果 Python 返回 shim，直接调用本地的 factory 函数
            try {
                if (py::hasattr(py_bm, "get_layer_to_pool_mapping") && py::hasattr(py_bm, "get_pool_primary_base_ptrs")) {
                    // 直接使用本文件中定义的 factory 函数，避免导入模块
                    SizeType32 tokens_per_block = this->getTokensPerBlock();
                    auto sp = make_block_manager_from_shim(py_bm, tokens_per_block);
                    if (!sp) throw std::runtime_error("make_block_manager_from_shim returned null shared_ptr");
                    _block_manager_holder = sp;
                    return *(_block_manager_holder);
                } else {
                    throw std::runtime_error("get_block_manager returned an object that is neither C++ BlockManager nor expected shim");
                }
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("get_block_manager: failed to construct C++ BlockManager from Python shim: ") + e.what());
            }
        }
        throw std::runtime_error("getBlockManager not implemented in adapter");
    }

    std::deque<tensorrt_llm::executor::KVCacheEvent> getLatestEvents(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt) const override
    {
        throw std::runtime_error("getLatestEvents not implemented in adapter");
    }

    tr::ITensor::SharedPtr getPrimaryPool(SizeType32 poolIdx) const override {
        throw std::runtime_error("getPrimaryPool not implemented in adapter");
    }

    tr::ITensor::SharedPtr getIndexerKCachePool() const override { return nullptr; }
    tr::ITensor::SharedPtr getUniquePrimaryPool() const override { return nullptr; }

    SizeType32 getPoolLayerIdx(SizeType32 layer_idx) const override {
        throw std::runtime_error("getPoolLayerIdx not implemented in adapter");
    }

    SizeType32 getNumReusedBlocks() const noexcept override {
        py::gil_scoped_acquire gil;
        if (has_attr("get_num_reused_blocks")) 
            return py::cast<SizeType32>(_py_impl.attr("get_num_reused_blocks")());
        return 0;
    }

    tbk::OffsetTableDimensions getOffsetTableDimensions() const override {
        return tbk::OffsetTableDimensions{};
    }

    SizeType32 getNeededBlocksOneStep(const tensorrt_llm::batch_manager::LlmRequest&, bool, SizeType32) const override {
        py::gil_scoped_acquire gil;
        if (has_attr("get_needed_blocks_one_step"))
            return py::cast<SizeType32>(_py_impl.attr("get_needed_blocks_one_step")());
        return 0;
    }

    SizeType32 getRemainingBlocksToCompletion(const tensorrt_llm::batch_manager::LlmRequest&, SizeType32) const override {
        return 0;
    }

    void pinBlocks(RequestIdType requestId) override {
        py::gil_scoped_acquire gil;
        if (has_attr("pin_blocks")) _py_impl.attr("pin_blocks")(requestId);
    }

    tr::ITensor::SharedPtr getBlockScalePoolPointers() const override {
        return nullptr;
    }

    tbk::GenerationRequest& getSequence(RequestIdType requestId) override {
        throw std::runtime_error("getSequence non-const not implemented in adapter");
    }

    void storeNewBlock(const tensorrt_llm::batch_manager::LlmRequest&) override {
        py::gil_scoped_acquire gil;
        if (has_attr("store_new_block")) _py_impl.attr("store_new_block")();
    }

    std::optional<tbk::KVCacheBlock::IdType> getLastBlockId(RequestIdType requestId) const override {
        py::gil_scoped_acquire gil;
        if (has_attr("get_last_block_id")) {
            auto res = _py_impl.attr("get_last_block_id")(requestId);
            if (res.is_none()) return std::nullopt;
            return py::cast<tbk::KVCacheBlock::IdType>(res);
        }
        return std::nullopt;
    }

    SizeType32 getMaxCapacityBatchSize(SizeType32, SizeType32) const override {
        return 0;
    }

    tbk::CacheType getCacheType() const override {
        return tbk::CacheType::kSELF;
    }

    std::shared_ptr<tbk::KVCacheBlock> findBlocksInReuseTreeByBlockKey(const tbk::BlockKey&, SizeType32) override {
        return nullptr;
    }

    void unpinBlocksById(tbk::KVCacheBlock::IdType blockId) override {
        py::gil_scoped_acquire gil;
        if (has_attr("unpin_blocks_by_id")) _py_impl.attr("unpin_blocks_by_id")(blockId);
    }

    void refreshBlocks() override { }
    void flushIterationEvents() override { }

private:
    py::object _py_impl;
    mutable std::shared_ptr<tbk::BlockManager> _block_manager_holder;
};

} // namespace

void tb::kv_cache_manager::KVCacheManagerBindings::initBindings(py::module_& m)
{
    py::class_<tbk::KvCacheStats>(m, "KvCacheStats")
        .def(py::init<>())
        .def_readwrite("max_num_blocks", &tbk::KvCacheStats::maxNumBlocks)
        .def_readwrite("free_num_blocks", &tbk::KvCacheStats::freeNumBlocks)
        .def_readwrite("used_num_blocks", &tbk::KvCacheStats::usedNumBlocks)
        .def_readwrite("tokens_per_block", &tbk::KvCacheStats::toksPerBlock)
        .def_readwrite("alloc_total_blocks", &tbk::KvCacheStats::allocTotalBlocks)
        .def_readwrite("alloc_new_blocks", &tbk::KvCacheStats::allocNewBlocks)
        .def_readwrite("reused_blocks", &tbk::KvCacheStats::reusedBlocks)
        .def_readwrite("missed_blocks", &tbk::KvCacheStats::missedBlocks)
        .def_readwrite("cache_hit_rate", &tbk::KvCacheStats::cacheHitRate)
        .def_readwrite("num_free_blocks_per_window_size", &tbk::KvCacheStats::numFreeBlocksPerWindowSize)
        .def_readonly("allocated_bytes", &tbk::KvCacheStats::allocatedBytes);

    py::class_<tbk::TempAttentionWindowInputs>(m, "TempAttentionWindowInputs")
        .def(py::init<>())
        .def_readwrite("paged_context_fmha", &tbk::TempAttentionWindowInputs::pagedContextFMHA)
        .def_readwrite("max_input_len", &tbk::TempAttentionWindowInputs::maxInputLen)
        .def_readwrite("max_num_tokens", &tbk::TempAttentionWindowInputs::maxNumTokens);

    py::class_<tbk::BlockKey>(m, "BlockKey")
        .def(py::init<>())
        .def(py::init<VecTokens const&, std::optional<tr::LoraTaskIdType>>(), py::arg("tokens"),
            py::arg("lora_task_id") = std::nullopt)
        .def(py::init<bool, std::optional<tr::LoraTaskIdType>, VecUniqueTokens const&>(), py::arg("uses_extra_ids"),
            py::arg("lora_task_id"), py::arg("unique_tokens"))
        .def_readonly("uses_extra_ids", &tbk::BlockKey::usesExtraIds)
        .def_readonly("lora_task_id", &tbk::BlockKey::loraTaskId)
        .def_readonly("unique_tokens", &tbk::BlockKey::uniqueTokens);

    py::class_<tbk::BlockKeyHasher>(m, "BlockKeyHasher")
        .def_static("hash", &tbk::BlockKeyHasher::hash, py::arg("block_key"), py::arg("parent_hash") = 0);

    py::class_<tbk::KVCacheEventManager, std::shared_ptr<tbk::KVCacheEventManager>>(m, "KVCacheEventManager")
        .def(py::init<size_t, std::optional<SizeType32>, std::optional<SizeType32>, SizeType32>(),
            py::arg("max_kv_event_entries"), py::arg("attention_dp_rank") = std::nullopt,
            py::arg("attention_dp_size") = std::nullopt, py::arg("attention_dp_events_gather_period_ms") = 5);

    py::classh<tbk::BaseKVCacheManager, PyKvCacheManager>(m, "BaseKVCacheManager")
        .def_static("calculate_max_num_blocks", &tbk::BaseKVCacheManager::calculateMaxNumBlocks, py::arg("config"),
            py::arg("is_cross_attention"), py::arg("dtype"), py::arg("model_config"), py::arg("world_config"),
            py::arg("window_size_to_layers"), py::arg("allotted_primary_mem_bytes"),
            py::arg("allotted_secondary_mem_bytes"), py::arg("extra_cost_memory"), py::arg("kv_factor"),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "allocate_pools",
            [](tbk::BaseKVCacheManager& self, bool useUvm) {
                if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) {
                    adapter->allocatePools(useUvm);
                    return;
                }
                self.allocatePools(useUvm);
            },
            py::arg("useUvm") = false, py::call_guard<py::gil_scoped_release>())
        .def(
            "release_pools",
            [](tbk::BaseKVCacheManager& self) {
                if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) {
                    adapter->releasePools();
                    return;
                }
                self.releasePools();
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "start_scheduling",
            [](tbk::BaseKVCacheManager& self) {
                if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) {
                    adapter->startScheduling();
                    return;
                }
                self.startScheduling();
            },
            py::call_guard<py::gil_scoped_release>())
        .def_property_readonly(
            "tokens_per_block",
            [](tbk::BaseKVCacheManager& self) -> SizeType32 {
                if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) return adapter->getTokensPerBlock();
                return self.getTokensPerBlock();
            })
        .def_property_readonly(
            "max_num_blocks",
            [](tbk::BaseKVCacheManager& self) -> SizeType32 {
                if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) return adapter->getMaxNumBlocks();
                return self.getMaxNumBlocks();
            })
        .def_property_readonly(
            "num_pools",
            [](tbk::BaseKVCacheManager& self) -> SizeType32 {
                if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) return adapter->getNumPools();
                return self.getNumPools();
            })
        .def(
            "get_kv_cache_stats",
            [](tbk::BaseKVCacheManager& self) {
                if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) return adapter->getKvCacheStats();
                return self.getKvCacheStats();
            },
            py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("max_blocks_per_seq",
            [](tbk::BaseKVCacheManager& self) { return self.getOffsetTableDimensions().maxBlocksPerSeq; })
        .def(
            "get_needed_blocks_one_step",
            [](tbk::BaseKVCacheManager& self, const tensorrt_llm::batch_manager::LlmRequest& req, bool a, SizeType32 b) {
                if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) return adapter->getNeededBlocksOneStep(req, a, b);
                return self.getNeededBlocksOneStep(req, a, b);
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "get_remaining_blocks_to_completion",
            [](tbk::BaseKVCacheManager& self, const tensorrt_llm::batch_manager::LlmRequest& req, SizeType32 window_size) -> SizeType32 {
                if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) return adapter->getRemainingBlocksToCompletion(req, window_size);
                return self.getRemainingBlocksToCompletion(req, window_size);
            },
            py::arg("request"), py::arg("window_size"), py::call_guard<py::gil_scoped_release>())
        .def(
            "add_token",
            [](tbk::BaseKVCacheManager& self, tb::LlmRequest::RequestIdType requestId) {
                if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) { adapter->addToken(requestId); return; }
                self.addToken(requestId);
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "add_sequence",
            [](tbk::BaseKVCacheManager& self, tb::LlmRequest::RequestIdType requestId, SizeType32 inputLength, SizeType32 beamWidth,
            tensorrt_llm::common::OptionalRef<tensorrt_llm::batch_manager::LlmRequest> llmRequest = std::nullopt) {
                if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) { adapter->addSequence(requestId, inputLength, beamWidth, llmRequest); return; }
                self.addSequence(requestId, inputLength, beamWidth, llmRequest);
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "remove_sequence",
            [](tbk::BaseKVCacheManager& self, tb::LlmRequest::RequestIdType requestId,
                tensorrt_llm::common::OptionalRef<const tensorrt_llm::batch_manager::LlmRequest> llmRequest = std::nullopt,
        bool pinOnRelease = false) -> std::optional<tbk::KVCacheBlock::IdType> {
                if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) {
                    return adapter->removeSequence(requestId, llmRequest, pinOnRelease);
                }
                return self.removeSequence(requestId, llmRequest, pinOnRelease);
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "pin_blocks",
            [](tbk::BaseKVCacheManager& self, tb::LlmRequest::RequestIdType requestId) {
                if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) { adapter->pinBlocks(requestId); return; }
                self.pinBlocks(requestId);
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "scheduling_remove_sequence",
            [](tbk::BaseKVCacheManager& self, tb::LlmRequest::RequestIdType requestId) {
                if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) { adapter->schedulingRemoveSequence(requestId); return; }
                self.schedulingRemoveSequence(requestId);
            },
            py::call_guard<py::gil_scoped_release>())
         .def(
             "get_block_pool_pointers",
             [](tbk::BaseKVCacheManager& self)
             {
                 std::optional<at::Tensor> block_pool_pointers{std::nullopt};
                 // 优先由 adapter 提供实现；基类虚函数会正确分发，但保留显式 adapter 分支以便一致性
                 if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) {
                     auto tensor = adapter->getBlockPoolPointers();
                     if (tensor) block_pool_pointers = tr::Torch::tensor(tensor);
                 } else {
                     auto tensor = self.getBlockPoolPointers();
                     if (tensor)
                     {
                         std::shared_ptr<tensorrt_llm::runtime::ITensor> _tensor = std::move(tensor);
                         block_pool_pointers = tr::Torch::tensor(_tensor);
                     }
                 }
                 return block_pool_pointers;
             },
             py::call_guard<py::gil_scoped_release>())
         .def(
             "get_block_scale_pool_pointers",
             [](tbk::BaseKVCacheManager& self)
             {
                 std::optional<at::Tensor> block_scale_pool_pointers{std::nullopt};
                 if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) {
                     auto tensor = adapter->getBlockScalePoolPointers();
                     if (tensor) block_scale_pool_pointers = tr::Torch::tensor(tensor);
                 } else {
                     auto tensor = self.getBlockScalePoolPointers();
                     if (tensor)
                     {
                         std::shared_ptr<tensorrt_llm::runtime::ITensor> _tensor = std::move(tensor);
                         block_scale_pool_pointers = tr::Torch::tensor(_tensor);
                     }
                 }
                 return block_scale_pool_pointers;
             },
             py::call_guard<py::gil_scoped_release>())
         .def(
             "get_layer_to_pool_mapping",
             [](tbk::BaseKVCacheManager& self)
             {
                 std::optional<at::Tensor> layer_to_pool_mapping{std::nullopt};
                 if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) {
                     auto tensor = adapter->getLayerToPoolMapping();
                     if (tensor) layer_to_pool_mapping = tr::Torch::tensor(tensor);
                 } else {
                     auto tensor = self.getLayerToPoolMapping();
                     if (tensor)
                     {
                         std::shared_ptr<tensorrt_llm::runtime::ITensor> _tensor = std::move(tensor);
                         layer_to_pool_mapping = tr::Torch::tensor(_tensor);
                     }
                 }
                 return layer_to_pool_mapping;
             },
             py::call_guard<py::gil_scoped_release>())
         .def(
             "get_primary_pool_data",
             [](tbk::BaseKVCacheManager& self, SizeType32 layer_idx) -> at::Tensor
             {
                 if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) {
                     auto pool = tr::Torch::tensor(adapter->getPrimaryPool(layer_idx));
                     auto pool_layer_idx = adapter->getPoolLayerIdx(layer_idx);
                     return pool.index({torch::indexing::Slice(), pool_layer_idx});
                 }
                 auto pool = tr::Torch::tensor(self.getPrimaryPool(layer_idx));
                 auto pool_layer_idx = self.getPoolLayerIdx(layer_idx);
                 return pool.index({torch::indexing::Slice(), pool_layer_idx});
             },
             py::call_guard<py::gil_scoped_release>())
         .def(
             "get_indexer_k_cache_pool_data",
             [](tbk::BaseKVCacheManager& self, SizeType32 layer_idx) -> at::Tensor
             {
                 if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) {
                     auto pool = tr::Torch::tensor(adapter->getIndexerKCachePool());
                     return pool.index({torch::indexing::Slice(), layer_idx});
                 }
                 auto pool = tr::Torch::tensor(self.getIndexerKCachePool());
                 return pool.index({torch::indexing::Slice(), layer_idx});
             },
             py::call_guard<py::gil_scoped_release>())
         .def(
             "get_unique_primary_pool",
             [](tbk::BaseKVCacheManager& self) {
                 if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) return adapter->getUniquePrimaryPool();
                 return self.getUniquePrimaryPool();
             },
             py::call_guard<py::gil_scoped_release>())
         .def(
             "get_block_offsets_of_batch",
             [](tbk::BaseKVCacheManager& self, at::Tensor output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize,
                 SizeType32 beamWidth)
             {
                 auto _output = from_torch(output);
                 TLLM_CHECK_WITH_INFO(_output.has_value(), "Invalid output tensor.");
                 if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) {
                     adapter->getBlockOffsetsOfBatch(*(_output.value()), firstBatchSlotIdx, batchSize, beamWidth);
                     return;
                 }
                 self.getBlockOffsetsOfBatch(*(_output.value()), firstBatchSlotIdx, batchSize, beamWidth);
             },
             py::call_guard<py::gil_scoped_release>())
         .def(
             "copy_block_offsets",
             [](tbk::BaseKVCacheManager& self, at::Tensor output, SizeType32 outputSlotOffset,
                 tb::LlmRequest::RequestIdType requestId)
             {
                 auto _output = from_torch(output);
                 TLLM_CHECK_WITH_INFO(_output.has_value(), "Invalid output tensor.");
                 if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) {
                     return adapter->copyBlockOffsets(*(_output.value()), outputSlotOffset, requestId);
                 }
                 auto maxBlockCount = self.copyBlockOffsets(*(_output.value()), outputSlotOffset, requestId);
                 return maxBlockCount;
             },
             py::call_guard<py::gil_scoped_release>())
         .def(
             "copy_batch_block_offsets",
             [](tbk::BaseKVCacheManager& self, at::Tensor output,
                 std::vector<tb::LlmRequest::RequestIdType> const& requestIds, SizeType32 const beamWidth,
                 SizeType32 const offset)
             {
                 auto _output = from_torch(output);
                 TLLM_CHECK_WITH_INFO(_output.has_value(), "Invalid output tensor.");
                 if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) {
                     for (size_t i = 0; i < requestIds.size(); ++i) adapter->copyBlockOffsets(*(_output.value()), i * beamWidth + offset, requestIds[i]);
                     return;
                 }
                 for (size_t i = 0; i < requestIds.size(); ++i) self.copyBlockOffsets(*(_output.value()), i * beamWidth + offset, requestIds[i]);
             },
             py::call_guard<py::gil_scoped_release>())
         .def(
             "get_latest_events",
             [](tbk::BaseKVCacheManager& self, std::optional<double> timeout_ms = std::nullopt)
             {
                 if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self)) {
                     if (timeout_ms) return adapter->getLatestEvents(std::chrono::milliseconds(static_cast<int64_t>(*timeout_ms)));
                     return adapter->getLatestEvents(std::nullopt);
                 }
                 if (timeout_ms) {
                     return self.getLatestEvents(std::chrono::milliseconds(static_cast<int64_t>(*timeout_ms)));
                 }
                 return self.getLatestEvents(std::nullopt);
             },
             py::arg("timeout_ms") = std::nullopt, py::call_guard<py::gil_scoped_release>())
         .def_property_readonly("enable_block_reuse", &BaseKVCacheManager::isEnableBlockReuse)
         .def("rewind_kv_cache", &BaseKVCacheManager::rewindKVCache, py::call_guard<py::gil_scoped_release>())
         .def_property_readonly("cross_kv", &BaseKVCacheManager::isCrossKv)
         .def("store_context_blocks", &BaseKVCacheManager::storeContextBlocks, py::call_guard<py::gil_scoped_release>())
         .def("store_blocks_for_reuse", &BaseKVCacheManager::storeBlocksForReuse,
             py::call_guard<py::gil_scoped_release>())
         .def("get_cache_block_ids", &BaseKVCacheManager::getCacheBlockIds, py::call_guard<py::gil_scoped_release>())
         .def("get_batch_cache_block_ids", &BaseKVCacheManager::getBatchCacheBlockIds,
             py::call_guard<py::gil_scoped_release>())
         .def("flush_iteration_events", &BaseKVCacheManager::flushIterationEvents,
             py::call_guard<py::gil_scoped_release>())
         .def("get_last_block_id", &BaseKVCacheManager::getLastBlockId, py::call_guard<py::gil_scoped_release>())
         .def("unpin_blocks_by_id", &BaseKVCacheManager::unpinBlocksById, py::call_guard<py::gil_scoped_release>())
         .def_static("make_kvcached_adapter", 
         [](py::object py_impl) -> std::shared_ptr<tbk::BaseKVCacheManager> {
             return std::shared_ptr<tbk::BaseKVCacheManager>(new PyKVCacheAdapter(py_impl));
         })
        // 为 Python 端统一暴露 register_kv_tensors，优先调用 PyKVCacheAdapter::registerKvTensors
        .def(
            "register_kv_tensors",
            [](tbk::BaseKVCacheManager& self, py::object kv_tensors)
            {
                // 如果底层是我们创建的 Python 适配器，转发到其 registerKvTensors
                if (auto *adapter = dynamic_cast<PyKVCacheAdapter*>(&self))
                {
                    adapter->registerKvTensors(std::move(kv_tensors));
                    return;
                }
            },
            py::arg("kv_tensors"), py::call_guard<py::gil_scoped_release>())
;

    py::enum_<tbk::CacheType>(m, "CacheType")
        .value("SELF", tbk::CacheType::kSELF)
        .value("CROSS", tbk::CacheType::kCROSS)
        .value("SELFKONLY", tbk::CacheType::kSELFKONLY);

    py::classh<tbk::KVCacheManager, tbk::BaseKVCacheManager>(m, "KVCacheManager")
        .def(py::init<std::vector<SizeType32> const&, SizeType32, SizeType32,
                 std::map<SizeType32, std::tuple<SizeType32, SizeType32>> const&, SizeType32, SizeType32,
                 std::vector<SizeType32> const&, std::optional<tbk::TempAttentionWindowInputs> const&,
                 nvinfer1::DataType, SizeType32, bool, int64_t, bool, bool, tbk::CacheType,
                 std::optional<tensorrt_llm::executor::RetentionPriority>, std::shared_ptr<tbk::KVCacheEventManager>,
                 bool, bool, std::shared_ptr<tbc::KvCacheConnectorManager>, bool, SizeType32, SizeType32>(),
            py::arg("num_kv_heads_per_layer"), py::arg("size_per_head"), py::arg("tokens_per_block"),
            py::arg("blocks_per_window"), py::arg("max_num_sequences"), py::arg("max_beam_width"),
            py::arg("max_attention_window_vec"), py::arg("temp_attention_window_inputs"), py::arg("dtype"),
            py::arg("sink_token_length"), py::arg("stream"), py::arg("max_sequence_length"),
            py::arg("enable_block_reuse") = false, py::arg("onboard_blocks") = true,
            py::arg_v("cache_type", tbk::CacheType::kSELF, "bindings.internal.batch_manager.CacheType.SELF"),
            py::arg("secondary_offload_min_priority") = std::nullopt, py::arg("event_manager") = nullptr,
            py::arg("enable_partial_reuse") = true, py::arg("copy_on_partial_reuse") = true,
            py::arg("kv_connector_manager") = nullptr, py::arg("enable_indexer_k_cache") = false,
            py::arg("indexer_k_cache_quant_block_size") = 128, py::arg("indexer_k_cache_index_head_dim") = 0,
            py::call_guard<py::gil_scoped_release>());

    // expose factory to Python
    register_kv_cache_factory(m);
}

void tb::BasePeftCacheManagerBindings::initBindings(py::module_& m)
{
    py::classh<tb::BasePeftCacheManager, PyBasePeftCacheManager>(m, "BasePeftCacheManager")
        .def("add_request_peft", &tb::BasePeftCacheManager::addRequestPeft, py::arg("request"),
            py::arg("try_gpu_cache") = true, py::call_guard<py::gil_scoped_release>())
        .def(
            "ensure_batch",
            [](tb::BasePeftCacheManager& self, tb::RequestVector const& contextRequests,
                tb::RequestVector const& generationRequests, bool resetGpuCache)
            { return self.ensureBatch(contextRequests, generationRequests, resetGpuCache); },
            py::arg("context_requests"), py::arg("generation_requests"), py::arg("reset_gpu_cache") = false,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "reset_device_cache", &tb::BasePeftCacheManager::resetDeviceCache, py::call_guard<py::gil_scoped_release>())
        .def("mark_request_done", &tb::BasePeftCacheManager::markRequestDone, py::arg("request"),
            py::arg("pause") = false, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("max_device_pages", &tb::BasePeftCacheManager::getMaxDevicePages)
        .def_property_readonly("max_host_pages", &tb::BasePeftCacheManager::getMaxHostPages)
        .def("determine_num_pages", &tb::BasePeftCacheManager::determineNumPages, py::arg("request"),
            py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("enabled", &tb::BasePeftCacheManager::enabled);

    py::classh<tb::PeftCacheManager, tb::BasePeftCacheManager>(m, "PeftCacheManager")
        .def(py::init<tb::PeftCacheManagerConfig, tr::ModelConfig, tr::WorldConfig, tr::BufferManager>(),
            py::arg("config"), py::arg("model_config"), py::arg("world_config"), py::arg("buffer_manager"),
            py::call_guard<py::gil_scoped_release>())
        .def("is_task_cached", &tb::PeftCacheManager::isTaskCached, py::arg("taskId"),
            py::call_guard<py::gil_scoped_release>());

    py::classh<tb::NoOpPeftCacheManager, tb::BasePeftCacheManager>(m, "NoOpPeftCacheManager")
        .def(py::init<>(), py::call_guard<py::gil_scoped_release>());
}
