import torch
from tensorrt_llm.logger import logger

from ..flashinfer_utils import ENABLE_PDL, IS_FLASHINFER_AVAILABLE

if IS_FLASHINFER_AVAILABLE:
    from flashinfer.activation import silu_and_mul
    from flashinfer.norm import (fused_add_rmsnorm, gemma_fused_add_rmsnorm,
                                 gemma_rmsnorm, rmsnorm)
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

    # Warp this into custom op since flashinfer didn't warp it properly and we want to avoid graph break between mlp layer for user buffer optimization
    @torch.library.custom_op("trtllm::flashinfer_silu_and_mul", mutates_args=())
    def flashinfer_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul(x, enable_pdl=ENABLE_PDL)

    @flashinfer_silu_and_mul.register_fake
    def _(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x).chunk(2, dim=-1)[1].contiguous()

    # Warp this into custom op since flashinfer provides default value for eps with would produce two different graphs depends on the eps value.
    @torch.library.custom_op("trtllm::flashinfer_rmsnorm", mutates_args=())
    def flashinfer_rmsnorm(input: torch.Tensor, weight: torch.Tensor,
                           eps: float) -> torch.Tensor:
        return rmsnorm(input, weight, eps, enable_pdl=ENABLE_PDL)

    @flashinfer_rmsnorm.register_fake
    def _(input: torch.Tensor, weight: torch.Tensor,
          eps: float) -> torch.Tensor:
        return torch.empty_like(input)

    @torch.library.custom_op("trtllm::flashinfer_gemma_rmsnorm",
                             mutates_args=())
    def flashinfer_gemma_rmsnorm(input: torch.Tensor, weight: torch.Tensor,
                                 eps: float) -> torch.Tensor:
        return gemma_rmsnorm(input, weight, eps, enable_pdl=ENABLE_PDL)

    @flashinfer_gemma_rmsnorm.register_fake
    def _(input: torch.Tensor, weight: torch.Tensor,
          eps: float) -> torch.Tensor:
        return torch.empty_like(input)

    @torch.library.custom_op("trtllm::flashinfer_fused_add_rmsnorm",
                             mutates_args=("input", "residual"))
    def flashinfer_fused_add_rmsnorm(input: torch.Tensor,
                                     residual: torch.Tensor,
                                     weight: torch.Tensor, eps: float) -> None:
        """
        调用 flashinfer 的 fused_add_rmsnorm。失败时记录调试信息并回退到 PyTorch 实现：
        回退逻辑：tmp = input + residual（不破坏原始张量除非写回），对 tmp 做 RMSNorm（按最后一维），
        将结果写回 input（in-place），并把 tmp 写回 residual（in-place），以尽量保持原有语义。
        """
        try:
            fused_add_rmsnorm(input, residual, weight, eps, enable_pdl=ENABLE_PDL)
            return
        except Exception as e:
            # 收集尽可能多的调试信息，但要防御性处理防止再次抛出
            try:
                info = {
                    "input_shape": tuple(input.shape),
                    "input_dtype": str(input.dtype),
                    "input_device": str(input.device),
                    "residual_shape": tuple(residual.shape),
                    "weight_shape": tuple(weight.shape) if weight is not None else None,
                    "input_has_nan": bool(torch.isnan(input).any().item()) if input.numel() > 0 else None,
                    "residual_has_nan": bool(torch.isnan(residual).any().item()) if residual.numel() > 0 else None,
                    "input_min": float(input.min().cpu().item()) if input.numel() > 0 else None,
                    "input_max": float(input.max().cpu().item()) if input.numel() > 0 else None,
                    "eps": float(eps),
                    "exception": str(e),
                }
            except Exception:
                info = {"exception_while_collecting_debug": True, "orig_exception": str(e)}
            logger.error("flashinfer fused_add_rmsnorm failed, falling back to PyTorch. Debug info: %s", info)

            # 回退实现（尽量使用 in-place 操作以匹配原 op 的 mutates_args 语义）
            try:
                # tmp = input + residual（在可能的大张量上分配一次临时）
                tmp = residual + input
                # RMSNorm: 按最后一维计算 inv = 1 / sqrt(mean(x^2) + eps)
                sq = tmp.to(torch.float32).pow(2)
                mean_sq = sq.mean(dim=-1, keepdim=True)
                inv = torch.rsqrt(mean_sq + eps)
                out = tmp * inv
                if weight is not None:
                    out = out * weight
                # 将结果写回 input（in-place），并把 tmp 写回 residual（in-place）
                input.copy_(out.to(input.dtype))
                residual.copy_(tmp)
                return
            except Exception as e2:
                logger.exception("Fallback fused_add_rmsnorm also failed: %s", e2)
                raise

    @torch.library.custom_op("trtllm::flashinfer_gemma_fused_add_rmsnorm",
                             mutates_args=("input", "residual"))
    def flashinfer_gemma_fused_add_rmsnorm(input: torch.Tensor,
                                           residual: torch.Tensor,
                                           weight: torch.Tensor,
                                           eps: float) -> None:
        gemma_fused_add_rmsnorm(input,
                                residual,
                                weight,
                                eps,
                                enable_pdl=ENABLE_PDL)

    @torch.library.custom_op(
        "trtllm::flashinfer_apply_rope_with_cos_sin_cache_inplace",
        mutates_args=("query", "key"))
    def flashinfer_apply_rope_with_cos_sin_cache_inplace(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool = True,
    ) -> None:
        apply_rope_with_cos_sin_cache_inplace(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox,
        )

    @flashinfer_apply_rope_with_cos_sin_cache_inplace.register_fake
    def _(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool = True,
    ):
        return

def flashinfer_rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float, enable_pdl: bool = None):
    """
    Wrapper around flashinfer rmsnorm. On native failure, log debug info and fall back to a pure-PyTorch RMSNorm.
    """
    try:
        # 原有调用（保持不变）
        return rmsnorm(input, weight, eps, enable_pdl=ENABLE_PDL if enable_pdl is None else enable_pdl)
    except Exception as e:
        # 记录调试信息便于定位
        try:
            info = {
                "shape": tuple(input.shape),
                "dtype": str(input.dtype),
                "device": str(input.device),
                "weight_shape": tuple(weight.shape) if weight is not None else None,
                "has_nan": bool(torch.isnan(input).any().item()),
                "has_inf": bool(torch.isinf(input).any().item()),
                "min": float(input.min().cpu().item()) if input.numel() > 0 else None,
                "max": float(input.max().cpu().item()) if input.numel() > 0 else None,
                "eps": float(eps),
                "exception": str(e),
            }
        except Exception:
            info = {"exception_while_collecting_debug": True, "orig_exception": str(e)}
        logger.error("flashinfer rmsnorm failed, falling back to PyTorch implementation. Debug info: %s", info)
        # 回退实现：RMSNorm 的常见实现（按最后一维规约均方根）
        try:
            # input: (..., hidden)
            sq = input.to(torch.float32).pow(2)
            mean_sq = sq.mean(dim=-1, keepdim=True)
            inv = torch.rsqrt(mean_sq + eps)
            out = input * inv
            if weight is not None:
                out = out * weight
            # 保持 dtype 与 device
            return out.to(input.dtype)
        except Exception as e2:
            logger.exception("Fallback PyTorch RMSNorm also failed: %s", e2)
            raise
