"""Independent PyTorch reference for the legacy SM90 MegaMoE numerics.

The implementation is intentionally outside the production kernel path.  It
dequantizes the exact FP8 inputs and weights used by a test case, reproduces
the old per-row/per-64 E4M3 intermediate, stores each L2 contribution as BF16,
and reduces top-k contributions in FP32.
"""

from contextlib import contextmanager
import math
from typing import Iterator, List

import torch
import torch.distributed as dist

from deep_gemm.utils.math import ceil_to_ue8m0


def _dequant_per_token_per_128(
    values: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Dequantize ``(M, K)`` E4M3 values with one scale per row/K128."""
    rows, k = values.shape
    if k % 128 != 0 or scales.shape != (rows, k // 128):
        raise ValueError(
            f"invalid per-token FP8 shapes: values={values.shape}, "
            f"scales={scales.shape}"
        )
    view = values.float().view(rows, k // 128, 128)
    return (view * scales.unsqueeze(-1)).view(rows, k)


def _dequant_weight_block_128_128(
    values: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Dequantize one ``(N, K)`` E4M3 weight with 128x128 scales."""
    n, k = values.shape
    if n % 128 != 0 or k % 128 != 0:
        raise ValueError(f"weight shape must be 128-aligned, got {values.shape}")
    if scales.shape != (n // 128, k // 128):
        raise ValueError(
            f"invalid block-scale shape: values={values.shape}, scales={scales.shape}"
        )
    view = values.float().view(n // 128, 128, k // 128, 128)
    return (view * scales[:, None, :, None]).view(n, k)


def _legacy_per_row_per_64_e4m3(values: torch.Tensor) -> torch.Tensor:
    """Quantize/dequantize with the legacy power-of-two E4M3 K64 scale."""
    rows, k = values.shape
    if k % 64 != 0:
        raise ValueError(f"intermediate K must be divisible by 64, got {k}")
    view = values.view(rows, k // 64, 64)
    amax = view.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    scales = ceil_to_ue8m0(amax / 448.0)
    quantized = (view / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return (quantized.float() * scales.unsqueeze(-1)).view(rows, k)


def _swiglu(values: torch.Tensor, activation_clamp: float) -> torch.Tensor:
    gate, up = values.chunk(2, dim=-1)
    if math.isfinite(activation_clamp):
        gate = gate.clamp(max=activation_clamp)
        up = up.clamp(min=-activation_clamp, max=activation_clamp)
    return torch.nn.functional.silu(gate) * up


def _all_gather(tensor: torch.Tensor, group: dist.ProcessGroup) -> List[torch.Tensor]:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, tensor, group=group)
    return gathered


@contextmanager
def _highest_float32_matmul_precision() -> Iterator[None]:
    """Prevent TF32 from weakening the independent FP32 GEMM reference."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@torch.no_grad()
def legacy_quantized_pytorch_reference(
    x_fp8: torch.Tensor,
    x_scales: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    l1_weight_fp8: torch.Tensor,
    l1_weight_scales: torch.Tensor,
    l2_weight_fp8: torch.Tensor,
    l2_weight_scales: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    num_experts: int,
    activation_clamp: float,
) -> torch.Tensor:
    """Return the local-rank BF16 output of the legacy numerical path.

    Weights are gathered once, while tokens remain on their source rank.  Local
    routes are grouped by expert so temporary storage is bounded by one expert
    instead of expanding a weight matrix for every route.
    """
    world_size = dist.get_world_size(group)
    num_tokens, hidden = x_fp8.shape
    num_topk = topk_indices.shape[1]
    num_experts_per_rank = l1_weight_fp8.shape[0]
    intermediate_hidden = l2_weight_fp8.shape[-1]

    if num_experts != world_size * num_experts_per_rank:
        raise ValueError(
            f"num_experts={num_experts} does not match {world_size} ranks x "
            f"{num_experts_per_rank} local experts"
        )
    if topk_indices.shape != (num_tokens, num_topk):
        raise ValueError(f"invalid top-k index shape: {topk_indices.shape}")
    if topk_weights.shape != topk_indices.shape:
        raise ValueError("top-k weights and indices must have identical shapes")
    if l1_weight_fp8.shape[1:] != (2 * intermediate_hidden, hidden):
        raise ValueError(f"invalid L1 weight shape: {l1_weight_fp8.shape}")
    if l2_weight_fp8.shape[1:] != (hidden, intermediate_hidden):
        raise ValueError(f"invalid L2 weight shape: {l2_weight_fp8.shape}")

    all_l1_weights = _all_gather(l1_weight_fp8, group)
    all_l1_scales = _all_gather(l1_weight_scales, group)
    all_l2_weights = _all_gather(l2_weight_fp8, group)
    all_l2_scales = _all_gather(l2_weight_scales, group)

    x_fp32 = _dequant_per_token_per_128(x_fp8, x_scales)
    contributions = torch.zeros(
        (num_tokens, num_topk, hidden), dtype=torch.bfloat16, device=x_fp8.device
    )
    flat_experts = topk_indices.reshape(-1)

    with _highest_float32_matmul_precision():
        for expert_idx in range(num_experts):
            flat_routes = torch.nonzero(
                flat_experts == expert_idx, as_tuple=False
            ).squeeze(-1)
            if flat_routes.numel() == 0:
                continue

            token_indices = torch.div(
                flat_routes, num_topk, rounding_mode="floor"
            )
            topk_slots = flat_routes.remainder(num_topk)
            owner_rank = expert_idx // num_experts_per_rank
            local_expert = expert_idx % num_experts_per_rank

            l1_weight = _dequant_weight_block_128_128(
                all_l1_weights[owner_rank][local_expert],
                all_l1_scales[owner_rank][local_expert],
            )
            l1_output = torch.matmul(
                x_fp32.index_select(0, token_indices), l1_weight.transpose(0, 1)
            )
            del l1_weight

            route_weights = topk_weights[token_indices, topk_slots].unsqueeze(-1)
            l2_input = _swiglu(l1_output, activation_clamp) * route_weights
            del l1_output, route_weights
            l2_input = _legacy_per_row_per_64_e4m3(l2_input)

            l2_weight = _dequant_weight_block_128_128(
                all_l2_weights[owner_rank][local_expert],
                all_l2_scales[owner_rank][local_expert],
            )
            l2_output = torch.matmul(l2_input, l2_weight.transpose(0, 1))
            del l2_input, l2_weight
            contributions[token_indices, topk_slots] = l2_output.to(torch.bfloat16)
            del l2_output

    # Each contribution crosses the old K2->K3 boundary as BF16.  K3 promotes
    # to FP32 for the top-k sum and casts its final output to BF16.
    output = contributions.float().sum(dim=1).to(torch.bfloat16)
    return output.contiguous()
