import os
import torch
from typing import Tuple, Optional
from ..utils.math import align

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C

_symm_mem_backend: Optional[str] = None


def _maybe_set_symm_mem_backend() -> None:
    """Set the symmetric-memory backend from ``DG_SYMM_MEM_BACKEND``.

    :raises RuntimeError: If the requested backend changes after it was set.
    """
    backend = os.environ.get("DG_SYMM_MEM_BACKEND")
    if not backend:
        return
    global _symm_mem_backend
    if _symm_mem_backend is None:
        symm_mem.set_backend(backend)
        _symm_mem_backend = backend
    elif _symm_mem_backend != backend:
        raise RuntimeError(
            f"DG_SYMM_MEM_BACKEND changed from {_symm_mem_backend} to {backend}")


def _empty_symm_mem_for_group(num_bytes: int, group: dist.ProcessGroup) -> torch.Tensor:
    """Allocate symmetric memory for a process group.

    :param num_bytes: Number of bytes to allocate.
    :type num_bytes: int
    :param group: Process group that will rendezvous the allocation.
    :type group: dist.ProcessGroup
    :return: One-dimensional int8 symmetric-memory tensor.
    :rtype: torch.Tensor
    """
    group_name = group.group_name
    enable_fn = getattr(symm_mem, "enable_symm_mem_for_group", None)
    if enable_fn is not None:
        enable_fn(group_name)

    get_backend = getattr(symm_mem, "get_backend", None)
    backend = get_backend(torch.device("cuda")) if get_backend is not None else None
    try:
        return symm_mem.empty(num_bytes, dtype=torch.int8, device="cuda")
    except RuntimeError as exc:
        if "no group info associated" not in str(exc) or backend == "NCCL":
            raise

    allocator_owner = getattr(symm_mem, "_SymmetricMemory", None)
    allocator = getattr(allocator_owner, "empty_strided_p2p", None)
    if allocator is None:
        return symm_mem.empty(num_bytes, dtype=torch.int8, device="cuda")

    allocator_args = (
        (num_bytes,),
        (1,),
        torch.int8,
        torch.device(f"cuda:{torch.cuda.current_device()}"),
    )
    if backend == "NVSHMEM":
        return allocator(*allocator_args, None)

    return allocator(
        *allocator_args,
        group_name,
    )


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 # MoE arguments
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 use_fp8_dispatch: bool = True,
                 activation: str = 'swiglu'):
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden

        # Allocate a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            use_fp8_dispatch, activation
        )
        _maybe_set_symm_mem_backend()
        self.buffer = _empty_symm_mem_for_group(num_bytes, group)
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 use_fp8_dispatch: bool = True,
                                 activation: str = 'swiglu') -> SymmBuffer:
    # Token count must be aligned to block sizes
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        use_fp8_dispatch, activation
    )


def _interleave_l1_weights(l1_weights: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    def interleave(t, gran: int = 8) -> torch.Tensor:
        g, n, *rest = t.shape
        half = n // 2
        gate = t[:, :half].reshape(g, half // gran, gran, *rest)
        up = t[:, half:].reshape(g, half // gran, gran, *rest)
        return torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))

    return interleave(l1_weights[0]), interleave(l1_weights[1])


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    assert sf.dtype == torch.int and mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    return torch.empty_like(sf).copy_(result)


def transform_weights_for_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    # L1: interleave gate/up, then transpose SF for UTCCP
    l1_interleaved = _interleave_l1_weights(l1_weights)
    l1_weights = (l1_interleaved[0], _transpose_sf_for_utccp(l1_interleaved[1]))
    # L2: only transpose SF for UTCCP
    l2_weights = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    return l1_weights, l2_weights


def transform_weights_for_mega_moe_sm90(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """SM90 (Hopper) variant of `transform_weights_for_mega_moe`.

    SM90 has no TMEM / UTCCP path, so the SF tensors are consumed directly by
    WGMMA promote and don't need the 4x32 transpose. With block (128, 128)
    weight quantization, weight SFs are read by the math warpgroup directly
    from global memory in their natural ``(E, N/128, K/128)`` MN-major layout
    and require no transformation. Only L1's gate/up FP8 weight interleave is
    preserved.
    """
    l1_fp8, l1_sf = l1_weights
    # Reuse the gran-8 N interleave on the FP8 weight only; the block SF stays
    # in its natural ``(E, 2*IH/128, H/128)`` layout (gate then up along N).
    def _interleave_one(t, gran: int = 8) -> torch.Tensor:
        g, n, *rest = t.shape
        half = n // 2
        gate = t[:, :half].reshape(g, half // gran, gran, *rest)
        up = t[:, half:].reshape(g, half // gran, gran, *rest)
        return torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))

    return (_interleave_one(l1_fp8), l1_sf), l2_weights


def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True):
    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math
    )

def fp8_mega_moe(y: torch.Tensor,
                 l1_weights: Tuple[torch.Tensor, torch.Tensor],
                 l2_weights: Tuple[torch.Tensor, torch.Tensor],
                 sym_buffer: SymmBuffer,
                 cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                 recipe: Tuple[int, int, int] = (128, 128, 128),
                 activation: str = 'swiglu',
                 activation_clamp: Optional[float] = None,
                 fast_math: bool = True):
    """SM90 (Hopper) MegaMoE entry point.

    Expects FP8 e4m3 weights and block-(128, 128) float scale factors. The
    weight SF layout matches the convention used by ``DeepSeekV4FlashFp8`` /
    DeepEP, so the same SF tensors can be physically shared between the
    DeepEP path and this kernel.
    """
    _C.fp8_mega_moe(
        y,
        l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math
    )


def get_build_cudart_version_for_mega_moe() -> int:
    """Return the CUDA Runtime version used to build MegaMoE APIs.

    .. code-block:: python

       build_version = deep_gemm.get_build_cudart_version_for_mega_moe()

    :return: Integer ``CUDART_VERSION`` captured at extension build time.
    :rtype: int
    """
    return int(_C.get_build_cudart_version_for_mega_moe())


def _default_sm90_green_sms_split(
    device_sms: int,
    num_max_tokens_per_rank: Optional[int] = None,
    num_tokens: Optional[int] = None,
) -> Tuple[int, int, int]:
    """Return a legal default SM split for the SM90 Green Context graph.

    K1 and K2 run on Green Context SM partitions whose coscheduled group size
    is at least two SMs, so both partitions must be even. Odd-SM devices leave
    one SM outside the K1/K2 split while K3 still uses the full device.

    :param device_sms: Number of SMs reported by the current CUDA device.
    :type device_sms: int
    :param num_max_tokens_per_rank: Optional symmetric-buffer token capacity,
        used as a fallback when num_tokens is not available.
    :type num_max_tokens_per_rank: int, optional
    :param num_tokens: Optional active token count for the current graph. H200
        large-token profiles prefer a larger K2 partition.
    :type num_tokens: int, optional
    :raises ValueError: If fewer than four SMs are available.
    :return: K1, K2, and K3 SM counts.
    :rtype: Tuple[int, int, int]
    """
    if device_sms < 4:
        raise ValueError(f'SM90 MegaMoE Green Context requires at least 4 SMs, got {device_sms}')

    split_sms = device_sms - (device_sms % 2)
    if split_sms >= 80:
        active_tokens = num_tokens if num_tokens is not None else (num_max_tokens_per_rank or 0)
        if split_sms == 132 and active_tokens >= 8192:
            kernel2_sms = 60
        elif split_sms == 132 and active_tokens >= 4096:
            kernel2_sms = 54
        elif split_sms == 132 and active_tokens >= 2048:
            kernel2_sms = 56
        elif split_sms == 132 and active_tokens >= 512:
            kernel2_sms = 56
        else:
            kernel2_sms = 52
    else:
        kernel2_sms = max(2, (split_sms // 3) & ~1)
    kernel2_sms = min(kernel2_sms, split_sms - 2)
    if kernel2_sms % 2 != 0:
        kernel2_sms -= 1
    kernel2_sms = max(2, kernel2_sms)
    kernel1_sms = split_sms - kernel2_sms
    if kernel1_sms <= 1 or kernel2_sms <= 1:
        raise ValueError(
            f'Green Context partitions require at least 2 SMs, got {kernel1_sms}/{kernel2_sms}')
    if kernel1_sms % 2 != 0 or kernel2_sms % 2 != 0:
        raise ValueError(
            f'Green Context partitions must be even, got {kernel1_sms}/{kernel2_sms}')
    if kernel1_sms + kernel2_sms > device_sms:
        raise ValueError(
            f'Green Context split {kernel1_sms}+{kernel2_sms} exceeds {device_sms} device SMs')
    return kernel1_sms, kernel2_sms, device_sms


class SM90FP8MegaMoEGreenCtxGraph:
    """SM90 MegaMoE graph using Green Context split SM partitions.

    K1 runs dispatch + L1 + SwiGLU, K2 runs L2 + combine scatter concurrently
    on a disjoint SM partition, and K3 runs the final top-k combine reduce.
    The fused/two-kernel ``fp8_mega_moe`` API is unchanged; this graph is an
    explicit opt-in path for H200/Hopper tuning.
    """
    def __init__(self,
                 y: torch.Tensor,
                 l1_weights: Tuple[torch.Tensor, torch.Tensor],
                 l2_weights: Tuple[torch.Tensor, torch.Tensor],
                 sym_buffer: SymmBuffer,
                 cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                 recipe: Tuple[int, int, int] = (128, 128, 128),
                 activation: str = 'swiglu',
                 activation_clamp: Optional[float] = None,
                 fast_math: bool = True,
                 kernel1_sms: Optional[int] = None,
                 kernel2_sms: Optional[int] = None,
                 reduce_sms: Optional[int] = None):
        """Create an SM90 MegaMoE Green Context graph.

        .. code-block:: python

           graph = deep_gemm.SM90FP8MegaMoEGreenCtxGraph(
               y, l1_weights, l2_weights, sym_buffer,
               cumulative_local_expert_recv_stats=stats,
           )
           graph.replay()

        :param y: Output tensor with shape ``(num_tokens, hidden)``.
        :type y: torch.Tensor
        :param l1_weights: Linear1 FP8 weights and scale factors.
        :type l1_weights: Tuple[torch.Tensor, torch.Tensor]
        :param l2_weights: Linear2 FP8 weights and scale factors.
        :type l2_weights: Tuple[torch.Tensor, torch.Tensor]
        :param sym_buffer: Symmetric MegaMoE communication workspace.
        :type sym_buffer: SymmBuffer
        :param cumulative_local_expert_recv_stats: Optional stats buffer,
            defaults to None.
        :type cumulative_local_expert_recv_stats: torch.Tensor, optional
        :param recipe: MegaMoE tile recipe, defaults to ``(128, 128, 128)``.
        :type recipe: Tuple[int, int, int], optional
        :param activation: Activation name, defaults to ``'swiglu'``.
        :type activation: str, optional
        :param activation_clamp: Activation clamp value, defaults to None.
        :type activation_clamp: float, optional
        :param fast_math: Whether to use fast math, defaults to True.
        :type fast_math: bool, optional
        :param kernel1_sms: SMs assigned to dispatch + L1, defaults to None.
        :type kernel1_sms: int, optional
        :param kernel2_sms: SMs assigned to L2 + combine scatter, defaults to
            None.
        :type kernel2_sms: int, optional
        :param reduce_sms: SMs assigned to final combine reduce, defaults to
            None.
        :type reduce_sms: int, optional
        """
        build_cudart_version = get_build_cudart_version_for_mega_moe()
        if build_cudart_version < 13020:
            raise RuntimeError(
                f'SM90 MegaMoE Green Context requires CUDA Runtime 13.2+, '
                f'but DeepGEMM was built with CUDART_VERSION={build_cudart_version}')
        device_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        num_tokens = int(y.size(0))
        default_kernel1_sms, default_kernel2_sms, default_reduce_sms = _default_sm90_green_sms_split(
            device_sms, sym_buffer.num_max_tokens_per_rank, num_tokens)
        default_split_sms = default_kernel1_sms + default_kernel2_sms
        if kernel2_sms is None:
            kernel2_sms = default_kernel2_sms
        if kernel1_sms is None:
            kernel1_sms = default_split_sms - kernel2_sms
        if reduce_sms is None:
            reduce_sms = default_reduce_sms
        if kernel1_sms <= 1 or kernel2_sms <= 1 or reduce_sms <= 0:
            raise ValueError(
                f'Invalid Green Context SM counts: K1={kernel1_sms}, '
                f'K2={kernel2_sms}, K3={reduce_sms}')
        if kernel1_sms + kernel2_sms > device_sms:
            raise ValueError(
                f'Green Context split {kernel1_sms}+{kernel2_sms} exceeds {device_sms} device SMs')
        if reduce_sms > device_sms:
            raise ValueError(f'K3 SM count {reduce_sms} exceeds {device_sms} device SMs')
        if kernel1_sms % 2 != 0 or kernel2_sms % 2 != 0:
            raise ValueError(
                f'Green Context partitions must be even, got {kernel1_sms}/{kernel2_sms}')

        if cumulative_local_expert_recv_stats is None:
            num_experts_per_rank = l1_weights[0].shape[0]
            phase_profile_enabled = os.environ.get('DG_SM90_MOE_PHASE_PROFILE', '0') != '0'
            phase_profile_int_offset = num_experts_per_rank + (num_experts_per_rank % 2)
            stats_numel = phase_profile_int_offset + 256 if phase_profile_enabled else num_experts_per_rank
            cumulative_local_expert_recv_stats = torch.zeros(stats_numel, dtype=torch.int, device=y.device)
        self.stats = cumulative_local_expert_recv_stats
        self.y = y
        self.sym_buffer = sym_buffer
        self.kernel1_sms = kernel1_sms
        self.kernel2_sms = kernel2_sms
        self.reduce_sms = reduce_sms
        self._impl = _C.SM90FP8MegaMoEGreenCtxGraph(
            y,
            l1_weights, l2_weights,
            cumulative_local_expert_recv_stats,
            sym_buffer.buffer,
            sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
            sym_buffer.num_max_tokens_per_rank,
            sym_buffer.num_experts, sym_buffer.num_topk,
            recipe,
            activation, activation_clamp,
            fast_math,
            kernel1_sms, kernel2_sms, reduce_sms,
        )

    def replay(self):
        """Replay the captured Green Context graph on the current CUDA stream."""
        self._impl.replay()

    def get_green_context_ids(self):
        """Return the two Green Context execution context IDs.

        :return: K1 and K2 execution context IDs.
        :rtype: tuple[int, int]
        """
        return self._impl.get_green_context_ids()
