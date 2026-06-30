"""SM90 (Hopper) MegaMoE benchmark / NCU-profile harness.

Mirrors ``tests/test_mega_moe.py``'s ``--ncu-profile-only`` /
``--local-rank-idx`` interface so the same ``scripts/run_ncu_mega_moe.sh``
pattern can drive it for SM90.

In normal (non-NCU) mode it runs a list of ``num_tokens`` values (default:
1, 2, 4, 8, 16, 32) and reports CUDA-event wall time for the launched
MegaMoE path. The reported value is reduced with ``MAX`` across ranks,
which matches the latency of the distributed operation. This matters for Green
Context graph replay because K1 and K2 can overlap on disjoint SM partitions;
summing per-kernel profiler time would count that overlap twice.
"""

import argparse
from contextlib import contextmanager
import os
import random
import sys
import torch
import torch.distributed as dist
from typing import Callable, Iterator, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import get_arch_major


@contextmanager
def _masked_hint_scope(enabled: bool) -> Iterator[None]:
    """Set the SM90 MegaMoE masked-routing hint for one operation.

    :param enabled: Whether to set ``DG_SM90_MOE_MASKED_HINT=1``.
    :type enabled: bool
    :return: Context manager that restores the previous environment value.
    :rtype: Iterator[None]
    """
    old_masked_hint = os.environ.get('DG_SM90_MOE_MASKED_HINT')
    if enabled:
        os.environ['DG_SM90_MOE_MASKED_HINT'] = '1'
    try:
        yield
    finally:
        if enabled:
            if old_masked_hint is None:
                os.environ.pop('DG_SM90_MOE_MASKED_HINT', None)
            else:
                os.environ['DG_SM90_MOE_MASKED_HINT'] = old_masked_hint


def _quantize_grouped_fp8_block_128_128(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    g, n, k = w.shape
    assert n % 128 == 0 and k % 128 == 0
    chunk_g = 4
    w_fp8 = torch.empty_like(w, dtype=torch.float8_e4m3fn)
    sf = torch.empty((g, n // 128, k // 128), dtype=torch.float, device=w.device)
    for start in range(0, g, chunk_g):
        end = min(start + chunk_g, g)
        w_view = w[start:end].view(end - start, n // 128, 128, k // 128, 128).float()
        sf_chunk = w_view.abs().amax(dim=(-1, -3)).clamp(1e-4) / 448.0
        w_fp8[start:end].copy_(
            (w_view / sf_chunk.unsqueeze(-1).unsqueeze(-3)).to(torch.float8_e4m3fn).view(end - start, n, k))
        sf[start:end].copy_(sf_chunk)
    return w_fp8, sf.contiguous()


def _bench_cuda_events(stage_fn: Callable[[], None], launch_fn: Callable[[], torch.Tensor],
                       num_tests: int, num_warmups: int = 3) -> float:
    """Measure one launch path with CUDA event wall time.

    ``stage_fn`` prepares symmetric-memory inputs and performs any needed rank
    synchronization before the timed region. Only ``launch_fn`` is enclosed by
    CUDA events, so overlapping Green Context kernels are measured by their
    critical path instead of by the sum of individual kernel times. Each local
    timing sample is reduced with ``MAX`` across ranks before being considered.

    :param stage_fn: Function that stages inputs before each timed launch.
    :type stage_fn: Callable[[], None]
    :param launch_fn: Function that launches the benchmarked CUDA work.
    :type launch_fn: Callable[[], torch.Tensor]
    :param num_tests: Number of measured iterations.
    :type num_tests: int
    :param num_warmups: Number of warmup iterations, defaults to 3.
    :type num_warmups: int, optional
    :return: Best measured launch time in seconds.
    :rtype: float
    """
    for _ in range(num_warmups):
        stage_fn()
        launch_fn()
        torch.cuda.synchronize()
        dist.barrier()

    times = []
    for _ in range(num_tests):
        stage_fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch_fn()
        end.record()
        torch.cuda.synchronize()
        sample = torch.tensor([start.elapsed_time(end) / 1e3], dtype=torch.float64, device='cuda')
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sample, op=dist.ReduceOp.MAX)
        dist.barrier()
        times.append(float(sample.item()))
    return min(times) if times else float('nan')


def _run_one_config(args, num_tokens, num_max_tokens_per_rank,
                    hidden, intermediate_hidden,
                    num_experts, num_topk, num_ranks, rank_idx, group,
                    activation_clamp, fast_math,
                    print_perf=True):
    """Run one SM90 MegaMoE benchmark configuration.

    :param args: Parsed benchmark command-line arguments.
    :type args: argparse.Namespace
    :param num_tokens: Number of local tokens to benchmark.
    :type num_tokens: int
    :param num_max_tokens_per_rank: Symmetric buffer token capacity per rank.
    :type num_max_tokens_per_rank: int
    :param hidden: Hidden dimension.
    :type hidden: int
    :param intermediate_hidden: Intermediate hidden dimension.
    :type intermediate_hidden: int
    :param num_experts: Total number of experts across ranks.
    :type num_experts: int
    :param num_topk: Top-k routing count.
    :type num_topk: int
    :param num_ranks: Number of distributed ranks.
    :type num_ranks: int
    :param rank_idx: Current rank index.
    :type rank_idx: int
    :param group: Distributed process group.
    :type group: dist.ProcessGroup
    :param activation_clamp: Activation clamp value.
    :type activation_clamp: float
    :param fast_math: Whether to use fast math.
    :type fast_math: bool
    :param print_perf: Whether to print performance rows, defaults to True.
    :type print_perf: bool, optional
    """
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max_tokens_per_rank

    # Symmetric buffer (one per config: cheaper to recreate than to keep max-size)
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
    )

    # Inputs (bf16, then quantised)
    x_bf = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_bf = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden),
        dtype=torch.bfloat16, device='cuda') * 0.05
    l2_bf = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.bfloat16, device='cuda') * 0.05
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_w, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    if args.masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
        topk_w.masked_fill_(topk_idx < 0, 0)

    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128,
                                        use_packed_ue8m0=False)
    l1_w_fp8, l1_w_sf = _quantize_grouped_fp8_block_128_128(l1_bf)
    l2_w_fp8, l2_w_sf = _quantize_grouped_fp8_block_128_128(l2_bf)
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
        (l1_w_fp8, l1_w_sf), (l2_w_fp8, l2_w_sf),
    )

    phase_profile_enabled = os.environ.get('DG_SM90_MOE_PHASE_PROFILE', '0') != '0'
    phase_profile_int_offset = num_experts_per_rank + (num_experts_per_rank % 2)
    phase_profile_ints = 256 if phase_profile_enabled else 0
    stats_numel = phase_profile_int_offset + phase_profile_ints if phase_profile_enabled else num_experts_per_rank
    cum_stats = torch.zeros(stats_numel, dtype=torch.int, device='cuda')
    use_masked_hint = args.masked_ratio > 0
    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l2_flush_buffer = None
    if args.flush_l2_mib > 0:
        l2_flush_buffer = torch.empty(args.flush_l2_mib * 1024 * 1024, dtype=torch.int8, device='cuda')

    green_graph = None
    if args.green_context:
        with _masked_hint_scope(use_masked_hint):
            green_graph = deep_gemm.SM90FP8MegaMoEGreenCtxGraph(
                y, transformed_l1, transformed_l2, buffer,
                cumulative_local_expert_recv_stats=cum_stats,
                recipe=(128, 128, 128),
                activation='swiglu',
                activation_clamp=activation_clamp,
                fast_math=fast_math,
                kernel1_sms=args.kernel1_sms,
                kernel2_sms=args.kernel2_sms,
                reduce_sms=args.reduce_sms,
            )
        dist_print(
            f' green_ctx split: K1={green_graph.kernel1_sms} SMs, '
            f'K2={green_graph.kernel2_sms} SMs, K3={green_graph.reduce_sms} SMs, '
            f'ctx={green_graph.get_green_context_ids()}',
            once_in_node=True,
        )

    def stage_inputs(sync_ranks=True):
        if args.reset_sym_buffer:
            buffer.buffer.zero_()
            cum_stats.zero_()
        buffer.x[:num_tokens].copy_(x_fp8)
        buffer.x_sf[:num_tokens].copy_(x_sf)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_w)
        if l2_flush_buffer is not None:
            l2_flush_buffer.zero_()
        if sync_ranks:
            torch.cuda.synchronize()
            dist.barrier()

    def run_sm90_baseline(out):
        deep_gemm.fp8_mega_moe(
            out, transformed_l1, transformed_l2, buffer,
            cumulative_local_expert_recv_stats=cum_stats,
            recipe=(128, 128, 128),
            activation='swiglu',
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )

    def launch_sm90():
        if green_graph is not None:
            green_graph.replay()
        else:
            run_sm90_baseline(y)
        return y

    def launch_sm90_with_hint():
        with _masked_hint_scope(use_masked_hint):
            return launch_sm90()

    # Stage inputs once; bench-loop re-copies them each call.
    def run_sm90():
        stage_inputs()
        return launch_sm90_with_hint()

    if args.ncu_profile_only:
        dist_print(f'[NCU] tokens={num_tokens} hidden={hidden} ih={intermediate_hidden}',
                   once_in_node=True)
        run_sm90()
        torch.cuda.synchronize()
        dist.barrier()
        buffer.destroy()
        return

    green_bitwise = None
    green_max_abs = None
    if args.green_context and args.green_context_compare:
        y_ref = torch.empty_like(y)
        buffer.buffer.zero_()
        cum_stats.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        stage_inputs()
        with _masked_hint_scope(use_masked_hint):
            run_sm90_baseline(y_ref)
        torch.cuda.synchronize()
        dist.barrier()
        buffer.buffer.zero_()
        cum_stats.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        stage_inputs()
        launch_sm90_with_hint()
        torch.cuda.synchronize()
        dist.barrier()
        green_bitwise = torch.equal(y, y_ref)
        green_max_abs = (y.float() - y_ref.float()).abs().max().item()
        dist_print(f' green_ctx correctness: bitwise={green_bitwise} max_abs={green_max_abs:.6g}',
                   once_in_node=True)
        assert green_bitwise, f'green-context graph differs from baseline, max_abs={green_max_abs:.6g}'

    # Warm up + benchmark
    run_sm90()
    dist.barrier()
    if phase_profile_enabled:
        cum_stats.zero_()
        torch.cuda.synchronize()
        dist.barrier()
    t_sm90 = _bench_cuda_events(stage_inputs, launch_sm90_with_hint, args.num_tests)
    green_phase_profile = None
    if phase_profile_enabled:
        torch.cuda.synchronize()
        green_phase_profile = cum_stats.detach().clone()

    t_baseline = None
    baseline_phase_profile = None
    if args.green_context and args.green_context_benchmark_baseline:
        y_baseline = torch.empty_like(y)
        def launch_sm90_baseline_bench():
            with _masked_hint_scope(use_masked_hint):
                run_sm90_baseline(y_baseline)
            return y_baseline
        stage_inputs()
        launch_sm90_baseline_bench()
        torch.cuda.synchronize()
        dist.barrier()
        if phase_profile_enabled:
            cum_stats.zero_()
            torch.cuda.synchronize()
            dist.barrier()
        t_baseline = _bench_cuda_events(
            stage_inputs, launch_sm90_baseline_bench, args.num_tests)
        if phase_profile_enabled:
            torch.cuda.synchronize()
            baseline_phase_profile = cum_stats.detach().clone()

    # Count tokens that landed on this rank for stats
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[(gathered_topk_idx < rank_idx * num_experts_per_rank) |
                      (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = (gathered_topk_idx != -1).sum().item()

    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    workload_flops = 2 * num_recv_tokens * (hidden * intermediate_hidden * 3)
    tflops = safe_div(workload_flops / 1e12, t_sm90)
    h200_fp8_dense_tflops = 1979.0
    roof_us = workload_flops / (h200_fp8_dense_tflops * 1e12) * 1e6
    sol_pct = safe_div(tflops, h200_fp8_dense_tflops) * 100
    num_touched_experts = max(0, torch.unique(gathered_topk_idx.flatten()).numel() - 1)
    # FP8 weights = 1 byte, FP8 acts = 1 byte, BF16 output = 2 bytes
    num_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden +    # L1 weights
        num_touched_experts * hidden * intermediate_hidden +        # L2 weights
        num_recv_tokens * hidden +                                  # L1 acts read
        num_recv_tokens * intermediate_hidden +                     # L1 out write
        num_recv_tokens * intermediate_hidden +                     # L2 acts read
        num_recv_tokens * hidden * 2                                # L2 out write
    )
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_sm90)
    num_nvlink_bytes = num_recv_tokens * hidden * 3
    nvlink_gbs = safe_div(num_nvlink_bytes / 1e9, t_sm90)

    if print_perf:
        dist_print(
            f' tokens={num_tokens:4d}  recv={num_recv_tokens:5d}  experts={num_touched_experts:4d}  '
            f'{t_sm90 * 1e6:7.1f} us  {tflops:6.1f} TFLOPS  {sol_pct:5.1f}% SOL  '
            f'{hbm_gbs:6.0f} HBM GB/s  {nvlink_gbs:6.0f} NVL GB/s  (rank{rank_idx})',
            once_in_node=True,
        )
        baseline_us = t_baseline * 1e6 if t_baseline is not None else float('nan')
        ratio = t_sm90 / t_baseline if t_baseline is not None and t_baseline > 0 else float('nan')
        baseline_tflops = (
            float('nan') if t_baseline is None
            else safe_div(workload_flops / 1e12, t_baseline)
        )
        baseline_sol_pct = safe_div(baseline_tflops, h200_fp8_dense_tflops) * 100
        if t_baseline is not None:
            dist_print(
                f'   baseline={baseline_us:7.1f} us  baseline_SOL={baseline_sol_pct:5.1f}%  '
                f'green={t_sm90 * 1e6:7.1f} us  ratio={ratio:.4f}',
                once_in_node=True,
            )
        symm_backend = (os.environ.get('DG_SYMM_MEM_BACKEND') or 'unset').replace(' ', '_')
        nvshmem_disable_nvls = (os.environ.get('NVSHMEM_DISABLE_NVLS') or 'unset').replace(' ', '_')
        slurm_job_id = (os.environ.get('SLURM_JOB_ID') or 'unset').replace(' ', '_')
        slurm_partition = (os.environ.get('SLURM_JOB_PARTITION') or 'unset').replace(' ', '_')
        slurm_nodelist = (
            os.environ.get('SLURM_JOB_NODELIST') or os.environ.get('SLURM_NODELIST') or 'unset'
        ).replace(' ', '_')
        bitwise_value = 'na' if green_bitwise is None else str(bool(green_bitwise)).lower()
        max_abs_value = float('nan') if green_max_abs is None else green_max_abs
        dist_print(
            'SM90_MOE_RESULT '
            f'tokens={num_tokens} recv={num_recv_tokens} touched_experts={num_touched_experts} '
            f'ranks={num_ranks} hidden={hidden} intermediate_hidden={intermediate_hidden} '
            f'num_experts={num_experts} topk={num_topk} green_context={int(args.green_context)} '
            f'k1_sms={green_graph.kernel1_sms if green_graph is not None else 0} '
            f'k2_sms={green_graph.kernel2_sms if green_graph is not None else 0} '
            f'k3_sms={green_graph.reduce_sms if green_graph is not None else 0} '
            f'reset_sym_buffer={int(args.reset_sym_buffer)} flush_l2_mib={args.flush_l2_mib} '
            f'symm_backend={symm_backend} nvshmem_disable_nvls={nvshmem_disable_nvls} '
            f'slurm_job_id={slurm_job_id} slurm_partition={slurm_partition} '
            f'slurm_nodelist={slurm_nodelist} '
            f'latency_us={t_sm90 * 1e6:.3f} baseline_us={baseline_us:.3f} ratio={ratio:.6g} '
            f'tflops={tflops:.6g} roof_us={roof_us:.6g} sol_pct={sol_pct:.6g} '
            f'baseline_tflops={baseline_tflops:.6g} baseline_sol_pct={baseline_sol_pct:.6g} '
            f'hbm_gbs={hbm_gbs:.6g} nvlink_gbs={nvlink_gbs:.6g} '
            f'bitwise={bitwise_value} max_abs={max_abs_value:.6g}',
            once_in_node=True,
        )
        if phase_profile_enabled:
            torch.cuda.synchronize()
            names = [
                'dispatch_total', 'dispatch_pull', 'math_loop', 'combine_barrier',
                'combine_reduce', 'gemm_core', 'l1_epilogue', 'l2_epilogue',
                'l1_arrival_wait', 'l2_arrival_wait', 'workspace_cleanup',
                'full_barrier_wait', 'l1_wgmma_group', 'l1_accum_scale',
                'l2_wgmma_lo', 'l2_scale_lo', 'l2_wgmma_hi', 'l2_scale_hi',
                'l1_tma_store_wait', 'l1_ready_mask', 'l1_math_loop', 'l2_math_loop',
                'l2_sf_load', 'l2_sfa_load', 'l2_sfb_load',
                'l2_wgmma_lo_issue', 'l2_wgmma_lo_wait',
                'l2_wgmma_hi_issue', 'l2_wgmma_hi_wait', 'l2_empty_arrive',
                'l2_iter_core',
            ]
            num_profile_metrics = len(names)
            profile_tensor = green_phase_profile if green_phase_profile is not None else cum_stats
            profile = profile_tensor[
                phase_profile_int_offset:phase_profile_int_offset + phase_profile_ints
            ].view(torch.int64).cpu().tolist()
            for i, name in enumerate(names):
                total = profile[i]
                max_v = profile[num_profile_metrics + i]
                count = profile[2 * num_profile_metrics + i]
                avg = float(total) / count if count else 0.0
                dist_print(
                    f'   phase {name:16s} avg={avg:10.0f} max={max_v:10d} count={count}',
                    once_in_node=True,
                )
            if baseline_phase_profile is not None:
                baseline_profile = baseline_phase_profile[
                    phase_profile_int_offset:phase_profile_int_offset + phase_profile_ints
                ].view(torch.int64).cpu().tolist()
                for i, name in enumerate(names):
                    total = baseline_profile[i]
                    max_v = baseline_profile[num_profile_metrics + i]
                    count = baseline_profile[2 * num_profile_metrics + i]
                    avg = float(total) / count if count else 0.0
                    dist_print(
                        f'   baseline_phase {name:16s} avg={avg:10.0f} '
                        f'max={max_v:10d} count={count}',
                        once_in_node=True,
                    )

    dist.barrier()
    buffer.destroy()


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    """Run SM90 MegaMoE benchmarks for one local worker rank.

    :param local_rank: Local rank index passed by ``torch.multiprocessing``.
    :type local_rank: int
    :param num_local_ranks: Number of local worker ranks.
    :type num_local_ranks: int
    :param args: Parsed benchmark command-line arguments.
    :type args: argparse.Namespace
    """
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    forced_num_sms = int(os.environ.get('DG_SM90_MOE_SET_NUM_SMS', '0'))
    if forced_num_sms > 0:
        deep_gemm.set_num_sms(forced_num_sms)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    if get_arch_major() != 9:
        dist_print(f'[SKIP] requires SM90, got SM{get_arch_major()}0', once_in_node=True)
        dist.destroy_process_group()
        return

    if args.batches is None:
        batches = [1, 2, 4, 8, 16, 32]
    else:
        batches = args.batches

    dist_print(
        f'SM90 MegaMoE bench: ranks={num_ranks} hidden={args.hidden} '
        f'ih={args.intermediate_hidden} experts={args.num_experts} topk={args.num_topk} '
        f'masked_ratio={args.masked_ratio} fast_math={bool(args.fast_math)} '
        f'green_context={args.green_context} reset_sym_buffer={args.reset_sym_buffer} '
        f'flush_l2_mib={args.flush_l2_mib}',
        once_in_node=True,
    )

    # In NCU mode we run only one batch (the first one in `batches`) so that
    # ncu's `--launch-count 1` is unambiguous.
    if args.ncu_profile_only:
        batches = batches[:1]

    num_max_tokens_per_rank = (
        max(batches) if args.num_max_tokens_per_rank is None else args.num_max_tokens_per_rank)
    if num_max_tokens_per_rank < max(batches):
        raise ValueError(
            f"--num-max-tokens-per-rank must be >= max(batches), got {num_max_tokens_per_rank} < {max(batches)}")
    for num_tokens in batches:
        _run_one_config(
            args, num_tokens, num_max_tokens_per_rank,
            args.hidden, args.intermediate_hidden,
            args.num_experts, args.num_topk,
            num_ranks, rank_idx, group,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SM90 MegaMoE benchmark')

    parser.add_argument('--ncu-profile-only', action='store_true')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--local-rank-idx', type=int, default=None)

    parser.add_argument('--batches', type=int, nargs='+', default=None,
                        help='List of num_tokens to benchmark (default: 1 2 4 8 16 32)')
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--intermediate-hidden', type=int, default=2048)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--activation-clamp', type=float, default=10.0)
    parser.add_argument('--masked-ratio', type=float, default=0.0)
    parser.add_argument('--fast-math', type=int, default=1)
    parser.add_argument('--green-context', action='store_true',
                        help='Use the SM90 three-kernel Green Context graph path')
    parser.add_argument('--green-context-compare', action='store_true',
                        help='Before benchmarking, assert Green Context output is bitwise equal to fp8_mega_moe')
    parser.add_argument('--green-context-benchmark-baseline', action='store_true',
                        help='In Green Context mode, also time the sequential SM90 baseline and print green/baseline ratio')
    parser.add_argument('--reset-sym-buffer', action='store_true',
                        help='Zero the symmetric buffer before each timed launch, outside the timed region')
    parser.add_argument('--flush-l2-mib', type=int, default=0,
                        help='MiB of device memory to memset before each timed launch, outside the timed region')
    parser.add_argument('--kernel1-sms', type=int, default=None,
                        help='SMs assigned to dispatch+L1 in Green Context mode (default: even split remainder)')
    parser.add_argument('--kernel2-sms', type=int, default=None,
                        help='SMs assigned to L2+combine-scatter in Green Context mode (default: even 52-SM H200 split when possible)')
    parser.add_argument('--reduce-sms', type=int, default=None,
                        help='SMs assigned to final combine-reduce in Green Context mode (default: all SMs)')
    parser.add_argument('--num-tests', type=int, default=20)
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=None,
                        help="Override symmetric-buffer token capacity per rank for capacity-sensitivity sweeps")

    args = parser.parse_args()
    if args.flush_l2_mib < 0:
        parser.error('--flush-l2-mib must be non-negative')

    if args.local_rank_idx is not None:
        test(args.local_rank_idx, args.num_processes, args)
    else:
        np = args.num_processes
        torch.multiprocessing.spawn(test, args=(np, args), nprocs=np)
