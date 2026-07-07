#!/usr/bin/env python3
"""Validate the scale-free FP8 E5M2 MegaMoE combine path.

Each case generates one set of inputs, runs the exact sequential SM90 path
with BF16 K2-to-K3 contributions, then runs the Green Context candidate with
FP8 E5M2 contributions.  An optional independent PyTorch reference reproduces
the legacy per-row/per-64 E4M3 intermediate and BF16 contribution boundary.
Error metrics are reduced conservatively across all eight ranks.  The full
suite contains 112 deterministic cases.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
import random
import sys
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.testing import calc_diff, get_arch_major
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist
from mega_moe_sm90_pytorch_reference import (
    legacy_quantized_pytorch_reference,
)


@dataclass(frozen=True)
class AccuracyCase:
    """One deterministic correctness case."""

    index: int
    label: str
    seed_base: int
    num_tokens: int = 7000
    x_scale: float = 1.0
    weight_scale: float = 0.05
    score_scale: float = 1.0
    masked_ratio: float = 0.0
    distribution: str = "normal"


@contextmanager
def _masked_hint_scope(enabled: bool) -> Iterator[None]:
    old_value = os.environ.get("DG_SM90_MOE_MASKED_HINT")
    if enabled:
        os.environ["DG_SM90_MOE_MASKED_HINT"] = "1"
    try:
        yield
    finally:
        if enabled:
            if old_value is None:
                os.environ.pop("DG_SM90_MOE_MASKED_HINT", None)
            else:
                os.environ["DG_SM90_MOE_MASKED_HINT"] = old_value


def _quantize_grouped_fp8_block_128_128(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    groups, n, k = weight.shape
    assert n % 128 == 0 and k % 128 == 0
    chunk_groups = 4
    weight_fp8 = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scale = torch.empty(
        (groups, n // 128, k // 128), dtype=torch.float, device=weight.device
    )
    for start in range(0, groups, chunk_groups):
        end = min(start + chunk_groups, groups)
        view = weight[start:end].view(
            end - start, n // 128, 128, k // 128, 128
        ).float()
        scale_chunk = view.abs().amax(dim=(-1, -3)).clamp(1e-4) / 448.0
        weight_fp8[start:end].copy_(
            (
                view / scale_chunk.unsqueeze(-1).unsqueeze(-3)
            ).to(torch.float8_e4m3fn).view(end - start, n, k)
        )
        scale[start:end].copy_(scale_chunk)
    return weight_fp8, scale.contiguous()


def _build_cases() -> List[AccuracyCase]:
    cases: List[AccuracyCase] = []

    def add(label: str, seed_base: int, **kwargs) -> None:
        cases.append(
            AccuracyCase(
                index=len(cases), label=label, seed_base=seed_base, **kwargs
            )
        )

    # Production distribution: 32 independent random streams.
    for seed in range(32):
        add(f"default_seed_{seed:02d}", seed)

    # Magnitude coverage: 12 profiles x 4 independent seeds.
    magnitude_profiles = [
        ("x_low", 0.25, 0.05, 1.0),
        ("x_high", 4.0, 0.05, 1.0),
        ("weight_low", 1.0, 0.025, 1.0),
        ("weight_high", 1.0, 0.10, 1.0),
        ("score_low", 1.0, 0.05, 0.25),
        ("score_high", 1.0, 0.05, 4.0),
        ("all_small", 0.5, 0.025, 0.5),
        ("all_large", 2.0, 0.10, 2.0),
        ("x_high_weight_low", 4.0, 0.025, 1.0),
        ("x_low_weight_high", 0.25, 0.10, 1.0),
        ("x_high_score_high", 4.0, 0.05, 4.0),
        ("weight_high_score_high", 1.0, 0.10, 4.0),
    ]
    for profile_idx, (name, x_scale, weight_scale, score_scale) in enumerate(
        magnitude_profiles
    ):
        for seed_idx in range(4):
            add(
                f"magnitude_{name}_seed_{seed_idx}",
                1000 + profile_idx * 16 + seed_idx,
                x_scale=x_scale,
                weight_scale=weight_scale,
                score_scale=score_scale,
            )

    # Routing density and token-count coverage: 5 profiles x 4 seeds.
    routing_profiles = [
        ("mask_025", 7000, 0.25),
        ("mask_050", 7000, 0.50),
        ("mask_075", 7000, 0.75),
        ("tokens_1024", 1024, 0.0),
        ("tokens_4096", 4096, 0.0),
    ]
    for profile_idx, (name, num_tokens, masked_ratio) in enumerate(
        routing_profiles
    ):
        for seed_idx in range(4):
            add(
                f"routing_{name}_seed_{seed_idx}",
                2000 + profile_idx * 16 + seed_idx,
                num_tokens=num_tokens,
                masked_ratio=masked_ratio,
            )

    # Alternate activation/weight distributions: 3 profiles x 4 seeds.
    for distribution_idx, distribution in enumerate(
        ("uniform", "rademacher", "heavy_tail")
    ):
        for seed_idx in range(4):
            add(
                f"distribution_{distribution}_seed_{seed_idx}",
                3000 + distribution_idx * 16 + seed_idx,
                distribution=distribution,
            )

    assert len(cases) == 112, len(cases)
    assert [case.index for case in cases] == list(range(112))
    assert len({case.label for case in cases}) == 112
    return cases


def _rank_seed(case: AccuracyCase, rank_idx: int) -> int:
    # Distinct streams across both cases and ranks, within manual_seed's range.
    return (case.seed_base * 1_000_003 + rank_idx * 9_176 + 17) % (2**63 - 1)


def _make_bf16_tensor(
    shape: Sequence[int],
    distribution: str,
    scale: float,
    heavy_tail_offset: int,
) -> torch.Tensor:
    tensor = torch.empty(tuple(shape), dtype=torch.bfloat16, device="cuda")
    if distribution == "normal":
        tensor.normal_(mean=0.0, std=scale)
    elif distribution == "uniform":
        bound = math.sqrt(3.0) * scale
        tensor.uniform_(-bound, bound)
    elif distribution == "rademacher":
        tensor.bernoulli_(0.5).mul_(2.0).sub_(1.0).mul_(scale)
    elif distribution == "heavy_tail":
        stride = 1024
        multiplier = 16.0
        rms_correction = math.sqrt(
            1.0 + (multiplier * multiplier - 1.0) / stride
        )
        tensor.normal_(mean=0.0, std=scale / rms_correction)
        flat = tensor.view(-1)
        flat[heavy_tail_offset % stride :: stride].mul_(multiplier)
    else:
        raise ValueError(f"unsupported distribution: {distribution}")
    return tensor


def _copy_tensor_tuple(
    destination: Tuple[torch.Tensor, ...], source: Tuple[torch.Tensor, ...]
) -> None:
    assert len(destination) == len(source)
    for dst, src in zip(destination, source):
        assert dst.shape == src.shape and dst.dtype == src.dtype
        dst.copy_(src)


def _python_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summarize_reported_comparison(
    records: Sequence[Dict[str, object]], key: str
) -> Dict[str, object]:
    metrics = [record[key] for record in records]
    assert all(isinstance(metric, dict) for metric in metrics)
    typed_metrics = [metric for metric in metrics if isinstance(metric, dict)]
    golden_diffs = [float(metric["golden_diff"]) for metric in typed_metrics]
    relative_rmses = [float(metric["relative_rmse"]) for metric in typed_metrics]
    ordered_indices = sorted(
        range(len(records)),
        key=lambda index: golden_diffs[index],
        reverse=True,
    )
    return {
        "num_cases": len(records),
        "golden_diff_p50": _python_percentile(golden_diffs, 0.50),
        "golden_diff_p95": _python_percentile(golden_diffs, 0.95),
        "golden_diff_p99": _python_percentile(golden_diffs, 0.99),
        "golden_diff_max": max(golden_diffs) if golden_diffs else float("nan"),
        "relative_rmse_p50": _python_percentile(relative_rmses, 0.50),
        "relative_rmse_p95": _python_percentile(relative_rmses, 0.95),
        "relative_rmse_p99": _python_percentile(relative_rmses, 0.99),
        "relative_rmse_max": (
            max(relative_rmses) if relative_rmses else float("nan")
        ),
        "max_abs_max": max(
            (float(metric["max_abs"]) for metric in typed_metrics),
            default=float("nan"),
        ),
        "sampled_p99_abs_max": max(
            (float(metric["sampled_p99_abs"]) for metric in typed_metrics),
            default=float("nan"),
        ),
        "sampled_p999_abs_max": max(
            (float(metric["sampled_p999_abs"]) for metric in typed_metrics),
            default=float("nan"),
        ),
        "reference_nonfinite": sum(
            int(metric["reference_nonfinite"]) for metric in typed_metrics
        ),
        "candidate_nonfinite": sum(
            int(metric["candidate_nonfinite"]) for metric in typed_metrics
        ),
        "worst_case_indices": [
            int(records[index]["case"]["index"]) for index in ordered_indices[:5]
        ],
        "worst_case_labels": [
            str(records[index]["case"]["label"]) for index in ordered_indices[:5]
        ],
    }


def _compute_global_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> Dict[str, object]:
    reference_nonfinite = int((~torch.isfinite(reference)).sum().item())
    candidate_nonfinite = int((~torch.isfinite(candidate)).sum().item())

    if reference_nonfinite or candidate_nonfinite:
        local_error = torch.tensor(
            [float("inf")] * 5, dtype=torch.float64, device="cuda"
        )
        local_bitwise = torch.tensor([0], dtype=torch.int32, device="cuda")
    else:
        absolute = (candidate.float() - reference.float()).abs()
        sample = absolute.reshape(-1)[::16]
        quantiles = torch.quantile(
            sample, torch.tensor([0.99, 0.999], device="cuda")
        )
        rmse = torch.sqrt(torch.mean(absolute * absolute))
        reference_rms = torch.sqrt(torch.mean(reference.float() ** 2))
        relative_rmse = rmse / torch.clamp(reference_rms, min=1e-30)
        local_error = torch.tensor(
            [
                float(calc_diff(candidate, reference).item()),
                float(absolute.max().item()),
                float(relative_rmse.item()),
                float(quantiles[0].item()),
                float(quantiles[1].item()),
            ],
            dtype=torch.float64,
            device="cuda",
        )
        local_bitwise = torch.tensor(
            [int(torch.equal(candidate, reference))],
            dtype=torch.int32,
            device="cuda",
        )

    nonfinite = torch.tensor(
        [reference_nonfinite, candidate_nonfinite],
        dtype=torch.int64,
        device="cuda",
    )
    dist.all_reduce(local_error, op=dist.ReduceOp.MAX)
    dist.all_reduce(nonfinite, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_bitwise, op=dist.ReduceOp.MIN)
    values = local_error.cpu().tolist()
    counts = nonfinite.cpu().tolist()
    return {
        "golden_diff": values[0],
        "max_abs": values[1],
        "relative_rmse": values[2],
        "sampled_p99_abs": values[3],
        "sampled_p999_abs": values[4],
        "reference_nonfinite": int(counts[0]),
        "candidate_nonfinite": int(counts[1]),
        "bitwise": bool(local_bitwise.item()),
    }


def _select_cases(args: argparse.Namespace) -> List[AccuracyCase]:
    cases = _build_cases()
    if args.case_indices:
        requested = {int(value) for value in args.case_indices.split(",") if value}
        unknown = requested.difference(range(len(cases)))
        if unknown:
            raise ValueError(f"unknown case indices: {sorted(unknown)}")
        return [case for case in cases if case.index in requested]
    if args.suite == "smoke":
        labels = {"default_seed_00", "magnitude_all_large_seed_0"}
        return [case for case in cases if case.label in labels]
    return cases


def _run_worker(
    local_rank: int, num_local_ranks: int, args: argparse.Namespace
) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    del group
    if get_arch_major() != 9:
        raise RuntimeError(f"accuracy sweep requires SM90, got SM{get_arch_major()}0")
    if num_ranks != 8:
        raise RuntimeError(f"accuracy sweep requires 8 ranks, got {num_ranks}")

    cases = _select_cases(args)
    hidden = 3072
    intermediate_hidden = 1536
    num_experts = 128
    num_topk = 16
    num_max_tokens_per_rank = 7000
    num_experts_per_rank = num_experts // num_ranks

    output_file = None
    if rank_idx == 0 and args.output_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
        output_file = open(args.output_jsonl, "w", encoding="utf-8", buffering=1)

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    cumulative_stats = torch.zeros(
        num_experts_per_rank, dtype=torch.int, device="cuda"
    )

    persistent_l1: Optional[Tuple[torch.Tensor, ...]] = None
    persistent_l2: Optional[Tuple[torch.Tensor, ...]] = None
    graphs: Dict[Tuple[int, bool], object] = {}
    candidate_outputs: Dict[Tuple[int, bool], torch.Tensor] = {}
    reference_outputs: Dict[Tuple[int, bool], torch.Tensor] = {}
    results: List[Dict[str, object]] = []

    if rank_idx == 0:
        print(
            f"FP8_ACCURACY_SUITE_START cases={len(cases)} tol={args.tolerance} "
            f"k1_sms={args.kernel1_sms} k2_sms={args.kernel2_sms} "
            f"pytorch_reference={int(args.pytorch_reference)} "
            f"indices={','.join(str(case.index) for case in cases)}",
            flush=True,
        )

    for ordinal, case in enumerate(cases, start=1):
        seed = _rank_seed(case, rank_idx)
        torch.manual_seed(seed)
        random.seed(seed)

        x_bf = _make_bf16_tensor(
            (case.num_tokens, hidden), case.distribution, case.x_scale, seed
        )
        l1_bf = _make_bf16_tensor(
            (num_experts_per_rank, intermediate_hidden * 2, hidden),
            case.distribution,
            case.weight_scale,
            seed + 1,
        )
        l2_bf = _make_bf16_tensor(
            (num_experts_per_rank, hidden, intermediate_hidden),
            case.distribution,
            case.weight_scale,
            seed + 2,
        )
        # Keep routing scores continuous even for Rademacher activation/weight
        # cases so top-k ties do not collapse all tokens onto the same experts.
        scores = torch.randn(
            (case.num_tokens, num_experts), dtype=torch.float, device="cuda"
        ).mul_(case.score_scale)
        topk_weights, topk_indices = torch.topk(
            scores, num_topk, dim=-1, largest=True, sorted=False
        )
        if case.masked_ratio > 0:
            random_mask = torch.rand_like(topk_indices, dtype=torch.float)
            topk_indices.masked_fill_(random_mask < case.masked_ratio, -1)
            topk_weights.masked_fill_(topk_indices < 0, 0)
        del scores

        x_fp8, x_scale = per_token_cast_to_fp8(
            x_bf, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False
        )
        l1_fp8, l1_scale = _quantize_grouped_fp8_block_128_128(l1_bf)
        l2_fp8, l2_scale = _quantize_grouped_fp8_block_128_128(l2_bf)
        del x_bf, l1_bf, l2_bf
        transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
            (l1_fp8, l1_scale), (l2_fp8, l2_scale)
        )

        if persistent_l1 is None:
            persistent_l1 = tuple(transformed_l1)
            persistent_l2 = tuple(transformed_l2)
        else:
            assert persistent_l2 is not None
            _copy_tensor_tuple(persistent_l1, tuple(transformed_l1))
            _copy_tensor_tuple(persistent_l2, tuple(transformed_l2))
            del transformed_l1, transformed_l2

        if not args.pytorch_reference:
            del l1_fp8, l1_scale, l2_fp8, l2_scale

        assert persistent_l1 is not None and persistent_l2 is not None
        graph_key = (case.num_tokens, case.masked_ratio > 0)
        if graph_key not in graphs:
            candidate_output = torch.empty(
                (case.num_tokens, hidden), dtype=torch.bfloat16, device="cuda"
            )
            with _masked_hint_scope(case.masked_ratio > 0):
                graph = deep_gemm.SM90FP8MegaMoEGreenCtxGraph(
                    candidate_output,
                    persistent_l1,
                    persistent_l2,
                    buffer,
                    cumulative_local_expert_recv_stats=cumulative_stats,
                    recipe=(128, 128, 128),
                    activation="swiglu",
                    activation_clamp=10.0,
                    fast_math=True,
                    kernel1_sms=args.kernel1_sms,
                    kernel2_sms=args.kernel2_sms,
                    reduce_sms=124,
                )
            graphs[graph_key] = graph
            candidate_outputs[graph_key] = candidate_output
            reference_outputs[graph_key] = torch.empty_like(candidate_output)

        graph = graphs[graph_key]
        candidate_output = candidate_outputs[graph_key]
        reference_output = reference_outputs[graph_key]

        def stage_inputs() -> None:
            buffer.buffer.zero_()
            cumulative_stats.zero_()
            buffer.x[: case.num_tokens].copy_(x_fp8)
            buffer.x_sf[: case.num_tokens].copy_(x_scale)
            buffer.topk_idx[: case.num_tokens].copy_(topk_indices)
            buffer.topk_weights[: case.num_tokens].copy_(topk_weights)

        stage_inputs()
        torch.cuda.synchronize()
        dist.barrier()
        with _masked_hint_scope(case.masked_ratio > 0):
            deep_gemm.fp8_mega_moe(
                reference_output,
                persistent_l1,
                persistent_l2,
                buffer,
                cumulative_local_expert_recv_stats=cumulative_stats,
                recipe=(128, 128, 128),
                activation="swiglu",
                activation_clamp=10.0,
                fast_math=True,
            )
        torch.cuda.synchronize()
        dist.barrier()

        stage_inputs()
        torch.cuda.synchronize()
        dist.barrier()
        with _masked_hint_scope(case.masked_ratio > 0):
            graph.replay()
        torch.cuda.synchronize()
        dist.barrier()

        metrics = _compute_global_metrics(candidate_output, reference_output)
        sequential_vs_pytorch = None
        candidate_vs_pytorch = None
        if args.pytorch_reference:
            pytorch_output = legacy_quantized_pytorch_reference(
                x_fp8,
                x_scale,
                topk_indices,
                topk_weights,
                l1_fp8,
                l1_scale,
                l2_fp8,
                l2_scale,
                group=dist.group.WORLD,
                num_experts=num_experts,
                activation_clamp=10.0,
            )
            torch.cuda.synchronize()
            dist.barrier()
            sequential_vs_pytorch = _compute_global_metrics(
                reference_output, pytorch_output
            )
            candidate_vs_pytorch = _compute_global_metrics(
                candidate_output, pytorch_output
            )
            del pytorch_output

        passed = (
            metrics["reference_nonfinite"] == 0
            and metrics["candidate_nonfinite"] == 0
            and float(metrics["golden_diff"]) < args.tolerance
        )
        record: Dict[str, object] = {
            "ordinal": ordinal,
            "total_cases": len(cases),
            "case": asdict(case),
            **metrics,
            "tolerance": args.tolerance,
            "passed": passed,
        }
        if args.pytorch_reference:
            assert sequential_vs_pytorch is not None
            assert candidate_vs_pytorch is not None
            record["pytorch_reference_report_only"] = True
            record["sequential_vs_pytorch"] = sequential_vs_pytorch
            record["candidate_vs_pytorch"] = candidate_vs_pytorch
        results.append(record)
        if rank_idx == 0:
            line = "FP8_ACCURACY_CASE " + json.dumps(record, sort_keys=True)
            print(line, flush=True)
            if output_file is not None:
                output_file.write(json.dumps(record, sort_keys=True) + "\n")

        del x_fp8, x_scale, topk_indices, topk_weights
        if args.pytorch_reference:
            del l1_fp8, l1_scale, l2_fp8, l2_scale
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        dist.barrier()

    diffs = [float(record["golden_diff"]) for record in results]
    ordered_worst = sorted(
        results, key=lambda record: float(record["golden_diff"]), reverse=True
    )
    failures = [record for record in results if not bool(record["passed"])]
    summary = {
        "num_cases": len(results),
        "num_passed": len(results) - len(failures),
        "num_failed": len(failures),
        "tolerance": args.tolerance,
        "golden_diff_p50": _python_percentile(diffs, 0.50),
        "golden_diff_p95": _python_percentile(diffs, 0.95),
        "golden_diff_p99": _python_percentile(diffs, 0.99),
        "golden_diff_max": max(diffs) if diffs else float("nan"),
        "worst_case_indices": [
            int(record["case"]["index"]) for record in ordered_worst[:5]
        ],
        "worst_case_labels": [
            str(record["case"]["label"]) for record in ordered_worst[:5]
        ],
        "passed": not failures,
    }
    if args.pytorch_reference:
        summary["pytorch_reference_report_only"] = True
        summary["sequential_vs_pytorch"] = _summarize_reported_comparison(
            results, "sequential_vs_pytorch"
        )
        summary["candidate_vs_pytorch"] = _summarize_reported_comparison(
            results, "candidate_vs_pytorch"
        )
    if rank_idx == 0:
        print("FP8_ACCURACY_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
        if output_file is not None:
            output_file.write(
                json.dumps({"summary": summary}, sort_keys=True) + "\n"
            )
            output_file.close()

    dist.barrier()
    graphs.clear()
    candidate_outputs.clear()
    reference_outputs.clear()
    persistent_l1 = None
    persistent_l2 = None
    gc.collect()
    torch.cuda.synchronize()
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()
    if failures:
        raise AssertionError(
            f"FP8 E5M2 accuracy suite failed {len(failures)}/{len(results)} cases"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate FP8 E5M2 K2-to-K3 MegaMoE contributions"
    )
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--local-rank-idx", type=int, default=None)
    parser.add_argument("--kernel1-sms", type=int, default=70)
    parser.add_argument("--kernel2-sms", type=int, default=62)
    parser.add_argument("--suite", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--case-indices",
        type=str,
        default="",
        help="Comma-separated full-matrix indices; overrides --suite",
    )
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--output-jsonl", type=str, default="")
    parser.add_argument(
        "--pytorch-reference",
        action="store_true",
        help=(
            "report sequential and candidate errors versus an independent "
            "legacy per-row/per-64 E4M3 PyTorch reference; report-only"
        ),
    )
    args = parser.parse_args()
    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    if args.kernel1_sms <= 1 or args.kernel2_sms <= 1:
        parser.error("--kernel1-sms and --kernel2-sms must be greater than one")
    if args.kernel1_sms % 2 or args.kernel2_sms % 2:
        parser.error("--kernel1-sms and --kernel2-sms must be even")

    if args.local_rank_idx is not None:
        _run_worker(args.local_rank_idx, args.num_processes, args)
    else:
        torch.multiprocessing.spawn(
            _run_worker, args=(args.num_processes, args), nprocs=args.num_processes
        )


if __name__ == "__main__":
    main()
