# H200 Green Context MegaMoE

This branch contains an exact-shape H200 optimization for the SM90 Green
Context MegaMoE graph. It is not a general H200 FP8 tuning policy.

## Target

- 8 H200 GPUs with full NVSwitch connectivity
- 7000 tokens per rank
- hidden size 3072
- intermediate hidden size 1536
- 128 experts, top-k 16
- K1/K2/K3 SM allocation 78/54/124
- K3 uses 384 threads

Two lossy numerical optimizations are independently opt-in:

- `DG_SM90_MOE_SCALE_FREE_L1_K2=1` stores the post-SwiGLU L1 activation as
  scale-free FP8 E5M2 and consumes it directly in K2.
- `DG_SM90_MOE_FP8_COMPRESSED_COMBINE=1` stores the K2-to-K3 contribution as
  unscaled FP8 E5M2; K3 accumulates in FP32 and writes BF16.

When both are enabled for the exact target with the 78/54/124 split, the
legacy scheduler uses a fixed experts-per-wave value of 4. An explicitly set
`DG_SM90_MOE_EXPERTS_PER_WAVE` remains a debugging override.

## Performance

Complete-graph latency is measured with CUDA events, reduced with `MAX` across
all eight ranks. Each reported run is the minimum of 20 trials after warmup
and uses an empty JIT cache.

The original `c4cb7e8` Green Context implementation measured 4.386 ms versus
its 4.720-ms sequential baseline and was bitwise identical.

For the optimized both-on path, three independent runs measured:

| Run | Latency (ms) | `golden_diff` |
| --- | ---: | ---: |
| 1 | 3.525792 | 0.0034818 |
| 2 | 3.535200 | 0.0034818 |
| 3 | 3.546592 | 0.0034818 |

The best result is 24.13% below the 4.6472-ms baseline measured in the same
optimization campaign, corresponding to a 1.318x speedup. The three minima
span 20.8 microseconds, or 0.59% of the best result.

## Accuracy

The E5M2 paths are not bitwise equivalent to the sequential SM90 path.
Correctness uses `golden_diff < 0.01` and requires finite output on every rank.

- K2-to-K3 E5M2 only: 112/112 cases passed; maximum
  `golden_diff=0.001402216799`.
- Both E5M2 paths: 112/112 cases passed; maximum
  `golden_diff=0.0035100201`.
- No reference or candidate result contained NaN or Inf.
- Against an independent PyTorch model of the legacy per-row/per-64 E4M3
  intermediate, the sequential path had maximum `golden_diff=1.3043175e-05`
  and the both-on path had maximum `golden_diff=0.0035101909`.

The standalone accuracy harness is
`tests/validate_mega_moe_fp8_combine_accuracy.py`.

## Benchmark

The retained configuration can be benchmarked with:

```bash
export DG_SM90_MOE_K3_THREADS=384
export DG_SM90_MOE_FORCE_BLOCK_M=192
export DG_SM90_MOE_FORCE_EPILOGUE_WG=3
export DG_SM90_MOE_L2_SPLIT_SFA_TMA=1
export DG_SM90_MOE_SCALE_FREE_L1_K2=1
export DG_SM90_MOE_FP8_COMPRESSED_COMBINE=1

python tests/bench_mega_moe_sm90_cuda_events.py \
  --num-processes 8 \
  --num-tests 20 \
  --batches 7000 \
  --num-max-tokens-per-rank 7000 \
  --hidden 3072 \
  --intermediate-hidden 1536 \
  --num-experts 128 \
  --num-topk 16 \
  --green-context \
  --kernel1-sms 78 \
  --kernel2-sms 54 \
  --reduce-sms 124
```
