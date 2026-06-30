# H200 Green Context Performance

## Result

Measured on 8 x H200 NVLink GPUs with an FP8 MegaMoE workload. Latency is the
maximum across ranks from the same validation run.

| Implementation | Latency (ms) | Relative performance |
| --- | ---: | ---: |
| Green Context | 4.386 | 1.076x |
| Existing SM90 baseline | 4.720 | 1.000x |

Green Context reduced latency by 7.1%. Output matched the baseline bit for bit
with a maximum absolute difference of zero.
