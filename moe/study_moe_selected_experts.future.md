# QMoE routing results

## Configuration

- Optimization: `future`
- Device: `cuda`
- Maximum epochs: `200`
- Learning rate: `0.005`
- Retention lambda: `100.0`
- Matrix coefficients: non-negative
- Early-stopping relative tolerance: `0.0001`
- Early-stopping patience: `10`

```text
P[t,m] = W0[m] @ P[t-1,m]
       + W1[m] @ S[t-1,m]

loss = mean(t,m) sum(i) (P[t,m,i] - S[t,m,i])**2
     + mean(t,m) sum(i,k=-16..-1,1..16) 100.0/(|k|+100.0) * S[t-k,m,i] * (P[t,m,i] - S[t-k,m,i])**2
```

## Short matrix regression with bidirectional retention penalty

- `future_mean_squared_error = 39.807`
- `future_retention_error = 33.145`
- `future_total_loss = 72.952`
- `future_w0_diagonal_mass_ratio = 0.094`
- `future_w0_diagonal_mean = 0.041`
- `future_w0_off_diagonal_mean = 0.002`
- `future_w0_diagonal_row_max_ratio = 0.636`
- `expert_swaps_k6144_first_5_mean = 508.486`
- `expert_swaps_k6144_first_5_median = 495.000`
- `expert_swaps_k6144_first_5_min = 246.000`
- `expert_swaps_k6144_first_5_max = 2486.000`
- `expert_swaps_k6144_last_5_mean = 539.721`
- `expert_swaps_k6144_last_5_median = 530.000`
- `expert_swaps_k6144_last_5_min = 313.000`
- `expert_swaps_k6144_last_5_max = 942.000`

**Routing results: `future`**

| Experts on GPU | All prompts | First 2 prompts | Last 5 prompts | Mean swaps, first 5 | Mean swaps, last 5 | Required swaps, last 5 | Random mean, last 5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8.000 | 8.000 | 8.000 | 0.000 | 0.000 | nan | 8.000 |
| 1024 | 2.824 | 2.764 | 2.877 | 288.158 | 303.303 | 133.357 | 7.171 |
| 2048 | 1.538 | 1.510 | 1.575 | 432.382 | 451.915 | 79.253 | 6.378 |
| 3072 | 0.868 | 0.857 | 0.897 | 512.718 | 534.736 | 49.157 | 5.559 |
| 4096 | 0.484 | 0.486 | 0.502 | 546.171 | 572.896 | 30.491 | 4.803 |
| 5120 | 0.260 | 0.261 | 0.272 | 543.406 | 573.732 | 18.372 | 3.983 |
| 6144 | 0.127 | 0.130 | 0.134 | 508.488 | 539.721 | 10.150 | 3.180 |
| 7168 | 0.051 | 0.057 | 0.054 | 451.695 | 476.794 | 4.928 | 2.399 |
| 8192 | 0.015 | 0.022 | 0.016 | 374.638 | 387.375 | 2.019 | 1.567 |
| 9216 | 0.002 | 0.005 | 0.002 | 215.721 | 220.019 | 0.494 | 0.779 |
| 10240 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**Routing results with at most 2 swaps: `future`**

| Experts on GPU | All prompts | First 2 prompts | Last 5 prompts | Mean swaps, first 5 | Mean swaps, last 5 | Required swaps, last 5 | Random mean, last 5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8.000 | 8.000 | 8.000 | 0.000 | 0.000 | nan | 8.000 |
| 1024 | 4.959 | 6.137 | 4.770 | 2.000 | 2.000 | 133.357 | 7.171 |
| 2048 | 3.656 | 5.173 | 3.412 | 2.000 | 2.000 | 79.253 | 6.378 |
| 3072 | 2.631 | 4.255 | 2.373 | 2.000 | 2.000 | 49.157 | 5.559 |
| 4096 | 1.792 | 3.437 | 1.539 | 2.000 | 2.000 | 30.491 | 4.803 |
| 5120 | 1.124 | 2.650 | 0.889 | 2.000 | 2.000 | 18.372 | 3.983 |
| 6144 | 0.611 | 1.917 | 0.420 | 2.000 | 2.000 | 10.150 | 3.180 |
| 7168 | 0.283 | 1.249 | 0.158 | 2.000 | 2.000 | 4.928 | 2.399 |
| 8192 | 0.106 | 0.662 | 0.043 | 2.000 | 2.000 | 2.019 | 1.567 |
| 9216 | 0.021 | 0.166 | 0.006 | 2.000 | 2.000 | 0.494 | 0.779 |
| 10240 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**Routing results with at most 6 swaps: `future`**

| Experts on GPU | All prompts | First 2 prompts | Last 5 prompts | Mean swaps, first 5 | Mean swaps, last 5 | Required swaps, last 5 | Random mean, last 5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8.000 | 8.000 | 8.000 | 0.000 | 0.000 | nan | 8.000 |
| 1024 | 4.315 | 4.544 | 4.301 | 6.000 | 6.000 | 133.357 | 7.171 |
| 2048 | 2.983 | 3.441 | 2.940 | 6.000 | 6.000 | 79.253 | 6.378 |
| 3072 | 2.030 | 2.576 | 1.980 | 6.000 | 6.000 | 49.157 | 5.559 |
| 4096 | 1.289 | 1.874 | 1.232 | 6.000 | 6.000 | 30.491 | 4.803 |
| 5120 | 0.728 | 1.307 | 0.673 | 6.000 | 6.000 | 18.372 | 3.983 |
| 6144 | 0.346 | 0.826 | 0.300 | 6.000 | 6.000 | 10.150 | 3.180 |
| 7168 | 0.137 | 0.454 | 0.105 | 6.000 | 6.000 | 4.928 | 2.399 |
| 8192 | 0.045 | 0.215 | 0.027 | 6.000 | 6.000 | 2.019 | 1.567 |
| 9216 | 0.007 | 0.054 | 0.002 | 6.000 | 6.000 | 0.494 | 0.779 |
| 10240 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

