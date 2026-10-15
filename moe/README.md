# MoE routing data

This directory contains routing data collected from the 40 QMoE layers of the
Qwen 3.5 35B CUDA FP16 model.

## Selected experts

[`moe-no-offload-unbounded-expert-activation-correlations-selected-experts.zip`](moe-no-offload-unbounded-expert-activation-correlations-selected-experts.zip)
contains one CSV file with 1,447,360 selected-expert records:

| Column | Description |
|---|---|
| `iteration` | One-based decode iteration, accumulated independently for each QMoE across all prompts |
| `qmoe_number` | Zero-based QMoE number after sorting nodes by model layer |
| `expert_number` | Zero-based expert number local to the QMoE |
| `position` | Zero-based global expert position before the popularity-counter update |

The first lines of the uncompressed CSV are:

```csv
iteration,qmoe_number,expert_number,position
1,0,236,112
1,0,43,12
1,0,30,7
1,0,191,95
1,0,124,2850
1,0,73,2827
1,0,188,94
1,0,238,114
1,1,33,18
1,1,86,31
```

The trace contains 10 prompts and 4,533 generated tokens. The first generated
token of each prompt comes from prefill, so the CSV covers 4,523 decode
iterations. Every decode iteration selects 8 experts in each of the 40 QMoE
layers:

```text
4,523 iterations * 40 QMoE * 8 experts = 1,447,360 rows
```

The token counts per prompt are:

```text
217, 222, 254, 431, 408, 665, 451, 501, 647, 737
```

Generation used ONNX Runtime PR
[`microsoft/onnxruntime#33347`](https://github.com/microsoft/onnxruntime/pull/33347)
at commit `cc5debcb18`, with no CPU-offloaded experts, popularity coefficient
`alpha=0.9`, greedy decoding, and `--until-eos`. There was no artificial
new-token limit; the model context length of 262,144 tokens remained the hard
limit.

The generation command was equivalent to:

```bash
ORT_ENABLE_CUDNN_FLASH_ATTENTION=0 \
PYTHONPATH=<onnxruntime-cuda-build> \
python tools/python/qmoe_prompt_runner.py \
  <qwen-3.5-35b-cuda-fp16-model> \
  --prompts-file <ten-prompts.json> \
  --provider cuda \
  --until-eos \
  --moe-cpu-offload-experts 0 \
  --moe-expert-counter-alpha 0.9 \
  --output moe-no-offload-unbounded.json \
  --counter-log moe-no-offload-unbounded.log
```

The CSV was exported from the resulting counter trace with:

```bash
PYTHONPATH=tools/python \
python tools/python/qmoe_expert_activation_correlations.py \
  moe-no-offload-unbounded.log \
  --output-prefix moe-no-offload-unbounded-expert-activation-correlations
```

Archive SHA-256:

```text
135f4be1e915ed4f6819494b4bb6ade90cdba64b609dbbe53b7f486bc5182f37
```

## Binary selection tensor

[`study_moe_selected_experts.py`](study_moe_selected_experts.py) reads the
ZIP archive directly with pandas and creates the binary tensor
`S[t, m, i]`. It is `1` when expert `i` is selected by QMoE `m` at decode
iteration `t`, and `0` otherwise:

| Argument | Description |
|---|---|
| `archive` | ZIP archive containing the four-column routing CSV |
| `--qmoe-count` | Expected QMoE count; default `40` |
| `--expert-count` | Experts per QMoE; default `256` |
| `--selected-expert-count` | Selected experts per iteration and QMoE; default `8` |
| `--optimization` | `alpha`, `matrix`, or `none`; default `alpha` |
| `--alpha-grid-size` | Coarse optimization grid size; default `101` |
| `--matrix-epochs` | PyTorch training epochs for `W0`, `W1`, and `W2`; default `100` |
| `--matrix-learning-rate` | Adam learning rate for matrix regression; default `0.005` |
| `--markdown-output` | Markdown result path; default `study_moe_selected_experts.md` beside the script |

Run `python study_moe_selected_experts.py --help` to display these options and
complete command-line examples.

```bash
python study_moe_selected_experts.py \
  moe-no-offload-unbounded-expert-activation-correlations-selected-experts.zip
```

The resulting tensor has shape `(4523, 40, 256)`, dtype `uint8`, and exactly
8 ones along the expert axis for every `(t, m)` pair. It remains in memory and
is not written to disk:

```python
from study_moe_selected_experts import build_selection_tensor, read_selected_experts

selected_experts = read_selected_experts(
    "moe-no-offload-unbounded-expert-activation-correlations-selected-experts.zip"
)
S = build_selection_tensor(selected_experts)
```

By default, the script optimizes the popularity coefficient in
`P(t) = alpha * P(t-1) + (1-alpha) * S(t-1)` by minimizing
`mean(t,m) sum(i) (P(t,m,i) - S(t,m,i))**2`. This error is averaged
over iterations and QMoEs and therefore lies between 0 and 256. The three
scalar-alpha computations are restricted to `0.5 < alpha <= 1`:

The `alpha` mode runs the three scalar-alpha methods. The `matrix` mode runs
only the joint W0/W1/W2 optimization; the two modes are mutually exclusive.

```bash
python study_moe_selected_experts.py \
  moe-no-offload-unbounded-expert-activation-correlations-selected-experts.zip
```

The function `optimize_popularity_alpha(S)` returns
`(alpha, mean_squared_error)`.
For this dataset and an all-zero initial popularity tensor, it gives:

```text
alpha = 0.910
mean_squared_error = 6.788
```

The script also simulates GPU expert retention. At every iteration, it sorts
all 10,240 values in `P[t, :, :]` in increasing order, retains the last `K`
experts (the highest popularities) on GPU, and builds the binary prediction
`Q[t, :, :]`. Thus, `K` is the number of experts kept on GPU.
`compute_popularity(S, alpha)` constructs the complete popularity tensor `P`.
`popularity_error(S, P)` computes the mean quadratic objective used to
optimize `alpha`. `moe_routing_error(S, P, K)` then returns
`mean(t,m) sum(i) S(t,m,i) * |Q(t,m,i)-S(t,m,i)|`. The error therefore
counts only selected experts that were not predicted and lies between 0 and
8.
The script evaluates this second error at the same `alpha` obtained by
minimizing the quadratic popularity error; it does not optimize a second
alpha. It reports this routing error for `K = 1024 * n`, with
`n = 0, 1, ..., 10`, over all ten prompts, the first two prompts, and the last
five prompts. The popularity history preceding each reported interval is
retained. The ten prompts contain respectively
`216, 221, 253, 430, 407, 664, 450, 500, 646, 736` decode iterations.

The three random baselines each draw one fixed uniform random popularity
distribution over the 10,240 experts and reuse it for every iteration. They
apply the same global sort and values of `K`, and their errors are evaluated
only on the last five prompts. NumPy seeds `0`, `1`, and `2` make the three
columns reproducible. Each fixed distribution has shape `(40, 256)`;
`moe_routing_error` accepts it directly without expanding it over iterations.
The script runs the linear-regression estimate first because it is the fastest
of the three scalar-alpha computations. CSV tables remain on standard output;
GitHub-flavored Markdown is written to `study_moe_selected_experts.md` beside
the Python script.

The first table uses `alpha = 0.910`, optimized for mean squared
popularity error:

```text
alpha_objective,gpu_expert_count,all_prompts,first_2_prompts,last_5_prompts,random_1_last_5_prompts,random_2_last_5_prompts,random_3_last_5_prompts
squared_error,0,8.000,8.000,8.000,8.000,8.000,8.000
squared_error,1024,3.689,3.567,3.736,7.216,7.128,7.170
squared_error,2048,2.347,2.277,2.390,6.363,6.355,6.417
squared_error,3072,1.542,1.503,1.567,5.550,5.527,5.600
squared_error,4096,1.012,0.958,1.025,4.802,4.777,4.829
squared_error,5120,0.653,0.602,0.659,3.996,3.950,4.002
squared_error,6144,0.394,0.410,0.393,3.207,3.126,3.206
squared_error,7168,0.199,0.282,0.197,2.412,2.360,2.425
squared_error,8192,0.084,0.203,0.078,1.573,1.525,1.602
squared_error,9216,0.028,0.125,0.017,0.743,0.774,0.820
squared_error,10240,0.000,0.000,0.000,0.000,0.000,0.000
```

`optimize_routing_alpha(S, gpu_expert_count=6144)` performs a second
optimization over all prompts. It finds
`alpha = 0.991`, with routing error `0.354` at `K=6144`.
The same output columns for this alpha are:

```text
alpha_objective,gpu_expert_count,all_prompts,first_2_prompts,last_5_prompts,random_1_last_5_prompts,random_2_last_5_prompts,random_3_last_5_prompts
routing_error_k6144,0,8.000,8.000,8.000,8.000,8.000,8.000
routing_error_k6144,1024,4.308,3.993,4.352,7.216,7.128,7.170
routing_error_k6144,2048,2.840,2.543,2.877,6.363,6.355,6.417
routing_error_k6144,3072,1.849,1.616,1.878,5.550,5.527,5.600
routing_error_k6144,4096,1.154,1.011,1.178,4.802,4.777,4.829
routing_error_k6144,5120,0.669,0.625,0.685,3.996,3.950,4.002
routing_error_k6144,6144,0.354,0.400,0.360,3.207,3.126,3.206
routing_error_k6144,7168,0.181,0.281,0.179,2.412,2.360,2.425
routing_error_k6144,8192,0.080,0.203,0.072,1.573,1.525,1.602
routing_error_k6144,9216,0.028,0.125,0.016,0.743,0.774,0.820
routing_error_k6144,10240,0.000,0.000,0.000,0.000,0.000,0.000
```

`optimize_regression_alpha(S)` fits the no-intercept linear model

```text
S(t) - S(t-1) = alpha * (P(t-1) - S(t-1))
```

and recomputes `P` until the fitted coefficient converges. It gives
`alpha = 0.719`. The fit uses
`sklearn.linear_model.LinearRegression(fit_intercept=False)`. The
corresponding routing table is:

```text
alpha_objective,gpu_expert_count,all_prompts,first_2_prompts,last_5_prompts,random_1_last_5_prompts,random_2_last_5_prompts,random_3_last_5_prompts
linear_regression,0,8.000,8.000,8.000,8.000,8.000,8.000
linear_regression,1024,3.700,3.593,3.759,7.216,7.128,7.170
linear_regression,2048,2.428,2.337,2.485,6.363,6.355,6.417
linear_regression,3072,1.563,1.539,1.591,5.550,5.527,5.600
linear_regression,4096,1.021,0.974,1.035,4.802,4.777,4.829
linear_regression,5120,0.655,0.601,0.662,3.996,3.950,4.002
linear_regression,6144,0.395,0.410,0.394,3.207,3.126,3.206
linear_regression,7168,0.199,0.281,0.197,2.412,2.360,2.425
linear_regression,8192,0.085,0.203,0.079,1.573,1.525,1.602
linear_regression,9216,0.028,0.125,0.017,0.743,0.774,0.820
linear_regression,10240,0.000,0.000,0.000,0.000,0.000,0.000
```

The matrix regression uses PyTorch and learns three `256 x 256` matrices per
QMoE:

```bash
python study_moe_selected_experts.py \
  moe-no-offload-unbounded-expert-activation-correlations-selected-experts.zip \
  --optimization matrix
```

```text
P[t,m] = W0[m] @ P[t-1,m]
       + W1[m] @ S[t-1,m]
       + W2[m] @ S[t,m-1]
```

`P[-1,m]`, `S[-1,m]`, and `S[t,-1]` are zero. Full-batch Adam starts from
`W0[m] = 0.719 I`, `W1[m] = 0.281 I`, and `W2[m] = 0`. After every Adam
step, `W0` is projected to an induced infinity norm below one to keep the
recurrence stable. After 100 epochs with learning rate `0.005`, it gives:

```text
matrix_regression_mean_squared_error = 3.959
W0.shape = W1.shape = W2.shape = (40, 256, 256)
```

The routing figures are in-sample: all prompts used for the reported columns
also participate in matrix training.

```text
alpha_objective,gpu_expert_count,all_prompts,first_2_prompts,last_5_prompts,random_1_last_5_prompts,random_2_last_5_prompts,random_3_last_5_prompts
matrix_regression,0,8.000,8.000,8.000,8.000,8.000,8.000
matrix_regression,1024,0.380,0.414,0.378,7.216,7.128,7.170
matrix_regression,2048,0.084,0.092,0.083,6.363,6.355,6.417
matrix_regression,3072,0.038,0.044,0.037,5.550,5.527,5.600
matrix_regression,4096,0.024,0.028,0.023,4.802,4.777,4.829
matrix_regression,5120,0.018,0.021,0.018,3.996,3.950,4.002
matrix_regression,6144,0.016,0.019,0.015,3.207,3.126,3.206
matrix_regression,7168,0.013,0.016,0.012,2.412,2.360,2.425
matrix_regression,8192,0.009,0.012,0.008,1.573,1.525,1.602
matrix_regression,9216,0.004,0.005,0.004,0.743,0.774,0.820
matrix_regression,10240,0.000,0.000,0.000,0.000,0.000,0.000
```
