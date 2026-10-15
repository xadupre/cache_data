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
| `--optimize-alpha`, `--no-optimize-alpha` | Enable or disable alpha optimization and routing errors; enabled by default |
| `--alpha-grid-size` | Coarse optimization grid size; default `101` |

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
over iterations and QMoEs and therefore lies between 0 and 256. All three
alpha computations are restricted to `0.5 < alpha <= 1`:

```bash
python study_moe_selected_experts.py \
  moe-no-offload-unbounded-expert-activation-correlations-selected-experts.zip
```

The function `optimize_popularity_alpha(S)` returns
`(alpha, mean_squared_error)`.
For this dataset and an all-zero initial popularity tensor, it gives:

```text
alpha = 0.910330054018
mean_squared_error = 6.78780329411
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
of the three alpha computations.

The first table uses `alpha = 0.910330054018`, optimized for mean squared
popularity error:

```text
alpha_objective,gpu_expert_count,all_prompts,first_2_prompts,last_5_prompts,random_1_last_5_prompts,random_2_last_5_prompts,random_3_last_5_prompts
squared_error,0,8,8,8,8,8,8
squared_error,1024,3.68853084236,3.56670480549,3.73584779706,7.21638017356,7.12787049399,7.16999332443
squared_error,2048,2.34730820252,2.27705949657,2.39001168224,6.36300066756,6.3545894526,6.41693925234
squared_error,3072,1.54232257351,1.50263157895,1.56744826435,5.54979973298,5.52671895861,5.60034212283
squared_error,4096,1.01185606898,0.95766590389,1.02489152203,4.80168558077,4.77712783712,4.82947263017
squared_error,5120,0.652619942516,0.602173913043,0.659429238985,3.99633678238,3.95002503338,4.00191922563
squared_error,6144,0.393560689808,0.409610983982,0.392832109479,3.20702603471,3.12550901202,3.20624165554
squared_error,7168,0.198524209595,0.28209382151,0.197046061415,2.41239986649,2.36003838451,2.4245493992
squared_error,8192,0.0844295821358,0.202688787185,0.078020694259,1.57272196262,1.52489152203,1.60231975968
squared_error,9216,0.0282611098828,0.124599542334,0.0166889185581,0.742531708945,0.773815086782,0.819851468625
squared_error,10240,0,0,0,0,0,0
```

`optimize_routing_alpha(S, gpu_expert_count=6144)` performs a second
optimization over all prompts. It finds
`alpha = 0.990556466934`, with routing error `0.353940968384` at `K=6144`.
The same output columns for this alpha are:

```text
alpha_objective,gpu_expert_count,all_prompts,first_2_prompts,last_5_prompts,random_1_last_5_prompts,random_2_last_5_prompts,random_3_last_5_prompts
routing_error_k6144,0,8,8,8,8,8,8
routing_error_k6144,1024,4.30809197435,3.9930778032,4.3516605474,7.21638017356,7.12787049399,7.16999332443
routing_error_k6144,2048,2.83971921291,2.54273455378,2.87661048064,6.36300066756,6.3545894526,6.41693925234
routing_error_k6144,3072,1.84904377625,1.61550343249,1.8784212283,5.54979973298,5.52671895861,5.60034212283
routing_error_k6144,4096,1.15437762547,1.01127002288,1.17813751669,4.80168558077,4.77712783712,4.82947263017
routing_error_k6144,5120,0.669489277029,0.625171624714,0.684521028037,3.99633678238,3.95002503338,4.00191922563
routing_error_k6144,6144,0.353940968384,0.39971395881,0.359654539386,3.20702603471,3.12550901202,3.20624165554
routing_error_k6144,7168,0.181251381826,0.280720823799,0.179247329773,2.41239986649,2.36003838451,2.4245493992
routing_error_k6144,8192,0.0804333407031,0.202688787185,0.0721211615487,1.57272196262,1.52489152203,1.60231975968
routing_error_k6144,9216,0.0279736900287,0.124599542334,0.0162550066756,0.742531708945,0.773815086782,0.819851468625
routing_error_k6144,10240,0,0,0,0,0,0
```

`optimize_regression_alpha(S)` fits the no-intercept linear model

```text
S(t) - S(t-1) = alpha * (P(t-1) - S(t-1))
```

and recomputes `P` until the fitted coefficient converges. It gives
`alpha = 0.719428897297`. The fit uses
`sklearn.linear_model.LinearRegression(fit_intercept=False)`. The
corresponding routing table is:

```text
alpha_objective,gpu_expert_count,all_prompts,first_2_prompts,last_5_prompts,random_1_last_5_prompts,random_2_last_5_prompts,random_3_last_5_prompts
linear_regression,0,8,8,8,8,8,8
linear_regression,1024,3.69957439752,3.59313501144,3.75934579439,7.21638017356,7.12787049399,7.16999332443
linear_regression,2048,2.42837718329,2.33655606407,2.48478805073,6.36300066756,6.3545894526,6.41693925234
linear_regression,3072,1.5625580367,1.53901601831,1.5909629506,5.54979973298,5.52671895861,5.60034212283
linear_regression,4096,1.02139619721,0.974256292906,1.03549732977,4.80168558077,4.77712783712,4.82947263017
linear_regression,5120,0.65532832191,0.601029748284,0.662216288385,3.99633678238,3.95002503338,4.00191922563
linear_regression,6144,0.395180190139,0.410411899314,0.394292389853,3.20702603471,3.12550901202,3.20624165554
linear_regression,7168,0.198944284767,0.281464530892,0.197479973298,2.41239986649,2.36003838451,2.4245493992
linear_regression,8192,0.0848386026973,0.202688787185,0.0785964619493,1.57272196262,1.52489152203,1.60231975968
linear_regression,9216,0.0282721644926,0.124599542334,0.0167056074766,0.742531708945,0.773815086782,0.819851468625
linear_regression,10240,0,0,0,0,0,0
```
