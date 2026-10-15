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
