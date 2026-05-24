# Fine-Tune

LoRA fine-tuning and evaluation for MITRE ATT&CK log analysis.  
Supports three models; each gets its own output folder for side-by-side comparison.

**Supported models** (select with `--model`):

| Alias | Model | VRAM (bfloat16) |
|---|---|---|
| `qwen` | `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB |
| `llama` | `meta-llama/Llama-3.2-3B-Instruct` | ~6 GB |
| `phi` | `microsoft/Phi-4-mini-instruct` | ~8 GB |

Each model uses its **native chat template** during training and evaluation so the prompts
the model was trained on exactly match the prompts used at inference time.

---

## Files

| File | Purpose |
|---|---|
| `fine_tune.py` | **LoRA fine-tuning** — headless Python script, run with `nohup` or directly |
| `metrics.py` | **Fine-tuned model evaluation** — Transformers inference, same metrics as `base_model_eval.py` |
| `requirements.txt` | Python dependencies |
| `fine-tune.ipynb` | Kaggle notebook (Qwen only) — kept as reference; **not the primary script** |
| `metrics.ipynb` | Kaggle notebook (Qwen only) — kept as reference; **not the primary script** |

> **Data source:** both scripts download `abirashab/train-test-val` from Kaggle automatically
> via `kagglehub`. Requires `~/.kaggle/kaggle.json` — or pass `--data-path` for local files.

---

## Installation

```bash
# Python 3.10+, Linux, CUDA 12.x
pip install -r requirements.txt
```

### Kaggle credentials

`kagglehub` reads your API key from `~/.kaggle/kaggle.json`.  
Generate one at **kaggle.com → Settings → API → Create New Token**, then:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

---

## Fine-Tuning (`fine_tune.py`)

### Quick start

```bash
# Llama-3.2-3B (downloads data, saves to models/Llama-3.2-3B/)
python fine_tune.py --model llama

# Phi-4-mini, full dataset
python fine_tune.py --model phi --sample-percentage 1.0

# Background run
nohup python fine_tune.py --model llama > checkpoints/Llama-3.2-3B/train.log 2>&1 &
```

### What it does

1. Downloads `abirashab/train-test-val` from Kaggle (or reads from `--data-path`)
2. Loads the base model and tokenizer
3. Applies LoRA (r=16, alpha=32, targets: `q_proj k_proj v_proj o_proj`)
4. Tokenises examples using the model's native chat template; masks prompt tokens with `-100` so loss is computed on output tokens only
5. Trains with `Trainer` + `EarlyStoppingCallback`; auto-resumes from latest checkpoint
6. Saves final model to `models/{ModelName}/`
7. Runs a single-sample smoke-test inference

### LoRA config (all models)

| Parameter | Value |
|---|---|
| r | 16 |
| lora_alpha | 32 |
| lora_dropout | 0.05 |
| target_modules | q\_proj, k\_proj, v\_proj, o\_proj |
| task_type | CAUSAL_LM |

### Training hyperparameters (default, matches `fine-tune.ipynb`)

| Parameter | Value |
|---|---|
| sample_percentage | 0.4 (40 % of train split) |
| batch_size | 1 |
| gradient_accumulation_steps | 4 (effective batch = 4) |
| epochs | 3 |
| learning_rate | 2e-4 |
| lr_scheduler | cosine |
| warmup_ratio | 0.1 |
| weight_decay | 0.01 |
| optimizer | AdamW |
| fp16 | True |
| gradient_checkpointing | False |
| early_stopping_patience | 5 |
| max_length | 2000 tokens |

### Output layout

```
Fine-Tune/
├── checkpoints/
│   ├── Qwen2.5-1.5B/    # training checkpoints
│   ├── Llama-3.2-3B/
│   └── Phi-4-mini/
└── models/
    ├── Qwen2.5-1.5B/    # final model / LoRA adapters + fine_tune_run.log
    ├── Llama-3.2-3B/
    └── Phi-4-mini/
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model` | required | `qwen` \| `llama` \| `phi` |
| `--data-path` | *(Kaggle download)* | Local dir with `train.jsonl` / `val.jsonl` |
| `--output-dir` | `checkpoints/{Model}` | Checkpoint directory |
| `--final-model-dir` | `models/{Model}` | Where to save the final model |
| `--epochs` | `3` | Training epochs |
| `--lr` | `2e-4` | Learning rate |
| `--sample-percentage` | `0.4` | Fraction of training data to use |
| `--lora-r` | `16` | LoRA rank |
| `--no-resume` | off | Start fresh, ignore existing checkpoints |

---

## Evaluation (`metrics.py`)

Evaluates the fine-tuned model on the test split. Uses Transformers (not vLLM) so PEFT
LoRA adapters can be loaded directly without merging weights.

### Quick start

```bash
# Evaluate fine-tuned Llama
python metrics.py --model llama --model-path models/Llama-3.2-3B

# Full test set
python metrics.py --model phi --model-path models/Phi-4-mini --eval-limit None

# Background run
nohup python metrics.py --model llama --model-path models/Llama-3.2-3B \
    --eval-limit None > results/Llama-3.2-3B/metrics.log 2>&1 &

# Smoke-test (50 samples)
python metrics.py --model llama --model-path models/Llama-3.2-3B \
    --eval-limit 50 --no-save
```

### Output layout

```
Fine-Tune/results/
├── Qwen2.5-1.5B/
│   ├── metrics_run.log
│   ├── evaluation_results.csv       # per-example predictions + scores
│   ├── metrics_summary.csv          # scalar metrics (matches base_model_eval format)
│   ├── metrics.png                  # overall + macro vs weighted bar chart
│   └── confusion_matrix.png         # row-normalised confusion matrix
├── Llama-3.2-3B/
└── Phi-4-mini/
```

### Metrics computed (identical to `Base-LLM-Evaluation/base_model_eval.py`)

**Classification (Normal vs Suspicious)**
- Accuracy
- Precision / Recall / F1-Score — Macro & Weighted
- Per-class `sklearn` classification report
- Confusion matrix (row-normalised, with raw counts)

**Output quality**
- Exact Match Accuracy
- Average Partial Match (word-overlap ratio)
- Average Word-level F1

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model` | required | `qwen` \| `llama` \| `phi` |
| `--model-path` | required | Path to fine-tuned model / LoRA adapters |
| `--data-path` | *(Kaggle download)* | Local dir with JSONL splits |
| `--splits` | `test` | One or more of `train val test` |
| `--eval-limit` | `2000` | Sample cap; `None` = full set |
| `--suspicious-ratio` | `0.30` | Fraction of suspicious samples |
| `--output-dir` | `results/{Model}` | Override the auto-named output folder |
| `--no-save` | off | Print metrics only, skip CSV/PNG output |

---

## Chat Templates

Each model's native format is applied consistently in `fine_tune.py` and `metrics.py`:

| Model | Training & inference format |
|---|---|
| **Qwen** | `{instruction}\n\n{input}\n\nAnalysis:\n{output}` (plain text) |
| **Llama** | `<\|begin_of_text\|><\|start_header_id\|>user<\|end_header_id\|>\n\n{instruction}\n\n{input}\n\nAnalysis:<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>\n\n{output}<\|eot_id\|>` |
| **Phi** | `<\|user\|>\n{instruction}\n\n{input}\n\nAnalysis:<\|end\|>\n<\|assistant\|>\n{output}<\|end\|>` |

---

## Comparing Base vs Fine-Tuned

After running `base_model_eval.py` and `metrics.py` for a model:

```python
import pandas as pd

model = "Llama-3.2-3B"   # or Qwen2.5-1.5B, Phi-4-mini

base  = pd.read_csv(f"../Base-LLM-Evaluation/results/{model}/base_model_metrics_summary.csv")
tuned = pd.read_csv(f"results/{model}/metrics_summary.csv")

comp = base.merge(tuned, on="Metric", suffixes=("_base", "_finetuned"))
print(comp.to_string(index=False))
```

---

## Troubleshooting

**CUDA out of memory during training**  
Lower `--lora-r` (e.g. 8) or reduce `max_length` in `CONFIG`. Llama (~6 GB) and Phi (~8 GB)
require more VRAM than Qwen (~3 GB). `gradient_checkpointing` is intentionally disabled
(causes CUDA errors with `max_length=2000`).

**`401 Unauthorized` from kagglehub**  
Your `~/.kaggle/kaggle.json` is missing or stale. Generate a fresh token at
kaggle.com → Settings → API → Create New Token, then `chmod 600 ~/.kaggle/kaggle.json`.
Or pass `--data-path` to skip the download.

**All predictions are `UNKNOWN`**  
The model is not following the expected output format. Check `metrics_run.log` for sample
predictions. This often happens when `--model` does not match the model the weights were
actually trained for (e.g. passing `--model qwen` but pointing `--model-path` at Llama weights).
