# Base LLM Evaluation

Evaluates **Qwen/Qwen2.5-1.5B-Instruct** (the base model, **no LoRA adapters**) on the same
test set and with the same metrics as `Fine-Tune/metrics.ipynb`, so the two sets of numbers
can be directly compared.

---

## Files

| File | Purpose |
|---|---|
| `base_model_eval.py` | **Primary** — headless Python script, run with `nohup` or directly |
| `requirements.txt` | Python dependencies (includes `kagglehub` for automatic data download) |

> **Data source:** the evaluation dataset is downloaded automatically from the Kaggle dataset
> [`abirashab/train-test-val`](https://www.kaggle.com/datasets/abirashab/train-test-val)
> via `kagglehub` on first run and cached locally.  
> Pass `--data-path /local/dir` to use files you already have on disk.

---

## Why a Separate Folder?

`Fine-Tune/metrics.ipynb` loads the fine-tuned model (LoRA adapters).  
This script loads the **raw base model** from HuggingFace and runs **vLLM offline batch
inference** — all prompts submitted in one `llm.generate()` call, processed in parallel
across all available GPUs.

| Aspect | `Fine-Tune/metrics.ipynb` | `base_model_eval.py` |
|---|---|---|
| Model | Fine-tuned (LoRA adapters) | Base model only |
| Inference | Transformers sequential loop | **vLLM offline batch** |
| Execution | Interactive Kaggle notebook | Headless script / `nohup` |
| Prompt format | Identical | Identical |
| Metrics | Identical | Identical |
| Output files | `evaluation_results.csv` | `base_model_evaluation_results.csv` |

---

## Hardware

Optimised for **4 × NVIDIA RTX Ada 6000 (48 GB each, 192 GB total VRAM)**.  
Works on any NVIDIA setup — adjust `--tensor-parallel-size` to match your GPU count.

| Setting | Value | Notes |
|---|---|---|
| `tensor_parallel_size` | `4` | One shard per GPU |
| `gpu_memory_utilization` | `0.90` | 90 % of each GPU's VRAM |
| `dtype` | `bfloat16` | Numerically stable; supported on Ada / Ampere |
| Swap space | 4 GB | CPU-RAM fallback for KV cache overflow |

---

## Installation

```bash
# Python 3.10+ recommended (vLLM requirement)
pip install -r requirements.txt
```

`requirements.txt` pins the packages needed. vLLM wheels are published for CUDA 12.x;
if your driver uses CUDA 11.8 install the matching vLLM wheel instead:

```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118
```

### Kaggle credentials (required for automatic data download)

`kagglehub` reads your Kaggle API key from `~/.kaggle/kaggle.json`.  
Create one at **kaggle.com → Settings → API → Create New Token**, then:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

If you already have the JSONL files locally, skip this step and pass `--data-path`.

---

## Running the Script

### Foreground (see live output)

```bash
python base_model_eval.py
```

### Background with `nohup` (recommended for long runs)

```bash
nohup python base_model_eval.py > eval.log 2>&1 &
echo "PID: $!"

# Follow progress live:
tail -f eval.log
# or the dedicated rotating log file:
tail -f base_eval_run.log
```

### CLI overrides (no need to edit the file)

```bash
# Default — downloads test split from Kaggle, evaluates 2 000 samples
python base_model_eval.py

# Full test set, all 4 GPUs
python base_model_eval.py --eval-limit None --tensor-parallel-size 4

# Evaluate on ALL splits (train + val + test) combined
python base_model_eval.py --splits train val test --eval-limit None

# Evaluate on val split only
python base_model_eval.py --splits val

# Use locally cached data instead of downloading from Kaggle
python base_model_eval.py --data-path /path/to/train-test-val

# Quick smoke-test on 500 samples using 2 GPUs
python base_model_eval.py --eval-limit 500 --tensor-parallel-size 2

# Custom output directory; skip saving CSVs
python base_model_eval.py --output-dir ./results --no-save
```

### All available flags

| Flag | Default | Description |
|---|---|---|
| `--model-name` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model ID |
| `--data-path` | *(Kaggle download)* | Local directory with JSONL splits — skips Kaggle download |
| `--splits` | `test` | One or more of `train val test` to evaluate |
| `--eval-limit` | `2000` | Number of samples; `None` = full set |
| `--suspicious-ratio` | `0.30` | Fraction of suspicious samples |
| `--max-new-tokens` | `512` | Max tokens to generate per response |
| `--temperature` | `0.7` | Sampling temperature |
| `--tensor-parallel-size` | `4` | GPUs used for tensor parallelism |
| `--gpu-memory-utilization` | `0.90` | Fraction of VRAM vLLM may use per GPU |
| `--dtype` | `bfloat16` | `float16`, `bfloat16`, or `float32` |
| `--output-dir` | `.` | Where to write all output files |
| `--no-save` | off | Disable CSV saving |

---

## Configuration (in-file)

All defaults live in the `CONFIG` dict at the top of `base_model_eval.py`.
Edit that dict to permanently change any setting without CLI flags.

---

## Pipeline Steps

The script runs these steps sequentially and logs each one with a timestamp:

| Step | What happens |
|---|---|
| 1 | Initialise `vllm.LLM` with `Qwen/Qwen2.5-1.5B-Instruct` across `tensor_parallel_size` GPUs |
| 2 | Download `abirashab/train-test-val` via `kagglehub` (cached after first run), or read from `--data-path`; load and concatenate all requested `--splits` |
| 3 | Stratified sampling (identical logic to `Fine-Tune/metrics.ipynb`) |
| 4 | Build prompt list — same format as training data |
| 5 | **Single-call batch inference** via `llm.generate()` (vLLM handles continuous batching) |
| 6 | Compute per-example metrics (exact match, partial match, word-level F1) |
| 7 | Compute classification metrics (accuracy, precision/recall/F1 macro+weighted) |
| 8 | Save all outputs |

---

## Output Files

All files are written to `--output-dir` (default: current directory).

| File | Description |
|---|---|
| `base_eval_run.log` | Timestamped log of the entire run (same content as stdout) |
| `base_model_evaluation_results.csv` | Per-example: input, expected, predicted, labels, scores |
| `base_model_metrics_summary.csv` | One-row summary of all scalar metrics |
| `base_model_metrics.png` | Bar chart — overall + macro vs weighted |
| `base_model_confusion_matrix.png` | Row-normalised confusion matrix |

---

## Metrics Computed

All metrics are identical to `Fine-Tune/metrics.ipynb`:

**Classification (Normal vs Suspicious)**
- Accuracy
- Precision / Recall / F1-Score — Macro & Weighted
- Per-class `sklearn` classification report
- Confusion matrix (row-normalised, with raw counts)

**Output quality**
- Exact Match Accuracy
- Average Partial Match (word-overlap ratio)
- Average Word-level F1

---

## Comparing Base vs Fine-Tuned

After both evaluations are done, compare the metric CSVs:

```python
import pandas as pd

base  = pd.read_csv("Base-LLM-Evaluation/base_model_metrics_summary.csv")
tuned = pd.read_csv("Fine-Tune/metrics_summary.csv")   # adjust path as needed

comp = base.merge(tuned, on="Metric", suffixes=("_base", "_finetuned"))
print(comp.to_string(index=False))
```

---

## Troubleshooting

**`ImportError: No module named 'vllm'`**  
Run `pip install -r requirements.txt`. vLLM requires Linux + CUDA; it does **not** run on macOS.

**CUDA out of memory during `LLM()` init**  
Reduce `--gpu-memory-utilization 0.80` or lower `--tensor-parallel-size`.  
A 1.5 B model in bfloat16 uses ~3 GB; OOM here usually means another process is already holding VRAM.

**All predictions are `UNKNOWN`**  
The base model may not follow the training-data output format.  
Check `base_eval_run.log` for sample predictions, then update `extract_status_label()`
in the script to match the actual output style.

**Kaggle download fails / `401 Unauthorized`**  
Your `~/.kaggle/kaggle.json` is missing or invalid.  
Generate a fresh token at kaggle.com → Settings → API → Create New Token, then `chmod 600 ~/.kaggle/kaggle.json`.  
Alternatively, pass `--data-path` to point directly at a local copy of the JSONL files.

**Split file not found warning**  
If `kagglehub` downloaded the dataset but a split file is missing (e.g. no `train.jsonl`),
the script skips that split with a warning and continues with the remaining ones.
