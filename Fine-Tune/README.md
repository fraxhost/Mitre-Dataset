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
| `compare_results.py` | **Multi-model comparison** — auto-discovers all result folders and produces side-by-side tables, bar charts, and confusion matrix grids |
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
│   ├── confusion_matrix.png         # row-normalised confusion matrix
│   └── eval_checkpoint.json         # transient — present only during a run; auto-deleted on success
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
| `--no-resume` | off | Ignore any existing checkpoint; start from scratch |
| `--checkpoint-interval` | `50` | Save a checkpoint every N samples |

### Crash recovery / resume

`metrics.py` saves a checkpoint to `results/{Model}/eval_checkpoint.json` every 50 samples
(configurable). If the process is interrupted, simply re-run the same command — it will
detect the checkpoint, skip already-processed samples, and continue from where it left off.
The checkpoint file is deleted automatically on successful completion.

```bash
# Long run interrupted? Just re-run the same command:
python metrics.py --model llama --model-path models/Llama-3.2-3B --eval-limit None

# Force a completely fresh run (discard any checkpoint):
python metrics.py --model llama --model-path models/Llama-3.2-3B --no-resume

# Save more frequently (every 20 samples instead of 50):
python metrics.py --model llama --model-path models/Llama-3.2-3B --checkpoint-interval 20
```

---

## Methodology

This section explains how raw multi-source logs become fine-tuning data and how the trained model is evaluated. Three terms appear throughout the codebase:

> An **event** is a single log entry from one source — one line of activity.  
> A **session** is all events captured during one monitoring window, grouped by `session_id`.  
> A **chunk** is 7 consecutive events (by timestamp) from one session — the unit the model sees.

---

### 1. Events — Raw Multi-Source Log Entries

Events are collected from four sources simultaneously during an attack scenario and merged into one stream per session:

| Source | Tool | Event IDs / Type | Key fields |
|---|---|---|---|
| Windows Sysmon | ELK / Winlogbeat | 1 (process), 3 (network), 11 (file) | `Image`, `CommandLine`, `DestinationIp` |
| Windows Security | ELK / Winlogbeat | 4688 (process), 5156/5157 (firewall) | `NewProcessName`, `DestAddress` |
| Network packets | tshark → JSON | `event_type: "network"` | `layers.IP.src/dst`, `layers.TCP.dport` |
| Browser activity | ActivityWatch | `event_type: "browser"` | `URL`, `title`, `timestamp` |

Each raw event carries annotation fields (`label`, `mitre_techniques`) added by the human annotator. For example, a process-creation event and a network packet from the same attack session look like:

```json
// System event — Windows Sysmon EventID 1 (Process Creation)
{
  "session_id":       "20250918_091900",
  "label":            "suspicious",
  "mitre_techniques": ["T1059.001"],
  "@timestamp":       "2025-09-18T09:19:00Z",
  "winlog": {
    "event_id": 1,
    "event_data": {
      "Image":       "C:\\Windows\\System32\\powershell.exe",
      "CommandLine": "powershell.exe -NoProfile -ExecutionPolicy Bypass -EncodedCommand ...",
      "ParentImage": "C:\\Windows\\System32\\cmd.exe",
      "User":        "DESKTOP-ABC123\\Admin"
    }
  }
}

// Network event — PCAP capture
{
  "session_id":       "20250918_091900",
  "label":            "suspicious",
  "mitre_techniques": ["T1071.001"],
  "@timestamp":       "2025-09-18T09:19:15Z",
  "event_type":       "network",
  "layers": {
    "IP":  { "src": "192.168.1.100",  "dst": "147.185.221.22" },
    "TCP": { "sport": "52341", "dport": "4444", "flags": "A"  }
  },
  "length": 54
}
```

---

### 2. Sessions — One Monitoring Window

A session is identified by its `session_id` (format `YYYYMMDD_HHMMSS`) and contains all events recorded during one continuous monitoring window. A single session may span hundreds of events from all four sources interleaved by timestamp.

| `session_id` | Total events | Suspicious | Normal | Sources |
|---|---|---|---|---|
| `20250918_091900` | 147 | 12 | 135 | Sysmon, PCAP, browser |

> **Session integrity rule:** during train/val/test splitting, all chunks from one session are kept in the same split (70 / 15 / 15). This prevents data leakage between splits.

---

### 3. Chunks — The Model's Unit of Input

Events within a session are sorted by `@timestamp` and grouped into fixed-size **chunks of 7 events**. A chunk is intentionally mixed — it may contain events from different sources and a mix of normal and suspicious activity. The chunk is labelled suspicious if *any* of its 7 events is suspicious.

```
Chunk #0  —  session 20250918_091900
┌────┬──────────────┬────────────────────────────────────────────┬───────────┐
│  # │  Timestamp   │  Event                                     │  Label    │
├────┼──────────────┼────────────────────────────────────────────┼───────────┤
│  1 │ 09:19:00 UTC │ Sysmon 1 — powershell.exe (encoded cmd)    │ suspicious│
│  2 │ 09:19:15 UTC │ PCAP — 192.168.1.100 → 147.185.221.22:4444 │ suspicious│
│  3 │ 09:19:30 UTC │ Sysmon 3 — powershell.exe network conn     │ suspicious│
│  4 │ 09:19:45 UTC │ Sysmon 1 — explorer.exe (normal startup)   │ normal    │
│  5 │ 09:20:00 UTC │ Browser — HTTPS to accounts.google.com     │ normal    │
│  6 │ 09:20:15 UTC │ Sysmon 4688 — notepad.exe                  │ normal    │
│  7 │ 09:21:30 UTC │ Sysmon 11 — .txt.encrypted file created    │ suspicious│
└────┴──────────────┴────────────────────────────────────────────┴───────────┘
Chunk label: SUSPICIOUS  (≥1 suspicious event present)
```

---

### 4. Feature Engineering — Chunk → Training Triple

Each chunk is converted into an `instruction / input / output` triple. Two transformations are applied before the model sees the data:

**Strip label fields** — the model must not see the answer:
```
Removed per log:  label, mitre_techniques, session_id, chunk_label
```

**Flatten nested structures** — normalises field names across source types:
```
winlog.event_data.Image       →  Image
winlog.event_data.CommandLine →  CommandLine
layers.IP.dst                 →  DestinationIp
layers.TCP.dport              →  DestinationPort
```

The resulting training triple looks like:

**`instruction`** — fixed string, identical for every example:
```
Analyze this session log chunk and determine if it contains normal or suspicious
activity. If suspicious, identify all MITRE ATT&CK techniques and explain why.
```

**`input`** — JSON-encoded string (metadata + sanitised logs, no label fields):
```json
{
  "metadata": {
    "session_id":       "20250918_091900",
    "chunk_index":      0,
    "start_time":       "2025-09-18T09:19:00Z",
    "end_time":         "2025-09-18T09:21:30Z",
    "number_of_events": 7
  },
  "logs": [
    {
      "@timestamp":  "2025-09-18T09:19:00Z",
      "Image":       "C:\\Windows\\System32\\powershell.exe",
      "CommandLine": "powershell.exe -NoProfile -ExecutionPolicy Bypass -EncodedCommand ...",
      "ParentImage": "C:\\Windows\\System32\\cmd.exe"
    },
    {
      "@timestamp":      "2025-09-18T09:19:15Z",
      "event_type":      "network",
      "DestinationIp":   "147.185.221.22",
      "DestinationPort": "4444",
      "tcp_flags":       "A",
      "length":          54
    }
    // ... 5 more logs
  ]
}
```

**`output`** — security analysis generated by the annotation pipeline:
```
**SECURITY ALERT**: 4 out of 7 events show malicious activity (Severity: CRITICAL)

**Suspicious Events Detected:**

1. [Process] EventID 1 at 2025-09-18T09:19:00Z — CRITICAL
   powershell.exe spawned by cmd.exe with Base64-encoded command.
   Indicators: encoded argument, parent-child relationship with cmd.exe
   Fields: Image=powershell.exe, CommandLine=-EncodedCommand ...

2. [Network] at 2025-09-18T09:19:15Z — CRITICAL
   Connection to known C2 server 147.185.221.22:4444 (RevengeRAT).
   Indicators: external IP, high-risk port, small periodic packet (54 B)
   Fields: DestinationIp=147.185.221.22, DestinationPort=4444, tcp_flags=A

**MITRE ATT&CK Techniques:** T1027, T1059.001, T1071.001, T1095
**Attack Chain:** c2_beacon_with_execution
**Recommendation:** Immediate investigation required.
```

Dataset sizes after splitting: ~90 K train / 12 K val / 12 K test examples.

---

### 5. Fine-Tuning

The triples are fed to `fine_tune.py` as `instruction + input → output` using each model's native chat template. LoRA adapters are trained on the output tokens only (prompt tokens masked with `-100`). See [Fine-Tuning (`fine_tune.py`)](#fine-tuning-fine_tunepy) for hyperparameters and CLI flags.

---

### 6. Evaluation

At evaluation time the model receives only `instruction + input` (no labels, no techniques). Its generated output is scored against the reference on two levels:

| Metric group | Applies to | What it measures |
|---|---|---|
| Accuracy, Precision, Recall, F1 | `Status:` field | Correct classification — Normal vs Suspicious |
| Exact Match Accuracy | Full output | Strict full-string match against the reference |
| ROUGE-1 / ROUGE-2 / ROUGE-L | Full output | N-gram overlap of the generated explanation |
| METEOR | Full output | Semantic overlap with stemming |

`extract_status_label()` in `metrics.py` parses the Status from the free-text output by looking for markers such as `**SECURITY ALERT**`, `status: suspicious`, or `status: normal`. See [Evaluation (`metrics.py`)](#evaluation-metricspy) for full details.

---

## Chat Templates

Each model's native format is applied consistently in `fine_tune.py` and `metrics.py`:

| Model | Training & inference format |
|---|---|
| **Qwen** | `{instruction}\n\n{input}\n\nAnalysis:\n{output}` (plain text) |
| **Llama** | `<\|begin_of_text\|><\|start_header_id\|>user<\|end_header_id\|>\n\n{instruction}\n\n{input}\n\nAnalysis:<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>\n\n{output}<\|eot_id\|>` |
| **Phi** | `<\|user\|>\n{instruction}\n\n{input}\n\nAnalysis:<\|end\|>\n<\|assistant\|>\n{output}<\|end\|>` |

---

## Comparing Base vs Fine-Tuned (`compare_results.py`)

`compare_results.py` auto-discovers every model folder under `results/`, loads both the
fine-tuned and base-model metrics, and produces a full comparison in one command.

### Quick start

```bash
# Compare all discovered models (reads Fine-Tune/results/ and Base-LLM-Evaluation/results/)
python compare_results.py

# Print table only — no files written
python compare_results.py --no-save

# Restrict to specific models
python compare_results.py --models Llama-3.2-3B Phi-4-mini
```

### Output files (written to `results/`)

| File | Description |
|---|---|
| `comparison_summary.csv` | Wide table: all 10 metrics × (base / fine-tuned / Δ) per model |
| `comparison.png` | Grouped bar chart across all 10 metrics |
| `confusion_matrix_comparison.png` | 3 × 2 subplot grid — per-model confusion matrices, base vs fine-tuned |
| `compare_run.log` | Execution log |

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--results-dir` | `results` | Fine-Tune results directory |
| `--base-results-dir` | `../Base-LLM-Evaluation/results` | Base-model results directory |
| `--output-dir` | `results` | Where to write comparison outputs |
| `--models` | *(all discovered)* | Restrict to specific model names (space-separated) |
| `--no-save` | off | Print table only; skip CSV and PNG output |

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
