# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A cybersecurity research platform for generating MITRE ATT&CK attack scenarios, collecting and annotating security logs, preparing fine-tuning datasets, and running an AI-powered log analyzer. The project spans multiple sub-systems that form an end-to-end ML pipeline.

## Repository Structure

```
Mitre-Dataset/
├── attac-scenarios/          # Attack scenario generator web app (Node.js/React)
├── Automated Log/            # Automated log collection system (Python)
├── Annotate-attack-logs/     # Tools to annotate collected logs with MITRE labels
├── Log Cleaner/              # Cleans/normalizes raw logs from Google Drive
├── Log Labeler/              # Labels logs using Claude API
├── CSV-Log-Cleaner/          # Converts CSV/PCAP logs to JSON
├── Data-preparation/
│   ├── v1/                   # Legacy pipeline
│   └── v2/                   # Active pipeline: extract → clean → chunk → train split
├── Fine-Tune/                # LoRA fine-tuning + evaluation — Python scripts (fine_tune.py, metrics.py) + Kaggle notebooks; supports Qwen2.5-1.5B, Llama-3.2-3B, Phi-4-mini
├── Base-LLM-Evaluation/      # Headless Python script: base model evaluation (vLLM offline batch on 4× Ada 6000); supports Qwen2.5-1.5B, Llama-3.2-3B, Phi-4-mini
├── mitre-attack-analyzer/    # Full-stack inference app (FastAPI + React)
├── classifiers/              # Jupyter notebooks for exploratory classifiers
└── docs/                     # Research documentation and attack scenario descriptions
```

## ML Pipeline Flow

The end-to-end pipeline to go from raw logs to a fine-tuned model:

1. **Collect logs** (`Automated Log/main.py`) — screen recording, packet capture, browser logs, syslog from ELK
2. **Clean logs** (`Log Cleaner/main.py`) — parse and normalize raw logs from Google Drive
3. **Annotate** (`Annotate-attack-logs/main.py`) — label logs as suspicious/normal with MITRE techniques
4. **Prepare data** (`Data-preparation/v2/`) — extract → deduplicate → chunk (7 logs/chunk) → merge & balance → train/val/test split (70/15/15)
5. **Fine-tune** (`Fine-Tune/fine_tune.py`) — LoRA fine-tuning; supports Qwen/Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct, Phi-4-mini-instruct; Kaggle notebooks kept as reference
6. **Serve** (`mitre-attack-analyzer/`) — FastAPI backend loads the fine-tuned model; React frontend for log submission and analysis

## Sub-system Commands

### Attack Scenarios App (`attac-scenarios/`)
```bash
# Backend (Node.js/Express + MongoDB)
cd attac-scenarios/backend
npm install
cp .env.example .env   # set MONGODB_URI, PORT
npm run dev            # starts on :5000

# Frontend (React + Vite + Tailwind)
cd attac-scenarios/frontend
npm install
npm run dev            # starts on :5173
npm run build
npm run lint
```

### MITRE Attack Analyzer (`mitre-attack-analyzer/`)
```bash
# Backend (FastAPI + PyTorch + Beanie/MongoDB)
cd mitre-attack-analyzer/backend
python -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env   # set MODEL_PATH, MONGODB_URL, DEVICE
venv/bin/python -m uvicorn app.main:app --reload  # starts on :8000
# API docs: http://localhost:8000/docs

# Frontend (React + Vite + Tailwind)
cd mitre-attack-analyzer/frontend
npm install
npm run dev            # starts on :3000 (proxies /api → :8000)
npm run build

# Docker (runs both services)
cd mitre-attack-analyzer
docker-compose up
```

### Data Preparation v2 (`Data-preparation/v2/`)
```bash
cd Data-preparation/v2
python config.py               # validate source paths
python main.py                 # run full extract → clean → chunk pipeline

# Individual steps
python extract_suspicious_logs.py
python clean_suspicious_logs.py
python chunk_suspicious_logs.py
python extract_normal_logs.py
python clean_normal_logs.py
python chunk_normal_logs.py

# Merge, balance, and split for training
cd prepare_training
python merge_and_balance.py
python split_train_val_test.py
# or run all at once:
python main.py
```

### Base LLM Evaluation (`Base-LLM-Evaluation/`)
```bash
cd Base-LLM-Evaluation
pip install -r requirements.txt   # vLLM + datasets + kagglehub + sklearn + matplotlib

# Data downloaded automatically from Kaggle (abirashab/train-test-val).
# Requires ~/.kaggle/kaggle.json — or pass --data-path for local files.
# Results are written to results/{ModelName}/ automatically.

# Qwen2.5-1.5B (default)
python base_model_eval.py --model qwen --eval-limit None

# Llama-3.2-3B
python base_model_eval.py --model llama --eval-limit None

# Phi-4-mini
python base_model_eval.py --model phi --eval-limit None

# Background run
nohup python base_model_eval.py --model llama --eval-limit None \
    > results/Llama-3.2-3B/eval.log 2>&1 &

# Smoke-test (50 samples, 1 GPU)
python base_model_eval.py --model phi --eval-limit 50 --tensor-parallel-size 1 --no-save

# Key overrides
#   --model qwen|llama|phi      model to evaluate (required)
#   --splits train val test     which splits to load (default: test)
#   --tensor-parallel-size N    number of GPUs (default: 4)
#   --gpu-memory-utilization F  VRAM fraction per GPU (default: 0.90)
#   --dtype bfloat16|float16    weight dtype (default: bfloat16)
#   --data-path /path/to/dir    local directory with JSONL files (skips Kaggle)
#   --no-save                   print metrics only, skip CSV output
```

### Fine-Tune (`Fine-Tune/`)
```bash
cd Fine-Tune
pip install -r requirements.txt   # transformers + peft + accelerate + kagglehub

# LoRA fine-tuning — data downloaded from Kaggle automatically
python fine_tune.py --model qwen    # → models/Qwen2.5-1.5B/
python fine_tune.py --model llama   # → models/Llama-3.2-3B/
python fine_tune.py --model phi     # → models/Phi-4-mini/

# Full dataset (default uses 40 % sample)
python fine_tune.py --model llama --sample-percentage 1.0

# Local data, custom output, more epochs
python fine_tune.py --model phi --data-path /data/train-test-val \
    --output-dir /ckpts/phi --final-model-dir /models/phi --epochs 5

# Smoke-test (1 % data, 1 epoch)
python fine_tune.py --model llama --sample-percentage 0.01 --epochs 1 --no-resume

# Evaluate a fine-tuned model
python metrics.py --model llama --model-path models/Llama-3.2-3B
python metrics.py --model phi   --model-path models/Phi-4-mini --eval-limit None

# Key overrides for fine_tune.py
#   --model qwen|llama|phi          model to fine-tune (required)
#   --sample-percentage 0.4         fraction of training data (default: 0.4)
#   --epochs N                      training epochs (default: 3)
#   --lr F                          learning rate (default: 2e-4)
#   --lora-r N                      LoRA rank (default: 16)
#   --no-resume                     start fresh, ignore existing checkpoints

# Key overrides for metrics.py
#   --model qwen|llama|phi          model family (required)
#   --model-path /path/to/adapters  fine-tuned model path (required)
#   --eval-limit N|None             sample cap (default: 2000)
#   --splits train val test         splits to evaluate (default: test)
#   --no-save                       print metrics only
```

### Automated Log Collection (`Automated Log/`)
```bash
cd "Automated Log"
pip install -r requirements.txt
cp env.example .env             # configure Elasticsearch, network interface, intervals
# Requires admin/root privileges and: tshark, ActivityWatch, Google Drive credentials
python main.py
```

## Key Architecture Details

### `mitre-attack-analyzer` Backend (3-tier)
- **Controllers** (`app/controllers/`) — FastAPI routers (REST endpoints)
- **Services** (`app/services/`) — business logic; `ml_service.py` loads Qwen2.5-1.5B with LoRA adapters via `AutoModelForCausalLM.from_pretrained` (auto-detects `adapter_config.json`); `chunking_service.py` handles session chunking
- **Repositories** (`app/repositories/`) — async MongoDB via Beanie ODM
- **Models** (`app/models/`) — `LogAnalysis` and `SessionChunk` Beanie documents
- ML model loads once at startup via `lifespan`; first run downloads ~3 GB base model to HuggingFace cache

### Fine-Tuning (`Fine-Tune/`)
- **Models supported:** `Qwen/Qwen2.5-1.5B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`, `microsoft/Phi-4-mini-instruct`
- **Scripts:** `fine_tune.py` (training) + `metrics.py` (evaluation); Kaggle notebooks kept as reference
- LoRA config: r=16, alpha=32, targets: q/k/v/o_proj, dropout=0.05
- Each model uses its **native chat template** — Llama-3 header tokens, Phi-4 `<|user|>/<|end|>` tokens, Qwen raw text
- Prompt masking: prompt tokens set to -100 so loss is computed on output tokens only
- Dataset: ~90K train / 12K val / 12K test (JSONL); downloaded from Kaggle as `abirashab/train-test-val`
- Training hyperparameters: lr=2e-4, batch=1, grad_accum=4 (effective=4), epochs=3, cosine scheduler, AdamW, early stopping (patience=5)
- Checkpoints saved to `checkpoints/{ModelName}/`; final model to `models/{ModelName}/`
- Evaluation results saved to `results/{ModelName}/` (CSV + PNG charts)

### Training Data Format
Every training example is an instruction-tuning triple:
- `instruction`: Fixed string — *"Analyze this session log chunk and determine if it contains normal or suspicious activity..."*
- `input`: JSON string with `metadata` (session_id, chunk_index, timestamps) and `logs` array (fields stripped of `label`/`mitre_techniques`)
- `output`: `Status: Suspicious\nMITRE Techniques: T1059.001 (...)\nReason: ...` or `Status: Normal\nReason: ...`

### Data Pipeline Path Convention
- Source logs (Windows Sysmon + network): `Annotate-attack-logs/suspicious-logs/` and `.../normal-logs/`
- Intermediate files: `Data-preparation/v2/output/`
- Final training chunks: `Data-preparation/v2/training_data/`
- Train/val/test splits: `Data-preparation/v2/prepare_training/final_training_data/`

### Environment Variables
Each sub-system has its own `.env`. Key variables:
- `mitre-attack-analyzer/backend/.env`: `BASE_MODEL`, `MODEL_PATH` (path to fine-tuned LoRA adapters), `DEVICE` (cuda/cpu), `MONGODB_URL`, `CORS_ORIGINS`, `MAX_INPUT_CHARS=6000`
- `attac-scenarios/backend/.env`: `MONGODB_URI`, `PORT=5000`
- `Automated Log/.env`: `ELASTIC_*`, `NETWORK_INTERFACE`, `INTERVAL_MINUTES`, `SCREEN_*`, `BROWSER_API_BASE`

## Development Notes

- `Base-LLM-Evaluation/base_model_eval.py` evaluates the **base** (non-fine-tuned) model using vLLM offline batch inference on 4 × NVIDIA Ada 6000 GPUs; supports Qwen2.5-1.5B, Llama-3.2-3B, Phi-4-mini via `--model`; each model's native chat template is applied automatically; results land in `results/{ModelName}/` for side-by-side comparison
- `Fine-Tune/fine_tune.py` fine-tunes any supported model with LoRA; `Fine-Tune/metrics.py` evaluates the fine-tuned adapters — both use the same `MODEL_REGISTRY` and chat template logic as `base_model_eval.py` so metrics are directly comparable
- Kaggle notebooks `fine-tune.ipynb` and `metrics.ipynb` are kept as-is for Kaggle reference (Qwen only)
- `mitre-attack-analyzer/utils/extract_test_data.py` extracts test samples from the dataset; `mitre-attack-analyzer/data/` holds pre-extracted test JSON files with an `answer_key.json`
- Log chunks use `session_id` (timestamp format `YYYYMMDD_HHMMSS`) as the grouping key throughout — session integrity is preserved across all pipeline stages
- The `attac-scenarios` app manages ~60,000 pre-generated MITRE technique combination scenarios stored in MongoDB
