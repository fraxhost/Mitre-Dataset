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
├── Fine-Tune/                # Kaggle notebooks for LoRA fine-tuning and evaluation on Qwen2.5-1.5B
├── Base-LLM-Evaluation/      # Headless Python script: base model evaluation (vLLM offline batch on 4× Ada 6000, same metrics as Fine-Tune)
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
5. **Fine-tune** (`Fine-Tune/fine-tune.ipynb`) — LoRA on Qwen/Qwen2.5-1.5B-Instruct on Kaggle (2× Tesla T4)
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

# Data is downloaded automatically from Kaggle (abirashab/train-test-val).
# Requires ~/.kaggle/kaggle.json — or pass --data-path to use local files.

# Full test set — all 4 GPUs (recommended)
python base_model_eval.py --eval-limit None --output-dir ./results

# Evaluate on all splits (train + val + test) combined
python base_model_eval.py --splits train val test --eval-limit None --output-dir ./results

# Background run with live log tail
nohup python base_model_eval.py --eval-limit None --output-dir ./results \
    > results/eval.log 2>&1 &
tail -f results/base_eval_run.log

# Smoke-test on 500 samples
python base_model_eval.py --eval-limit 500 --tensor-parallel-size 4

# Key overrides
#   --splits train val test     which splits to load (default: test)
#   --tensor-parallel-size N    number of GPUs (default: 4)
#   --gpu-memory-utilization F  VRAM fraction per GPU (default: 0.90)
#   --dtype bfloat16|float16    weight dtype (default: bfloat16)
#   --data-path /path/to/dir    local directory with JSONL files (skips Kaggle)
#   --output-dir ./results      where to write CSVs, PNGs, log
#   --no-save                   print metrics only, skip CSV output
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

### Fine-Tuning (Kaggle, `Fine-Tune/fine-tune.ipynb`)
- Base model: `Qwen/Qwen2.5-1.5B-Instruct` with LoRA (r=16, alpha=32, targets: q/k/v/o_proj)
- Training data: instruction-tuning format — `instruction` + `input` (JSON log chunk) → `output` (Status + MITRE techniques + Reason)
- Prompt masking: only the output tokens contribute to loss
- Dataset: ~90K train / 12K val examples (JSONL); stored on Kaggle as `abirashab/train-test-val`
- All CONFIG in Cell 4 of the notebook is the single source of truth for hyperparameters

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

- `Base-LLM-Evaluation/base_model_eval.py` evaluates the **base** (non-fine-tuned) model using vLLM offline batch inference on 4 × NVIDIA Ada 6000 GPUs; outputs are directly comparable to `Fine-Tune/metrics.ipynb`. Dataset is downloaded automatically from Kaggle (`abirashab/train-test-val`) via `kagglehub`; use `--splits train val test` to evaluate across all splits, or `--data-path` to point at local files.
- The `Fine-Tune/metrics.ipynb` notebook evaluates the fine-tuned model against the test split
- `mitre-attack-analyzer/utils/extract_test_data.py` extracts test samples from the dataset; `mitre-attack-analyzer/data/` holds pre-extracted test JSON files with an `answer_key.json`
- Log chunks use `session_id` (timestamp format `YYYYMMDD_HHMMSS`) as the grouping key throughout — session integrity is preserved across all pipeline stages
- The `attac-scenarios` app manages ~60,000 pre-generated MITRE technique combination scenarios stored in MongoDB
