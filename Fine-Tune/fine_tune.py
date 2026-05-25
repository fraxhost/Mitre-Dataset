"""
Fine-Tuning Script — LoRA on Qwen2.5-1.5B / Llama-3.2-3B / Phi-4-mini
=======================================================================
Python rewrite of Fine-Tune/fine-tune.ipynb, extended to support three
base models.  All hyperparameters match the notebook exactly so results
are directly comparable.

Supported models (--model flag):
  qwen   →  Qwen/Qwen2.5-1.5B-Instruct
  llama  →  meta-llama/Llama-3.2-3B-Instruct
  phi    →  microsoft/Phi-4-mini-instruct

Each model's native chat template is applied during tokenisation so that
fine-tuned weights and the evaluation prompts are consistent.

Usage
-----
  pip install -r Fine-Tune/requirements.txt

  # Fine-tune Llama (data downloaded automatically from Kaggle)
  python Fine-Tune/fine_tune.py --model llama

  # Full dataset, more epochs
  python Fine-Tune/fine_tune.py --model phi --sample-percentage 1.0 --epochs 5

  # Local data, custom dirs
  python Fine-Tune/fine_tune.py --model qwen \\
      --data-path /data/train-test-val \\
      --output-dir /ckpts/qwen \\
      --final-model-dir /models/qwen

  # Smoke-test (1 % data, 1 epoch, resume disabled)
  python Fine-Tune/fine_tune.py --model llama \\
      --sample-percentage 0.01 --epochs 1 --no-resume
"""

# ── Standard library ─────────────────────────────────────────────────────────
import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

# ── Third-party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "qwen": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "display_name": "Qwen2.5-1.5B",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "chat_template": "qwen",
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    },
    "llama": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "display_name": "Llama-3.2-3B",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "chat_template": "llama3",
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
    },
    "phi": {
        "model_id": "microsoft/Phi-4-mini-instruct",
        "display_name": "Phi-4-mini",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "chat_template": "phi4",
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONFIGURATION  (matches fine-tune.ipynb Cell 4 exactly)
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Data ─────────────────────────────────────────────────────────────────
    "kaggle_dataset": "abirashab/train-test-val",
    "data_path": None,  # overridden by --data-path
    "train_file": "train.jsonl",
    "val_file": "val.jsonl",
    # ── Sampling ─────────────────────────────────────────────────────────────
<<<<<<< Updated upstream
    "sample_percentage": 0.4,  # fraction of training data to use
    "min_length": 10,  # discard tokenised sequences shorter than this
    "max_length": 2000,  # truncate / discard sequences longer than this
=======
    "sample_percentage":        1.0,    # fraction of training data to use
    "min_length":               10,     # discard tokenised sequences shorter than this
    "max_length":               2000,   # truncate / discard sequences longer than this

>>>>>>> Stashed changes
    # ── Training hyperparameters ─────────────────────────────────────────────
    "batch_size": 1,
    "gradient_accumulation_steps": 4,  # effective batch = 4
    "num_epochs": 3,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "cosine",
    "optim": "adamw_torch",
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "label_smoothing": 0.0,  # disabled to save memory
    # ── LoRA ─────────────────────────────────────────────────────────────────
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    # ── Strategy ─────────────────────────────────────────────────────────────
    "fp16": True,
    # gradient_checkpointing disabled — causes CUDA errors with max_length=2000
    "gradient_checkpointing": False,
    # ── Logging / checkpointing ───────────────────────────────────────────────
    "logging_steps": 25,
    "eval_steps": 100,
    "save_steps": 200,
    "save_total_limit": 3,
    "early_stopping_patience": 5,
    "early_stopping_threshold": 0.005,
    # ── Prompt ───────────────────────────────────────────────────────────────
    "max_input_chars": 6000,  # char truncation before tokenisation
    # ── Output dirs (populated from model key in apply_args) ─────────────────
    "output_dir": None,
    "final_model_dir": None,
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOGGING
# ══════════════════════════════════════════════════════════════════════════════


def setup_logging(output_dir: str, name: str = "fine_tune") -> logging.Logger:
    log_path = Path(output_dir) / "fine_tune_run.log"
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO, format=fmt, datefmt=datefmt, handlers=handlers, force=True
    )
    logger = logging.getLogger(name)
    logger.info(f"Log file: {log_path.resolve()}")
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CHAT TEMPLATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def format_training_example(
    instruction: str, input_text: str, output: str, model_key: str, max_input_chars: int
) -> tuple[str, str]:
    """
    Return (full_sequence, prompt_prefix) for a training example.

    full_sequence  — the complete text that gets tokenised and trained on
    prompt_prefix  — the instruction+input portion only (used to compute
                     how many tokens to mask with -100)

    Native chat templates are applied per model so fine-tuned weights are
    consistent with the prompts used in metrics.py / base_model_eval.py.
    """
    if len(input_text) > max_input_chars:
        input_text = input_text[:max_input_chars] + "... [truncated]"

    if model_key == "llama":
        prompt_prefix = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}\n\n{input_text}\n\nAnalysis:"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        full_sequence = prompt_prefix + output + "<|eot_id|>"

    elif model_key == "phi":
        prompt_prefix = (
            f"<|user|>\n{instruction}\n\n{input_text}\n\nAnalysis:<|end|>\n"
            "<|assistant|>\n"
        )
        full_sequence = prompt_prefix + output + "<|end|>"

    else:  # qwen / default — plain text matching existing notebook
        prompt_prefix = f"{instruction}\n\n{input_text}\n\nAnalysis:\n"
        full_sequence = prompt_prefix + output

    return full_sequence, prompt_prefix


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════


def resolve_data_dir(cfg: dict, log: logging.Logger) -> str:
    if cfg["data_path"]:
        log.info(f"Using local data directory: {cfg['data_path']}")
        return cfg["data_path"]
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub is not installed. Run: pip install kagglehub\n"
            "Or use --data-path to point at a local directory."
        )
    log.info(f"Downloading Kaggle dataset: {cfg['kaggle_dataset']}")
    log.info("(Cached after first download — subsequent runs are instant.)")
    data_dir = kagglehub.dataset_download(cfg["kaggle_dataset"])
    log.info(f"Dataset available at: {data_dir}")
    return data_dir


def load_data(cfg: dict, log: logging.Logger):
    log.info("=" * 70)
    log.info("STEP 1 — Loading training data")
    log.info("=" * 70)

    data_dir = resolve_data_dir(cfg, log)

    file_map = {
        "train": os.path.join(data_dir, cfg["train_file"]),
        "validation": os.path.join(data_dir, cfg["val_file"]),
    }
    for split, path in file_map.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{split} file not found: {path}")
        log.info(f"  {split:10s} → {path}")

    dataset = load_dataset("json", data_files=file_map)
    log.info(
        f"Loaded — train: {len(dataset['train']):,}  val: {len(dataset['validation']):,}"
    )
    log.info(f"Columns: {dataset['train'].column_names}")
    return dataset


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MODEL & TOKENIZER
# ══════════════════════════════════════════════════════════════════════════════


def load_model_and_tokenizer(cfg: dict, log: logging.Logger):
    reg = MODEL_REGISTRY[cfg["model_key"]]

    log.info("=" * 70)
    log.info("STEP 2 — Loading base model and tokenizer")
    log.info("=" * 70)
    log.info(f"Model ID      : {cfg['model_name']}")
    log.info(f"Display name  : {cfg['display_name']}")
    log.info(f"Chat template : {reg['chat_template']}")
    log.info(f"torch_dtype   : {reg['torch_dtype']}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        trust_remote_code=reg["trust_remote_code"],
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=reg["torch_dtype"],
        device_map="auto",
        trust_remote_code=reg["trust_remote_code"],
    )

    log.info(f"GPU memory after load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    log.info(f"Model device: {model.device}")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — LORA
# ══════════════════════════════════════════════════════════════════════════════


def apply_lora(model, cfg: dict, log: logging.Logger):
    reg = MODEL_REGISTRY[cfg["model_key"]]

    log.info("=" * 70)
    log.info("STEP 3 — Applying LoRA")
    log.info("=" * 70)
    log.info(f"  r             : {cfg['lora_r']}")
    log.info(f"  alpha         : {cfg['lora_alpha']}")
    log.info(f"  dropout       : {cfg['lora_dropout']}")
    log.info(f"  target_modules: {reg['lora_targets']}")

    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Freeze base weights; cast 1-D tensors (layer norms) to fp32 for stability
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=reg["lora_targets"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    log.info(
        f"Trainable parameters: {trainable:,} / {total:,}  ({trainable / total * 100:.3f}%)"
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — DATASET PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════


def preprocess_dataset(dataset, tokenizer, cfg: dict, log: logging.Logger):
    """
    Tokenise every example, mask prompt tokens with -100 (loss only on output),
    filter by min/max sequence length, and optionally sub-sample training set.

    Mirrors the logic in fine-tune.ipynb Cell 8.
    """
    log.info("=" * 70)
    log.info("STEP 4 — Preprocessing dataset")
    log.info("=" * 70)

    model_key = cfg["model_key"]
    max_input_chars = cfg["max_input_chars"]

    def tokenise_and_mask(example):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        full_seq, prompt_prefix = format_training_example(
            instruction, input_text, output, model_key, max_input_chars
        )

        # Tokenise the full sequence
        tokenised = tokenizer(
            full_seq,
            truncation=False,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )

        # Tokenise the prompt prefix to find the boundary
        prompt_ids = tokenizer(
            prompt_prefix,
            truncation=False,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )["input_ids"]
        prompt_len = len(prompt_ids)

        labels = tokenised["input_ids"].copy()
        # Mask everything up to (and including) the prompt — only train on output
        labels[:prompt_len] = [-100] * prompt_len

        tokenised["labels"] = labels
        tokenised["length"] = len(tokenised["input_ids"])
        return tokenised

    log.info("Tokenising training split...")
    train_tok = dataset["train"].map(
        tokenise_and_mask,
        remove_columns=dataset["train"].column_names,
        desc="Tokenising train",
    )
    log.info("Tokenising validation split...")
    val_tok = dataset["validation"].map(
        tokenise_and_mask,
        remove_columns=dataset["validation"].column_names,
        desc="Tokenising val",
    )

    # Filter by length
    n_before = len(train_tok)
    train_tok = train_tok.filter(
        lambda x: cfg["min_length"] <= x["length"] <= cfg["max_length"]
    )
    log.info(
        f"Train after length filter [{cfg['min_length']}–{cfg['max_length']}]: "
        f"{n_before:,} → {len(train_tok):,}"
    )

    n_before = len(val_tok)
    val_tok = val_tok.filter(
        lambda x: cfg["min_length"] <= x["length"] <= cfg["max_length"]
    )
    log.info(f"Val   after length filter: {n_before:,} → {len(val_tok):,}")

    # Sub-sample training set (reproducible, seed=42 matches notebook)
    if cfg["sample_percentage"] < 1.0:
        n_sample = max(1, int(len(train_tok) * cfg["sample_percentage"]))
        random.seed(42)
        indices = random.sample(range(len(train_tok)), n_sample)
        train_tok = train_tok.select(indices)
        log.info(
            f"Sampled {cfg['sample_percentage']*100:.0f}% of train → {len(train_tok):,} examples"
        )

    return train_tok, val_tok


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════


def train(model, tokenizer, train_dataset, val_dataset, cfg: dict, log: logging.Logger):
    log.info("=" * 70)
    log.info("STEP 5 — Training")
    log.info("=" * 70)

    # Match the precision flags to how the model was loaded
    is_bf16 = MODEL_REGISTRY[cfg["model_key"]]["torch_dtype"] == torch.bfloat16
    log.info(f"Mixed precision: {'bf16' if is_bf16 else 'fp16'}")

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        optim=cfg["optim"],
        adam_beta1=cfg["adam_beta1"],
        adam_beta2=cfg["adam_beta2"],
        adam_epsilon=cfg["adam_epsilon"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        # fp16=cfg["fp16"],
        fp16=not is_bf16,
        bf16=is_bf16,
        gradient_checkpointing=cfg["gradient_checkpointing"],
        label_smoothing_factor=cfg["label_smoothing"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg["early_stopping_patience"],
                early_stopping_threshold=cfg["early_stopping_threshold"],
            )
        ],
    )

    # Auto-resume from the latest checkpoint unless --no-resume was passed
    resume_from = None
    if not cfg.get("no_resume", False):
        ckpt_dir = Path(cfg["output_dir"])
        if ckpt_dir.exists():
            ckpts = sorted(
                ckpt_dir.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[1]),
            )
            if ckpts:
                resume_from = str(ckpts[-1])
                log.info(f"Resuming from checkpoint: {resume_from}")

    log.info(
        f"Training on {len(train_dataset):,} examples, validating on {len(val_dataset):,}"
    )
    log.info(
        f"Effective batch size: {cfg['batch_size'] * cfg['gradient_accumulation_steps']}  "
        f"| Epochs: {cfg['num_epochs']}  | LR: {cfg['learning_rate']}"
    )

    t0 = time.time()
    try:
        trainer.train(resume_from_checkpoint=resume_from)
    except KeyboardInterrupt:
        log.warning("Training interrupted by user. Saving current state...")

    elapsed = time.time() - t0
    log.info(f"Training complete: {elapsed / 60:.2f} min")

    eval_results = trainer.evaluate()
    log.info(f"Final eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
    return trainer


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — SAVE MODEL
# ══════════════════════════════════════════════════════════════════════════════


def save_model(trainer, tokenizer, cfg: dict, log: logging.Logger):
    log.info("=" * 70)
    log.info("STEP 6 — Saving fine-tuned model")
    log.info("=" * 70)

    final_dir = cfg["final_model_dir"]
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    log.info(f"Model saved to: {final_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — SMOKE-TEST INFERENCE
# ══════════════════════════════════════════════════════════════════════════════


def test_inference(model, tokenizer, cfg: dict, log: logging.Logger):
    """
    Quick sanity-check: run a single example through the trained model and
    print the prediction.  Mirrors notebook Cell 14.
    """
    log.info("=" * 70)
    log.info("STEP 7 — Smoke-test inference")
    log.info("=" * 70)

    # Sample suspicious log (PowerShell + C2 connection, same as notebook)
    instruction = (
        "Analyze this session log chunk and determine if it contains normal or "
        "suspicious activity. If suspicious, identify all MITRE ATT&CK techniques "
        "and explain why."
    )
    input_text = (
        '{"metadata": {"session_id": "20240101_120000", "chunk_index": 0, '
        '"number_of_events": 3}, "logs": ['
        '{"EventID": 4688, "ProcessName": "powershell.exe", '
        '"CommandLine": "powershell -enc SQBFAFgA...", "ParentProcess": "cmd.exe"}, '
        '{"EventID": 3, "DestinationIP": "185.220.101.5", "DestinationPort": 4444, '
        '"Protocol": "TCP"}, '
        '{"EventID": 4624, "LogonType": 3, "SourceIP": "185.220.101.5"}]}'
    )

    full_seq, prompt_prefix = format_training_example(
        instruction, input_text, "", cfg["model_key"], cfg["max_input_chars"]
    )
    # Use only the prompt part for inference
    inputs = tokenizer(
        prompt_prefix,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        add_special_tokens=False,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    autocast_dtype = MODEL_REGISTRY[cfg["model_key"]]["torch_dtype"]
    device_type = model.device.type
    if device_type == "cpu" and autocast_dtype == torch.float16:
        autocast_dtype = torch.bfloat16
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=autocast_dtype):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    log.info("Smoke-test prediction:")
    log.info(generated[:600])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — ARGUMENT PARSING & MAIN
# ══════════════════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(
        description="LoRA fine-tuning for MITRE ATT&CK log analysis — "
        "supports Qwen2.5-1.5B, Llama-3.2-3B, and Phi-4-mini."
    )
    p.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to fine-tune: qwen | llama | phi",
    )
    p.add_argument(
        "--data-path",
        default=None,
        help="Local dir with train.jsonl / val.jsonl (skips Kaggle download)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Checkpoint directory (default: checkpoints/{ModelName})",
    )
    p.add_argument(
        "--final-model-dir",
        default=None,
        help="Where to save the final model (default: models/{ModelName})",
    )
    p.add_argument(
        "--epochs", default=None, type=int, help="Training epochs (default: 3)"
    )
    p.add_argument(
        "--lr", default=None, type=float, help="Learning rate (default: 2e-4)"
    )
    p.add_argument(
        "--sample-percentage",
        default=None,
        type=float,
        help="Fraction of training data to use (default: 0.4)",
    )
    p.add_argument("--lora-r", default=None, type=int, help="LoRA rank (default: 16)")
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Start training from scratch, ignore existing checkpoints",
    )
    return p.parse_args()


def apply_args(cfg: dict, args) -> dict:
    reg = MODEL_REGISTRY[args.model]
    cfg["model_key"] = args.model
    cfg["model_name"] = reg["model_id"]
    cfg["display_name"] = reg["display_name"]

    # Auto-set output dirs from display name unless overridden
    cfg["output_dir"] = args.output_dir or os.path.join(
        "checkpoints", reg["display_name"]
    )
    cfg["final_model_dir"] = args.final_model_dir or os.path.join(
        "models", reg["display_name"]
    )

    if args.data_path:
        cfg["data_path"] = args.data_path
    if args.epochs is not None:
        cfg["num_epochs"] = args.epochs
    if args.lr is not None:
        cfg["learning_rate"] = args.lr
    if args.sample_percentage is not None:
        cfg["sample_percentage"] = args.sample_percentage
    if args.lora_r is not None:
        cfg["lora_r"] = args.lora_r
    cfg["no_resume"] = args.no_resume
    return cfg


def main():
    args = parse_args()
    cfg = apply_args(CONFIG, args)

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    log = setup_logging(cfg["output_dir"])
    log.info(f"PID {os.getpid()} — fine_tune.py starting")
    log.info(f"Model        : {cfg['display_name']}  ({cfg['model_name']})")
    log.info(f"Checkpoints  : {cfg['output_dir']}")
    log.info(f"Final model  : {cfg['final_model_dir']}")
    log.info(f"Python {sys.version.split()[0]}  |  cwd: {os.getcwd()}")

    log.info("=" * 70)
    log.info("CONFIGURATION")
    log.info("=" * 70)
    for k, v in cfg.items():
        log.info(f"  {k:<30}: {v}")

    try:
        dataset = load_data(cfg, log)
        model, tokenizer = load_model_and_tokenizer(cfg, log)
        model = apply_lora(model, cfg, log)
        train_tok, val_tok = preprocess_dataset(dataset, tokenizer, cfg, log)
        trainer = train(model, tokenizer, train_tok, val_tok, cfg, log)
        save_model(trainer, tokenizer, cfg, log)
        test_inference(trainer.model, tokenizer, cfg, log)

    except Exception:
        log.exception("Fine-tuning failed with an unhandled exception.")
        sys.exit(1)

    log.info("=" * 70)
    log.info("Done. Run metrics.py to evaluate the fine-tuned model.")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
