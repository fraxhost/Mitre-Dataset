"""
Fine-Tuned Model Evaluation — multi-model
==========================================
Python rewrite of Fine-Tune/metrics.ipynb, extended to support three models.
Loads a fine-tuned model (LoRA adapters or merged weights) and evaluates it
on the test split using the same metrics as Base-LLM-Evaluation/base_model_eval.py
so base vs fine-tuned comparisons are exact.

Supported models (--model flag):
  qwen   →  Qwen/Qwen2.5-1.5B-Instruct
  llama  →  meta-llama/Llama-3.2-3B-Instruct
  phi    →  microsoft/Phi-4-mini-instruct

Uses Transformers (not vLLM) for inference so PEFT LoRA adapters can be
loaded directly without merging weights.

Usage
-----
  # Evaluate fine-tuned Llama (model-path is the output of fine_tune.py)
  python Fine-Tune/metrics.py --model llama --model-path models/Llama-3.2-3B

  # Full test set
  python Fine-Tune/metrics.py --model phi --model-path models/Phi-4-mini \\
      --eval-limit None

  # All splits combined
  python Fine-Tune/metrics.py --model qwen --model-path models/Qwen2.5-1.5B \\
      --splits train val test --eval-limit None

  # Smoke-test
  python Fine-Tune/metrics.py --model llama --model-path models/Llama-3.2-3B \\
      --eval-limit 50 --no-save
"""

# ── Matplotlib non-interactive backend before any other import ───────────────
import matplotlib
matplotlib.use("Agg")

# ── Standard library ─────────────────────────────────────────────────────────
import argparse
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

# ── Third-party ──────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "qwen": {
        "model_id":          "Qwen/Qwen2.5-1.5B-Instruct",
        "display_name":      "Qwen2.5-1.5B",
        "chat_template":     "qwen",
        "trust_remote_code": True,
        "torch_dtype":       torch.float16,
    },
    "llama": {
        "model_id":          "meta-llama/Llama-3.2-3B-Instruct",
        "display_name":      "Llama-3.2-3B",
        "chat_template":     "llama3",
        "trust_remote_code": False,
        "torch_dtype":       torch.bfloat16,
    },
    "phi": {
        "model_id":          "microsoft/Phi-4-mini-instruct",
        "display_name":      "Phi-4-mini",
        "chat_template":     "phi4",
        "trust_remote_code": False,
        "torch_dtype":       torch.bfloat16,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONFIGURATION  (matches metrics.ipynb Cell 3 exactly)
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Data ─────────────────────────────────────────────────────────────────
    "kaggle_dataset": "abirashab/train-test-val",
    "data_path":      None,
    "splits":         ["test"],
    "train_file":     "train.jsonl",
    "val_file":       "val.jsonl",
    "test_file":      "test.jsonl",

    # ── Evaluation ───────────────────────────────────────────────────────────
    "eval_limit":          None,   # None = full dataset
    "suspicious_ratio":    0.30,
    "prioritize_shortest": True,
    "max_new_tokens":      512,
    "temperature":         0.7,
    "do_sample":           True,
    "top_p":               0.9,

    # ── Prompt ───────────────────────────────────────────────────────────────
    "max_input_chars":     6000,
    "max_length_tokens":   3072,   # tokeniser truncation limit

    # ── Output ───────────────────────────────────────────────────────────────
    "show_examples":  10,
    "save_results":   True,
    "output_dir":     None,        # auto-set to results/{display_name}
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: str) -> logging.Logger:
    log_path = Path(output_dir) / "metrics_run.log"
    fmt      = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt  = "%Y-%m-%d %H:%M:%S"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt,
                        handlers=handlers, force=True)
    logger = logging.getLogger("metrics")
    logger.info(f"Log file: {log_path.resolve()}")
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CHAT TEMPLATE & PROMPT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(instruction: str, input_text: str, model_key: str,
                 max_input_chars: int) -> str:
    """Build the inference prompt using the model's native chat template."""
    if len(input_text) > max_input_chars:
        input_text = input_text[:max_input_chars] + "... [truncated]"

    if model_key == "llama":
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}\n\n{input_text}\n\nAnalysis:"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if model_key == "phi":
        return (
            f"<|user|>\n{instruction}\n\n{input_text}\n\nAnalysis:<|end|>\n"
            "<|assistant|>\n"
        )
    # qwen / default — plain text
    return f"{instruction}\n\n{input_text}\n\nAnalysis:\n"


def extract_status_label(text: str) -> str:
    """
    Parse NORMAL / SUSPICIOUS / UNKNOWN from model output.
    Identical to base_model_eval.py and metrics.ipynb for cross-script comparability.
    """
    if not text or not isinstance(text, str):
        return "UNKNOWN"

    t = text.lower()

    suspicious_markers = [
        "security alert", "**security alert", "suspicious activity",
        "malicious activity", "attack detected", "threat detected",
    ]
    normal_markers = [
        "normal activity detected", "**normal activity", "no malicious",
        "no suspicious indicators", "standard activity", "routine activity",
    ]

    for m in suspicious_markers:
        if m in t:
            return "SUSPICIOUS"
    for m in normal_markers:
        if m in t:
            return "NORMAL"

    if "status: normal" in t or "status:normal" in t:
        return "NORMAL"
    if "status: suspicious" in t or "status:suspicious" in t:
        return "SUSPICIOUS"

    return "UNKNOWN"


def calculate_exact_match(pred: str, target: str) -> float:
    return 1.0 if pred.strip().lower() == target.strip().lower() else 0.0


def calculate_partial_match(pred: str, target: str) -> float:
    target_words = set(target.strip().lower().split())
    pred_words   = set(pred.strip().lower().split())
    if not target_words:
        return 0.0
    return len(target_words & pred_words) / len(target_words)


def calculate_word_f1(pred: str, target: str) -> float:
    pred_words   = set(pred.strip().lower().split())
    target_words = set(target.strip().lower().split())
    if not pred_words or not target_words:
        return 0.0
    overlap   = len(pred_words & target_words)
    precision = overlap / len(pred_words)
    recall    = overlap / len(target_words)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(cfg: dict, log: logging.Logger):
    reg        = MODEL_REGISTRY[cfg["model_key"]]
    model_path = cfg["model_path"]

    log.info("=" * 70)
    log.info("STEP 1 — Loading fine-tuned model")
    log.info("=" * 70)
    log.info(f"Model path    : {model_path}")
    log.info(f"Display name  : {cfg['display_name']}")
    log.info(f"Chat template : {reg['chat_template']}")
    log.info(f"torch_dtype   : {reg['torch_dtype']}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=reg["trust_remote_code"],
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=reg["torch_dtype"],
        device_map="auto",
        trust_remote_code=reg["trust_remote_code"],
    )
    model.eval()

    log.info(f"GPU memory after load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    log.info(f"Model device: {model.device}")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DATA LOADING
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
    log.info("STEP 2 — Loading dataset")
    log.info("=" * 70)

    split_file_map = {
        "train": cfg["train_file"],
        "val":   cfg["val_file"],
        "test":  cfg["test_file"],
    }

    data_dir   = resolve_data_dir(cfg, log)
    data_files = {}
    for split in cfg["splits"]:
        fpath = os.path.join(data_dir, split_file_map[split])
        if not os.path.exists(fpath):
            log.warning(f"Split file not found, skipping: {fpath}")
        else:
            data_files[split] = fpath
            log.info(f"  {split:5s} → {fpath}")

    if not data_files:
        raise FileNotFoundError(
            f"No JSONL files found for splits {cfg['splits']} in: {data_dir}"
        )

    raw    = load_dataset("json", data_files=data_files)
    parts  = [raw[s] for s in data_files]
    dataset = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    log.info(
        f"Loaded splits: {list(data_files.keys())}  —  "
        f"{len(dataset):,} examples.  Columns: {dataset.column_names}"
    )
    return dataset


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — STRATIFIED SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def build_eval_set(dataset, cfg: dict, log: logging.Logger):
    log.info("=" * 70)
    log.info("STEP 3 — Stratified sampling")
    log.info("=" * 70)

    if cfg["eval_limit"] is None:
        log.info(f"Using full dataset: {len(dataset):,} examples")
        return dataset

    log.info(
        f"Target: {cfg['eval_limit']:,} examples  "
        f"({cfg['suspicious_ratio'] * 100:.0f}% suspicious / "
        f"{(1 - cfg['suspicious_ratio']) * 100:.0f}% normal)"
    )

    indexed = [
        {
            "index":  i,
            "status": extract_status_label(dataset[i]["output"]),
            "length": len(dataset[i]["input"]),
        }
        for i in range(len(dataset))
    ]

    suspicious = [s for s in indexed if s["status"] == "SUSPICIOUS"]
    normal     = [s for s in indexed if s["status"] == "NORMAL"]
    unknown    = [s for s in indexed if s["status"] == "UNKNOWN"]
    log.info(
        f"Available — suspicious: {len(suspicious):,}  "
        f"normal: {len(normal):,}  unknown: {len(unknown):,}"
    )

    if not suspicious and not normal:
        raise ValueError(
            "Status extraction returned only UNKNOWN — check test data format.\n"
            f"Sample output: {dataset[0]['output'][:300]}"
        )

    if cfg["prioritize_shortest"]:
        suspicious.sort(key=lambda x: x["length"])
        normal.sort(key=lambda x: x["length"])
        log.info("Sorted by ascending input length (shortest first).")

    n_susp   = int(cfg["eval_limit"] * cfg["suspicious_ratio"])
    n_normal = cfg["eval_limit"] - n_susp

    sel_susp   = suspicious[:min(n_susp,   len(suspicious))]
    sel_normal = normal[:min(n_normal, len(normal))]

    log.info(f"Selected — suspicious: {len(sel_susp):,}  normal: {len(sel_normal):,}")

    indices  = [s["index"] for s in sel_susp + sel_normal]
    eval_set = dataset.select(indices)
    log.info(f"Final eval set: {len(eval_set):,} examples")
    return eval_set


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def generate_response(model, tokenizer, instruction: str, input_text: str,
                      cfg: dict) -> str:
    """
    Generate a single prediction using Transformers .generate().
    Mirrors metrics.ipynb Cell 6 generate_response() exactly.
    """
    prompt = build_prompt(instruction, input_text, cfg["model_key"], cfg["max_input_chars"])

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cfg["max_length_tokens"],
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg["max_new_tokens"],
            temperature=cfg["temperature"],
            do_sample=cfg["do_sample"],
            top_p=cfg["top_p"],
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return generated.strip()


def run_evaluation(eval_set, model, tokenizer, cfg: dict, log: logging.Logger):
    log.info("=" * 70)
    log.info("STEP 4 — Running evaluation")
    log.info("=" * 70)
    log.info(f"Evaluating {len(eval_set):,} examples  (chat template: {MODEL_REGISTRY[cfg['model_key']]['chat_template']})")

    results     = []
    partial_scores = []
    word_f1_scores = []
    exact_match_total = 0.0

    t0 = time.time()
    for i, example in enumerate(eval_set):
        pred = generate_response(
            model, tokenizer,
            example["instruction"], example["input"],
            cfg,
        )
        exp = example["output"]

        em      = calculate_exact_match(pred, exp)
        partial = calculate_partial_match(pred, exp)
        wf1     = calculate_word_f1(pred, exp)

        exact_match_total += em
        partial_scores.append(partial)
        word_f1_scores.append(wf1)

        results.append({
            "index":         i,
            "instruction":   example["instruction"],
            "input":         example["input"],
            "expected":      exp,
            "predicted":     pred,
            "exact_match":   em,
            "partial_match": partial,
            "f1_score":      wf1,
        })

        if (i + 1) % 50 == 0 or i == 0:
            elapsed_so_far = time.time() - t0
            log.info(
                f"  [{i+1:>5}/{len(eval_set)}]  "
                f"{elapsed_so_far / 60:.1f} min elapsed  "
                f"({elapsed_so_far / (i+1):.1f} sec/example)"
            )

    elapsed = time.time() - t0

    # Show sample predictions
    n_show = min(cfg["show_examples"], len(results))
    for i in range(n_show):
        r = results[i]
        log.info("-" * 60)
        log.info(f"Example {i + 1}/{n_show}")
        log.info(f"  Input     : {r['input'][:100]}...")
        log.info(f"  Expected  : {r['expected'][:150]}...")
        log.info(f"  Predicted : {r['predicted'][:150]}...")
        log.info(
            f"  Scores    : exact={r['exact_match']:.0f}  "
            f"partial={r['partial_match']:.3f}  word_f1={r['f1_score']:.3f}"
        )

    log.info(
        f"Evaluation complete: {elapsed / 60:.2f} min total  "
        f"({elapsed / len(results):.2f} sec/example)"
    )
    return results, exact_match_total, partial_scores, word_f1_scores, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(results, exact_match_total, partial_scores, word_f1_scores,
                    elapsed, cfg, log):
    log.info("=" * 70)
    log.info("STEP 5 — Computing classification metrics")
    log.info("=" * 70)

    y_true = [extract_status_label(r["expected"])  for r in results]
    y_pred = [extract_status_label(r["predicted"]) for r in results]

    unique_labels = sorted(set(y_true + y_pred))
    log.info(f"Labels found: {unique_labels}")
    log.info(f"True  dist : {dict(Counter(y_true))}")
    log.info(f"Pred  dist : {dict(Counter(y_pred))}")

    # Warn if majority are UNKNOWN (model not following format)
    unknown_rate = y_pred.count("UNKNOWN") / len(y_pred)
    if unknown_rate > 0.5:
        log.warning(
            f"⚠  {unknown_rate*100:.1f}% of predictions are UNKNOWN — "
            "model may not be following the expected output format."
        )

    accuracy      = accuracy_score(y_true, y_pred)
    prec_macro    = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_macro     = recall_score(   y_true, y_pred, average="macro",    zero_division=0)
    rec_weighted  = recall_score(   y_true, y_pred, average="weighted", zero_division=0)
    f1_macro      = f1_score(       y_true, y_pred, average="macro",    zero_division=0)
    f1_weighted   = f1_score(       y_true, y_pred, average="weighted", zero_division=0)

    exact_match_acc = exact_match_total / len(results)
    avg_partial     = float(np.mean(partial_scores))
    avg_word_f1     = float(np.mean(word_f1_scores))

    conf_mat      = confusion_matrix(y_true, y_pred, labels=unique_labels)
    input_lengths = [len(r["input"]) for r in results]

    log.info("=" * 70)
    log.info("BASIC STATISTICS")
    log.info("=" * 70)
    log.info(f"  Total examples    : {len(results):,}")
    log.info(f"  Inference time    : {elapsed / 60:.2f} min")
    log.info(f"  Time per example  : {elapsed / len(results):.2f} sec")
    log.info(
        f"  Input length — min:{min(input_lengths):,}  "
        f"max:{max(input_lengths):,}  "
        f"mean:{np.mean(input_lengths):,.0f}  "
        f"median:{np.median(input_lengths):,.0f}"
    )

    log.info("=" * 70)
    log.info("STATUS CLASSIFICATION METRICS")
    log.info("=" * 70)
    log.info(f"  Accuracy             : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    log.info(f"  Precision (Macro)    : {prec_macro:.4f}")
    log.info(f"  Precision (Weighted) : {prec_weighted:.4f}")
    log.info(f"  Recall    (Macro)    : {rec_macro:.4f}")
    log.info(f"  Recall    (Weighted) : {rec_weighted:.4f}")
    log.info(f"  F1-Score  (Macro)    : {f1_macro:.4f}")
    log.info(f"  F1-Score  (Weighted) : {f1_weighted:.4f}")
    log.info("WORD-LEVEL METRICS")
    log.info(f"  Exact Match Accuracy : {exact_match_acc:.4f}  ({exact_match_acc * 100:.2f}%)")
    log.info(f"  Avg Partial Match    : {avg_partial:.4f}")
    log.info(f"  Avg Word-level F1    : {avg_word_f1:.4f}")
    log.info("SKLEARN CLASSIFICATION REPORT")
    log.info("\n" + classification_report(y_true, y_pred, labels=unique_labels, zero_division=0))

    metrics = {
        "accuracy":        accuracy,
        "prec_macro":      prec_macro,
        "prec_weighted":   prec_weighted,
        "rec_macro":       rec_macro,
        "rec_weighted":    rec_weighted,
        "f1_macro":        f1_macro,
        "f1_weighted":     f1_weighted,
        "exact_match_acc": exact_match_acc,
        "avg_partial":     avg_partial,
        "avg_word_f1":     avg_word_f1,
    }
    return metrics, y_true, y_pred, unique_labels, conf_mat


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(results, y_true, y_pred, unique_labels, conf_mat, metrics,
                 elapsed, cfg, log):
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("STEP 6 — Saving results")
    log.info("=" * 70)

    if cfg["save_results"]:
        results_df = pd.DataFrame(results)
        results_df["true_label"]      = y_true
        results_df["predicted_label"] = y_pred
        results_df["correct"]         = results_df["true_label"] == results_df["predicted_label"]

        results_path = out_dir / "evaluation_results.csv"
        results_df.to_csv(results_path, index=False)
        log.info(f"Per-example results saved : {results_path}")

        m = metrics
        metrics_df = pd.DataFrame({
            "Metric": [
                "Model",
                "Accuracy",
                "Precision (Macro)",
                "Precision (Weighted)",
                "Recall (Macro)",
                "Recall (Weighted)",
                "F1-Score (Macro)",
                "F1-Score (Weighted)",
                "Exact Match Accuracy",
                "Avg Partial Match",
                "Avg F1 (Word-level)",
            ],
            "Score": [
                cfg["display_name"] + " (fine-tuned)",
                m["accuracy"],
                m["prec_macro"],
                m["prec_weighted"],
                m["rec_macro"],
                m["rec_weighted"],
                m["f1_macro"],
                m["f1_weighted"],
                m["exact_match_acc"],
                m["avg_partial"],
                m["avg_word_f1"],
            ],
        })
        metrics_path = out_dir / "metrics_summary.csv"
        metrics_df.to_csv(metrics_path, index=False)
        log.info(f"Metrics summary saved     : {metrics_path}")

    # ── Figure 1: Metrics bar charts ─────────────────────────────────────────
    log.info("Creating metrics bar chart...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Fine-Tuned Model Evaluation — {cfg['display_name']}",
        fontsize=15, fontweight="bold", y=1.02,
    )

    m      = metrics
    names1 = ["Accuracy", "Precision\n(Weighted)", "Recall\n(Weighted)", "F1-Score\n(Weighted)"]
    vals1  = [m["accuracy"], m["prec_weighted"], m["rec_weighted"], m["f1_weighted"]]
    bars1  = ax1.bar(names1, vals1, color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"], alpha=0.8)
    ax1.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax1.set_title("Overall Performance Metrics", fontsize=14, fontweight="bold")
    ax1.set_ylim([0, 1])
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    ax1.grid(axis="y", alpha=0.3)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., h,
                 f"{h:.3f}\n({h * 100:.1f}%)", ha="center", va="bottom", fontweight="bold")

    comparison = {
        "Precision": [m["prec_macro"],  m["prec_weighted"]],
        "Recall":    [m["rec_macro"],   m["rec_weighted"]],
        "F1-Score":  [m["f1_macro"],    m["f1_weighted"]],
    }
    x      = np.arange(len(comparison))
    w      = 0.35
    bars2a = ax2.bar(x - w / 2, [v[0] for v in comparison.values()],
                     w, label="Macro",    color="#3498db", alpha=0.8)
    bars2b = ax2.bar(x + w / 2, [v[1] for v in comparison.values()],
                     w, label="Weighted", color="#e74c3c", alpha=0.8)
    ax2.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax2.set_title("Macro vs Weighted Metrics", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(comparison.keys())
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    for bars in [bars2a, bars2b]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., h,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig1_path = out_dir / "metrics.png"
    plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Metrics chart saved       : {fig1_path}")

    # ── Figure 2: Confusion matrix ────────────────────────────────────────────
    log.info("Creating confusion matrix...")
    n_labels = len(unique_labels)
    fig2, ax = plt.subplots(figsize=(max(8, n_labels * 2), max(6, n_labels * 2)))

    if n_labels > 20:
        top_labels = [lbl for lbl, _ in Counter(y_true).most_common(20)]
        idx        = [unique_labels.index(lbl) for lbl in top_labels]
        cm_sub     = conf_mat[np.ix_(idx, idx)]
        sns.heatmap(cm_sub, annot=True, fmt="d", cmap="Blues",
                    xticklabels=top_labels, yticklabels=top_labels, ax=ax)
        ax.set_title(f"Confusion Matrix (Top 20) — {cfg['display_name']} Fine-Tuned",
                     fontsize=14, fontweight="bold", pad=20)
    else:
        row_sums = conf_mat.sum(axis=1, keepdims=True)
        cm_norm  = np.where(row_sums > 0, conf_mat / row_sums, 0)
        annot    = np.array([
            [f"{conf_mat[i, j]}\n({cm_norm[i, j] * 100:.1f}%)"
             for j in range(conf_mat.shape[1])]
            for i in range(conf_mat.shape[0])
        ])
        sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues",
                    xticklabels=unique_labels, yticklabels=unique_labels,
                    ax=ax, vmin=0, vmax=1)
        ax.set_title(f"Confusion Matrix — {cfg['display_name']} Fine-Tuned (row-normalised)",
                     fontsize=14, fontweight="bold", pad=20)

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label",      fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig2_path = out_dir / "confusion_matrix.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    log.info(f"Confusion matrix saved    : {fig2_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_final_summary(metrics, eval_set, elapsed, cfg, log):
    m = metrics
    log.info("=" * 70)
    log.info("FINAL EVALUATION SUMMARY — FINE-TUNED MODEL")
    log.info("=" * 70)
    log.info(f"  Model              : {cfg['display_name']}  (fine-tuned from {MODEL_REGISTRY[cfg['model_key']]['model_id']})")
    log.info(f"  Model path         : {cfg['model_path']}")
    log.info(f"  Results folder     : {cfg['output_dir']}")
    log.info(f"  Samples            : {len(eval_set):,}")
    log.info(f"  Time               : {elapsed / 60:.2f} min  ({elapsed / len(eval_set):.2f} sec/example)")
    log.info("")
    log.info("  Status classification:")
    log.info(f"    Accuracy             : {m['accuracy']:.4f}  ({m['accuracy'] * 100:.2f}%)")
    log.info(f"    Precision (Weighted) : {m['prec_weighted']:.4f}")
    log.info(f"    Recall    (Weighted) : {m['rec_weighted']:.4f}")
    log.info(f"    F1-Score  (Weighted) : {m['f1_weighted']:.4f}")
    log.info(f"    Precision (Macro)    : {m['prec_macro']:.4f}")
    log.info(f"    Recall    (Macro)    : {m['rec_macro']:.4f}")
    log.info(f"    F1-Score  (Macro)    : {m['f1_macro']:.4f}")
    log.info("")
    log.info("  Output quality:")
    log.info(f"    Exact Match Accuracy : {m['exact_match_acc']:.4f}  ({m['exact_match_acc'] * 100:.2f}%)")
    log.info(f"    Avg Partial Match    : {m['avg_partial']:.4f}")
    log.info(f"    Avg Word-level F1    : {m['avg_word_f1']:.4f}")
    log.info("")
    log.info("  Compare with Base-LLM-Evaluation/results/ to measure fine-tuning gain.")
    log.info("=" * 70)
    log.info("Done.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — ARGUMENT PARSING & MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a fine-tuned model on the MITRE ATT&CK dataset."
    )
    p.add_argument("--model",             required=True,
                   choices=list(MODEL_REGISTRY.keys()),
                   help="Model family: qwen | llama | phi")
    p.add_argument("--model-path",        required=True,
                   help="Path to fine-tuned model / LoRA adapters (output of fine_tune.py)")
    p.add_argument("--data-path",         default=None,
                   help="Local directory with JSONL splits (skips Kaggle download)")
    p.add_argument("--splits",            default=None, nargs="+",
                   choices=["train", "val", "test"],
                   help="Splits to evaluate (default: test)")
    p.add_argument("--eval-limit",        default=None,
                   help="Max samples; pass 'None' for the full set (default: 2000)")
    p.add_argument("--suspicious-ratio",  default=None, type=float,
                   help="Fraction of suspicious samples (default: 0.30)")
    p.add_argument("--output-dir",        default=None,
                   help="Where to write results (default: results/{ModelName})")
    p.add_argument("--no-save",           action="store_true",
                   help="Print metrics only, skip CSV/PNG output")
    return p.parse_args()


def apply_args(cfg: dict, args) -> dict:
    reg = MODEL_REGISTRY[args.model]
    cfg["model_key"]    = args.model
    cfg["model_name"]   = reg["model_id"]
    cfg["display_name"] = reg["display_name"]
    cfg["model_path"]   = args.model_path

    cfg["output_dir"] = args.output_dir or os.path.join("results", reg["display_name"])

    if args.data_path:
        cfg["data_path"] = args.data_path
    if args.splits:
        cfg["splits"] = args.splits
    if args.eval_limit is not None:
        cfg["eval_limit"] = None if args.eval_limit.lower() == "none" else int(args.eval_limit)
    if args.suspicious_ratio is not None:
        cfg["suspicious_ratio"] = args.suspicious_ratio
    if args.no_save:
        cfg["save_results"] = False
    return cfg


def main():
    args = parse_args()
    cfg  = apply_args(CONFIG, args)

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    log = setup_logging(cfg["output_dir"])
    log.info(f"PID {os.getpid()} — metrics.py starting")
    log.info(f"Model        : {cfg['display_name']}  ({cfg['model_path']})")
    log.info(f"Output dir   : {cfg['output_dir']}")
    log.info(f"Python {sys.version.split()[0]}  |  cwd: {os.getcwd()}")

    log.info("=" * 70)
    log.info("CONFIGURATION")
    log.info("=" * 70)
    for k, v in cfg.items():
        log.info(f"  {k:<28}: {v}")

    try:
        model, tokenizer = load_model_and_tokenizer(cfg, log)
        dataset          = load_data(cfg, log)
        eval_set         = build_eval_set(dataset, cfg, log)

        results, exact_total, partial_scores, wf1_scores, elapsed = run_evaluation(
            eval_set, model, tokenizer, cfg, log
        )
        metrics, y_true, y_pred, unique_labels, conf_mat = compute_metrics(
            results, exact_total, partial_scores, wf1_scores, elapsed, cfg, log
        )
        save_outputs(results, y_true, y_pred, unique_labels, conf_mat, metrics,
                     elapsed, cfg, log)
        print_final_summary(metrics, eval_set, elapsed, cfg, log)

    except Exception:
        log.exception("Evaluation failed with an unhandled exception.")
        sys.exit(1)


if __name__ == "__main__":
    main()
