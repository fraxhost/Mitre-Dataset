"""
Base LLM Evaluation — Qwen/Qwen2.5-1.5B-Instruct (no fine-tuning)
===================================================================
Evaluates the base model on the same test set and metrics as
Fine-Tune/metrics.ipynb so results are directly comparable.

Uses vLLM offline batch inference — all prompts are submitted in a
single llm.generate() call and processed in parallel across all GPUs.
Designed for headless / nohup execution on NVIDIA hardware.

Optimised for 4 × NVIDIA RTX Ada 6000 (48 GB each, 192 GB total VRAM).
Also works on any other CUDA setup by adjusting --tensor-parallel-size.

  - All output is logged with timestamps to stdout AND a log file
  - Matplotlib uses the Agg backend (no display needed)
  - All figures are saved to disk, not shown interactively

Usage
-----
  # Foreground (see live output):
  python base_model_eval.py

  # Background with nohup:
  nohup python base_model_eval.py > eval.log 2>&1 &
  echo "PID: $!"
  tail -f eval.log            # follow stdout
  tail -f base_eval_run.log   # same content in the dedicated log file

  # Override key settings from the CLI:
  python base_model_eval.py --eval-limit 500 --tensor-parallel-size 4
  python base_model_eval.py --eval-limit None          # full test set
  python base_model_eval.py --output-dir ./results
"""

# ── Matplotlib must switch to non-interactive backend BEFORE any other import ──
import matplotlib
matplotlib.use("Agg")

# ── Standard library ────────────────────────────────────────────────────────────
import argparse
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# ── Third-party ─────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from vllm import LLM, SamplingParams


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION  (edit here or override with CLI flags)
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Model ─────────────────────────────────────────────────────────────────
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",   # base model, no LoRA adapters

    # ── Data paths ────────────────────────────────────────────────────────────
    "data_path": "./test_data",
    "test_file": "test.jsonl",

    # ── Evaluation ────────────────────────────────────────────────────────────
    "eval_limit": 2000,          # set to None to evaluate the full test set
    "suspicious_ratio": 0.30,    # fraction of suspicious samples when eval_limit is set
    "prioritize_shortest": True, # process shortest logs first (faster on limited hardware)
    "max_new_tokens": 512,
    "temperature": 0.7,          # matches Fine-Tune/metrics.ipynb
    "top_p": 0.9,                # matches Fine-Tune/metrics.ipynb

    # ── Prompt / tokenisation ─────────────────────────────────────────────────
    "max_input_chars": 6000,     # truncate input JSON beyond this character count

    # ── vLLM inference ────────────────────────────────────────────────────────
    # tensor_parallel_size: spread the model across N GPUs.
    #   4 × Ada 6000 (48 GB each, 192 GB total)  →  4
    #   2 × GPU                                   →  2
    #   1 × GPU                                   →  1
    "tensor_parallel_size": 4,

    # gpu_memory_utilization: fraction of VRAM vLLM may use per GPU.
    # 0.90 is the recommended default — leaves a small buffer for CUDA overhead.
    "gpu_memory_utilization": 0.90,

    # dtype: "float16" or "bfloat16".
    # Ada 6000 (Ampere-class) supports both; bfloat16 is slightly more numerically
    # stable for LLM inference. Use "float16" if you encounter NaN/inf issues.
    "dtype": "bfloat16",

    # ── Output ────────────────────────────────────────────────────────────────
    "show_examples": 10,
    "save_results": True,
    "output_dir": ".",           # directory for all output files
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: str) -> logging.Logger:
    """
    Write timestamped logs to both stdout and a dedicated log file so that
    nohup output and a live `tail -f` both work simultaneously.
    """
    log_path = Path(output_dir) / "base_eval_run.log"
    fmt      = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt  = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt, handlers=handlers)
    logger = logging.getLogger("base_eval")
    logger.info(f"Log file: {log_path.resolve()}")
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(instruction: str, input_text: str, max_input_chars: int) -> str:
    """
    Identical prompt format to training data and Fine-Tune/metrics.ipynb:
      {instruction}\\n\\n{input_text}\\n\\nAnalysis:\\n
    """
    if len(input_text) > max_input_chars:
        input_text = input_text[:max_input_chars] + "... [truncated]"
    return f"{instruction}\n\n{input_text}\n\nAnalysis:\n"


def extract_status_label(text: str) -> str:
    """
    Parse NORMAL / SUSPICIOUS / UNKNOWN from model output.
    Identical logic to Fine-Tune/metrics.ipynb for comparability.
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

    # Fallback: plain-text format from training data
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
# SECTION 4 — PIPELINE STEPS
# ══════════════════════════════════════════════════════════════════════════════

def load_model(cfg: dict, log: logging.Logger) -> LLM:
    """
    Initialise vLLM with the base Qwen2.5-1.5B-Instruct model.

    vLLM automatically detects all visible CUDA GPUs and distributes the
    model across tensor_parallel_size of them.  On 4 × Ada 6000 (48 GB
    each) this gives 192 GB of VRAM to work with, far more than needed
    for a 1.5 B parameter model — but tensor parallelism also maximises
    throughput by parallelising the KV-cache across devices.
    """
    log.info("=" * 70)
    log.info("STEP 1 — Loading base model with vLLM")
    log.info("=" * 70)
    log.info(f"Model                  : {cfg['model_name']}  (base — no LoRA adapters)")
    log.info(f"tensor_parallel_size   : {cfg['tensor_parallel_size']}")
    log.info(f"gpu_memory_utilization : {cfg['gpu_memory_utilization']}")
    log.info(f"dtype                  : {cfg['dtype']}")

    # Log GPU topology if nvidia-smi is available
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        for i, line in enumerate(smi.splitlines()):
            log.info(f"  GPU {i}: {line}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    llm = LLM(
        model=cfg["model_name"],
        tensor_parallel_size=cfg["tensor_parallel_size"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        dtype=cfg["dtype"],
        trust_remote_code=True,
        # Swap space (GB) on CPU RAM for KV cache overflow — rarely needed for
        # a 1.5 B model but set a small value for safety.
        swap_space=4,
    )

    log.info("vLLM engine initialised successfully.")
    return llm


def load_data(cfg: dict, log: logging.Logger):
    test_file_path = os.path.join(cfg["data_path"], cfg["test_file"])

    log.info("=" * 70)
    log.info("STEP 2 — Loading test dataset")
    log.info("=" * 70)
    log.info(f"Path: {test_file_path}")

    if not os.path.exists(test_file_path):
        raise FileNotFoundError(
            f"Test file not found: {test_file_path}\n"
            "Set --data-path to the directory containing test.jsonl."
        )

    dataset = load_dataset("json", data_files={"test": test_file_path})["test"]
    log.info(f"Loaded {len(dataset):,} examples.  Columns: {dataset.column_names}")

    sample = dataset[0]
    log.info(f"Sample instruction : {sample['instruction'][:100]}")
    log.info(f"Sample input       : {sample['input'][:120]}...")
    log.info(f"Sample output      : {sample['output'][:120]}...")
    return dataset


def build_eval_set(dataset, cfg: dict, log: logging.Logger):
    log.info("=" * 70)
    log.info("STEP 3 — Stratified sampling")
    log.info("=" * 70)

    if cfg["eval_limit"] is None:
        log.info(f"Using full test set: {len(dataset):,} examples")
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

    if sel_susp:
        log.info(
            f"Suspicious lengths: "
            f"{min(s['length'] for s in sel_susp):,} – "
            f"{max(s['length'] for s in sel_susp):,} chars"
        )
    if sel_normal:
        log.info(
            f"Normal lengths:     "
            f"{min(s['length'] for s in sel_normal):,} – "
            f"{max(s['length'] for s in sel_normal):,} chars"
        )

    indices  = [s["index"] for s in sel_susp + sel_normal]
    eval_set = dataset.select(indices)
    log.info(f"Final eval set: {len(eval_set):,} examples")
    return eval_set


def run_inference(eval_set, llm: LLM, cfg: dict, log: logging.Logger):
    log.info("=" * 70)
    log.info("STEP 4 — Building prompts")
    log.info("=" * 70)

    prompts  = [
        build_prompt(ex["instruction"], ex["input"], cfg["max_input_chars"])
        for ex in eval_set
    ]
    expected = [ex["output"] for ex in eval_set]
    log.info(f"{len(prompts):,} prompts built.")
    log.info(f"Sample prompt (first 300 chars):\n{prompts[0][:300]}...")

    log.info("=" * 70)
    log.info("STEP 5 — Offline batch inference (vLLM)")
    log.info("=" * 70)
    log.info(
        f"tensor_parallel_size: {cfg['tensor_parallel_size']}  |  "
        f"temp: {cfg['temperature']}  top_p: {cfg['top_p']}  "
        f"max_new_tokens: {cfg['max_new_tokens']}"
    )
    log.info(
        "Submitting all prompts in a single llm.generate() call — "
        "vLLM handles continuous batching internally."
    )

    sampling_params = SamplingParams(
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        max_tokens=cfg["max_new_tokens"],
    )

    t0      = time.time()
    outputs = llm.generate(prompts, sampling_params)   # blocking until all done
    elapsed = time.time() - t0

    # Extract generated text from vLLM RequestOutput objects
    predictions = [out.outputs[0].text.strip() for out in outputs]

    log.info(
        f"Inference complete: {elapsed / 60:.2f} min total  "
        f"({elapsed / len(prompts):.2f} sec/example)"
    )
    return predictions, expected, elapsed


def compute_metrics(predictions, expected, eval_set, elapsed, cfg, log):
    log.info("=" * 70)
    log.info("STEP 6 — Computing per-example metrics")
    log.info("=" * 70)

    results           = []
    exact_match_total = 0.0
    partial_scores    = []
    word_f1_scores    = []

    for i, (pred, exp) in enumerate(zip(predictions, expected)):
        em      = calculate_exact_match(pred, exp)
        partial = calculate_partial_match(pred, exp)
        wf1     = calculate_word_f1(pred, exp)

        exact_match_total += em
        partial_scores.append(partial)
        word_f1_scores.append(wf1)

        results.append({
            "index":         i,
            "instruction":   eval_set[i]["instruction"],
            "input":         eval_set[i]["input"],
            "expected":      exp,
            "predicted":     pred,
            "exact_match":   em,
            "partial_match": partial,
            "f1_score":      wf1,
        })

    # Log sample predictions
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

    # ── Classification metrics ────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("STEP 7 — Classification metrics")
    log.info("=" * 70)

    y_true = [extract_status_label(r["expected"])  for r in results]
    y_pred = [extract_status_label(r["predicted"]) for r in results]

    unique_labels = sorted(set(y_true + y_pred))
    log.info(f"Labels found: {unique_labels}")
    log.info(f"True  dist : {dict(Counter(y_true))}")
    log.info(f"Pred  dist : {dict(Counter(y_pred))}")

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

    # ── Prediction breakdown ──────────────────────────────────────────────────
    unknown_count    = y_pred.count("UNKNOWN")
    normal_count     = y_pred.count("NORMAL")
    suspicious_count = y_pred.count("SUSPICIOUS")
    log.info("PREDICTION BREAKDOWN")
    log.info(f"  SUSPICIOUS : {suspicious_count:,}  ({suspicious_count / len(results) * 100:.1f}%)")
    log.info(f"  NORMAL     : {normal_count:,}  ({normal_count / len(results) * 100:.1f}%)")
    log.info(f"  UNKNOWN    : {unknown_count:,}  ({unknown_count / len(results) * 100:.1f}%)")

    incorrect = [(yt, yp, r) for yt, yp, r in zip(y_true, y_pred, results) if yt != yp]
    log.info(f"  Correct   : {len(results) - len(incorrect):,} / {len(results):,}")
    log.info(f"  Incorrect : {len(incorrect):,} / {len(results):,}")
    if incorrect:
        yt, yp, r = incorrect[0]
        log.info("  Sample incorrect prediction:")
        log.info(f"    True label : {yt}  |  Pred label : {yp}")
        log.info(f"    Input      : {r['input'][:120]}...")
        log.info(f"    Expected   : {r['expected'][:150]}...")
        log.info(f"    Predicted  : {r['predicted'][:150]}...")

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
    return results, y_true, y_pred, unique_labels, conf_mat, metrics


def save_outputs(results, y_true, y_pred, unique_labels, conf_mat, metrics,
                 elapsed, eval_set, cfg, log):
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("STEP 8 — Saving results")
    log.info("=" * 70)

    if cfg["save_results"]:
        # ── CSV: per-example results ──────────────────────────────────────────
        results_df = pd.DataFrame(results)
        results_df["true_label"]      = y_true
        results_df["predicted_label"] = y_pred
        results_df["correct"]         = results_df["true_label"] == results_df["predicted_label"]

        results_path = out_dir / "base_model_evaluation_results.csv"
        results_df.to_csv(results_path, index=False)
        log.info(f"Per-example results saved : {results_path}")

        # ── CSV: metrics summary ──────────────────────────────────────────────
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
                cfg["model_name"] + " (base)",
                metrics["accuracy"],
                metrics["prec_macro"],
                metrics["prec_weighted"],
                metrics["rec_macro"],
                metrics["rec_weighted"],
                metrics["f1_macro"],
                metrics["f1_weighted"],
                metrics["exact_match_acc"],
                metrics["avg_partial"],
                metrics["avg_word_f1"],
            ],
        })
        metrics_path = out_dir / "base_model_metrics_summary.csv"
        metrics_df.to_csv(metrics_path, index=False)
        log.info(f"Metrics summary saved     : {metrics_path}")

        log.info("Sample results (first 10):")
        log.info(
            "\n" + results_df[["true_label", "predicted_label", "correct", "f1_score"]]
            .head(10).to_string(index=False)
        )

    # ── Figure 1: Overall + Macro vs Weighted bar charts ─────────────────────
    log.info("Creating metrics bar chart...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Base Model Evaluation — {cfg['model_name']}",
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
    fig1_path = out_dir / "base_model_metrics.png"
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
        ax.set_title("Confusion Matrix (Top 20 Labels) — Base Model",
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
        ax.set_title("Confusion Matrix — Base Model (row-normalised)",
                     fontsize=14, fontweight="bold", pad=20)

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label",      fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig2_path = out_dir / "base_model_confusion_matrix.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    log.info(f"Confusion matrix saved    : {fig2_path}")


def print_final_summary(metrics, eval_set, elapsed, cfg, log):
    m = metrics
    log.info("=" * 70)
    log.info("FINAL EVALUATION SUMMARY — BASE MODEL")
    log.info("=" * 70)
    log.info(f"  Model              : {cfg['model_name']}  (base — no fine-tuning)")
    log.info(f"  Inference engine   : vLLM  (tensor_parallel_size={cfg['tensor_parallel_size']})")
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
    log.info("  Compare with Fine-Tune/metrics.ipynb to measure the fine-tuning gain.")
    log.info("=" * 70)
    log.info("Done.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate base Qwen2.5-1.5B on the MITRE ATT&CK test set "
                    "using vLLM offline batch inference on NVIDIA GPUs."
    )
    p.add_argument("--model-name",              default=None,
                   help="HuggingFace model ID (default: Qwen/Qwen2.5-1.5B-Instruct)")
    p.add_argument("--data-path",               default=None,
                   help="Directory containing test.jsonl")
    p.add_argument("--eval-limit",              default=None,
                   help="Max samples; pass 'None' for the full set (default: 2000)")
    p.add_argument("--suspicious-ratio",        default=None, type=float,
                   help="Fraction of suspicious samples in the eval set (default: 0.30)")
    p.add_argument("--max-new-tokens",          default=None, type=int,
                   help="Max tokens to generate per response (default: 512)")
    p.add_argument("--temperature",             default=None, type=float,
                   help="Sampling temperature (default: 0.7)")
    p.add_argument("--tensor-parallel-size",    default=None, type=int,
                   help="Number of GPUs for tensor parallelism (default: 4)")
    p.add_argument("--gpu-memory-utilization",  default=None, type=float,
                   help="Fraction of VRAM vLLM may use per GPU (default: 0.90)")
    p.add_argument("--dtype",                   default=None,
                   choices=["float16", "bfloat16", "float32"],
                   help="Model weight dtype (default: bfloat16)")
    p.add_argument("--output-dir",              default=None,
                   help="Directory for all output files (default: current dir)")
    p.add_argument("--no-save",                 action="store_true",
                   help="Skip saving CSV output files")
    return p.parse_args()


def apply_args(cfg: dict, args) -> dict:
    """Merge CLI overrides into CONFIG."""
    if args.model_name:
        cfg["model_name"] = args.model_name
    if args.data_path:
        cfg["data_path"] = args.data_path
    if args.eval_limit is not None:
        cfg["eval_limit"] = None if args.eval_limit.lower() == "none" else int(args.eval_limit)
    if args.suspicious_ratio is not None:
        cfg["suspicious_ratio"] = args.suspicious_ratio
    if args.max_new_tokens is not None:
        cfg["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if args.tensor_parallel_size is not None:
        cfg["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization is not None:
        cfg["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.dtype is not None:
        cfg["dtype"] = args.dtype
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.no_save:
        cfg["save_results"] = False
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    cfg  = apply_args(CONFIG, args)

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    log = setup_logging(cfg["output_dir"])
    log.info(f"PID {os.getpid()} — base_model_eval.py starting")
    log.info(f"Python {sys.version.split()[0]}  |  cwd: {os.getcwd()}")

    log.info("=" * 70)
    log.info("CONFIGURATION")
    log.info("=" * 70)
    for k, v in cfg.items():
        log.info(f"  {k:<28}: {v}")

    try:
        llm                            = load_model(cfg, log)
        dataset                        = load_data(cfg, log)
        eval_set                       = build_eval_set(dataset, cfg, log)
        predictions, expected, elapsed = run_inference(eval_set, llm, cfg, log)
        results, y_true, y_pred, unique_labels, conf_mat, metrics = compute_metrics(
            predictions, expected, eval_set, elapsed, cfg, log
        )
        save_outputs(results, y_true, y_pred, unique_labels, conf_mat, metrics,
                     elapsed, eval_set, cfg, log)
        print_final_summary(metrics, eval_set, elapsed, cfg, log)

    except Exception:
        log.exception("Evaluation failed with an unhandled exception.")
        sys.exit(1)


if __name__ == "__main__":
    main()
