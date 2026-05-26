"""
Multi-model Results Comparison
===============================
Auto-discovers every model folder under Fine-Tune/results/, loads its
metrics_summary.csv (fine-tuned) and the corresponding base-model
metrics_summary.csv from Base-LLM-Evaluation/results/ (when available),
then produces:

  1. A side-by-side terminal table (all 10 numeric metrics + delta)
  2. comparison_summary.csv  — wide CSV for downstream analysis
  3. comparison.png          — grouped bar chart across all 10 metrics
  4. confusion_matrix_comparison.png — grid of per-model confusion matrices
                                       (base vs fine-tuned)

Usage
-----
  # Full comparison (all discovered models)
  python compare_results.py

  # Print only, no files written
  python compare_results.py --no-save

  # Restrict to specific models
  python compare_results.py --models Llama-3.2-3B Phi-4-mini

  # Custom result directories
  python compare_results.py \\
      --results-dir /path/to/Fine-Tune/results \\
      --base-results-dir /path/to/Base-LLM-Evaluation/results
"""

# ── Matplotlib non-interactive backend before any other import ───────────────
import matplotlib
matplotlib.use("Agg")

# ── Standard library ─────────────────────────────────────────────────────────
import argparse
import logging
import os
import sys
from pathlib import Path

# ── Third-party ──────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Directories ──────────────────────────────────────────────────────────
    "results_dir":      "results",                       # Fine-Tune/results/{Model}/
    "base_results_dir": "../Base-LLM-Evaluation/results",# base/{Model}/
    "output_dir":       "results",                       # comparison outputs land here
    # ── Behaviour ────────────────────────────────────────────────────────────
    "save_outputs": True,   # overridden by --no-save
    "models":       None,   # None = auto-discover; list = restrict to these names
}

# Ordered list of all numeric metric row names (as they appear in metrics_summary.csv)
NUMERIC_METRICS = [
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
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: str) -> logging.Logger:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "compare_run.log"
    fmt      = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt  = "%Y-%m-%d %H:%M:%S"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt,
                        handlers=handlers, force=True)
    logger = logging.getLogger("compare_results")
    logger.info(f"Log file: {log_path.resolve()}")
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def discover_models(results_dir: str, log: logging.Logger) -> list:
    """
    Return a sorted list of model names found in results_dir.
    A valid model directory must contain metrics_summary.csv.
    """
    base = Path(results_dir)
    if not base.exists():
        log.error(f"Results directory not found: {base.resolve()}")
        return []

    models = sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and (d / "metrics_summary.csv").exists()
    )
    log.info(f"Discovered {len(models)} model(s) in {base.resolve()}: {models}")
    return models


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — METRICS LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_model_metrics(
    model: str,
    results_dir: str,
    base_results_dir: str,
    log: logging.Logger,
) -> pd.DataFrame:
    """
    Load fine-tuned and (optionally) base metrics for one model.

    Returns a DataFrame with columns:
        Metric | {model}_finetuned | {model}_base (opt.) | {model}_delta (opt.)
    """
    ft_path   = Path(results_dir) / model / "metrics_summary.csv"
    base_path = Path(base_results_dir) / model / "base_model_metrics_summary.csv"

    # Fine-tuned (required)
    ft_df = pd.read_csv(ft_path)
    ft_df = ft_df.rename(columns={"Score": f"{model}_finetuned"})

    merged = ft_df.copy()

    # Base (optional)
    if base_path.exists():
        base_df = pd.read_csv(base_path)
        base_df = base_df.rename(columns={"Score": f"{model}_base"})
        merged  = merged.merge(base_df, on="Metric", how="outer")

        # Delta (numeric rows only)
        num_mask = merged["Metric"].isin(NUMERIC_METRICS)
        ft_vals  = pd.to_numeric(merged.loc[num_mask, f"{model}_finetuned"], errors="coerce")
        b_vals   = pd.to_numeric(merged.loc[num_mask, f"{model}_base"],      errors="coerce")
        merged.loc[num_mask, f"{model}_delta"] = ft_vals - b_vals
        log.info(f"  {model}: fine-tuned ✓  base ✓  (delta computed)")
    else:
        log.info(f"  {model}: fine-tuned ✓  base — not found at {base_path}")

    return merged


def build_comparison_table(
    models: list,
    results_dir: str,
    base_results_dir: str,
    log: logging.Logger,
) -> pd.DataFrame:
    """
    Merge per-model DataFrames on 'Metric' (outer join).
    Column order: Metric | M1_base | M1_finetuned | M1_delta | M2_base | ...
    """
    log.info("=" * 70)
    log.info("STEP 1 — Loading metrics")
    log.info("=" * 70)

    frames = []
    for m in models:
        df = load_model_metrics(m, results_dir, base_results_dir, log)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = frames[0]
    for df in frames[1:]:
        combined = combined.merge(df, on="Metric", how="outer")

    # Enforce canonical row order: Model row first, then numeric metrics
    order = ["Model"] + NUMERIC_METRICS
    combined = combined.set_index("Metric").reindex(order).reset_index()

    # Re-order columns for readability: base → finetuned → delta per model
    col_order = ["Metric"]
    for m in models:
        for suffix in ("_base", "_finetuned", "_delta"):
            col = f"{m}{suffix}"
            if col in combined.columns:
                col_order.append(col)
    combined = combined[[c for c in col_order if c in combined.columns]]

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame, models: list, log: logging.Logger) -> None:
    """Print a formatted comparison table to stdout / log."""
    log.info("=" * 70)
    log.info("COMPARISON TABLE")
    log.info("=" * 70)

    if df.empty:
        log.warning("No data to display.")
        return

    # Separate Model (string) row from numeric rows
    model_row  = df[df["Metric"] == "Model"]
    metric_rows = df[df["Metric"] != "Model"].copy()

    # Format numeric columns to 4 decimal places; delta gets a sign prefix
    display = metric_rows.copy()
    for col in display.columns:
        if col == "Metric":
            continue
        vals = pd.to_numeric(display[col], errors="coerce")
        if "_delta" in col:
            display[col] = vals.apply(
                lambda v: f"{v:+.4f}" if pd.notna(v) else "—"
            )
        else:
            display[col] = vals.apply(
                lambda v: f"{v:.4f}" if pd.notna(v) else "—"
            )

    # Shorten column headers for terminal readability
    short_cols = {}
    for c in display.columns:
        if c == "Metric":
            short_cols[c] = "Metric"
            continue
        for m in models:
            if c.startswith(m):
                suffix = c[len(m):]  # _base / _finetuned / _delta
                label  = suffix.replace("_finetuned", " FT").replace("_base", " Base").replace("_delta", " Δ")
                short_cols[c] = f"{m}{label}"
                break
    display = display.rename(columns=short_cols)

    # Model identifiers header
    if not model_row.empty:
        log.info("Model identifiers:")
        for col in model_row.columns[1:]:
            val = model_row.iloc[0][col]
            if pd.notna(val) and str(val) not in ("nan", "—"):
                short = short_cols.get(col, col)
                log.info(f"  {short:<40}: {val}")
        log.info("")

    log.info(display.to_string(index=False))
    log.info("")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SAVE CSV
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(df: pd.DataFrame, output_dir: str, log: logging.Logger) -> None:
    out_path = Path(output_dir) / "comparison_summary.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Comparison CSV saved      : {out_path.resolve()}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — BAR CHART (all 10 numeric metrics)
# ══════════════════════════════════════════════════════════════════════════════

def save_chart(
    df: pd.DataFrame,
    models: list,
    output_dir: str,
    log: logging.Logger,
) -> None:
    """Grouped bar chart — all 10 numeric metrics, base vs fine-tuned per model."""
    log.info("Creating metrics bar chart...")

    numeric_df = df[df["Metric"].isin(NUMERIC_METRICS)].copy()
    if numeric_df.empty:
        log.warning("No numeric metric data; skipping bar chart.")
        return

    metric_labels = list(numeric_df["Metric"])
    x = np.arange(len(metric_labels))

    # Collect series to plot: (label, values_array, colour)
    series = []
    # Use a colour palette: lighter shade for base, darker for fine-tuned
    palette_base = [
        "#a8d8ea", "#a8e6cf", "#ffd3b6",   # light blue, light green, light orange
    ]
    palette_ft = [
        "#1a73e8", "#1e8449", "#e67e22",    # blue, green, orange
    ]
    for i, m in enumerate(models):
        bc = palette_base[i % len(palette_base)]
        fc = palette_ft[i % len(palette_ft)]
        base_col = f"{m}_base"
        ft_col   = f"{m}_finetuned"
        if base_col in numeric_df.columns:
            series.append((f"{m} base",       pd.to_numeric(numeric_df[base_col], errors="coerce").values, bc))
        if ft_col in numeric_df.columns:
            series.append((f"{m} fine-tuned", pd.to_numeric(numeric_df[ft_col],   errors="coerce").values, fc))

    n_series = len(series)
    width     = 0.8 / n_series       # bar width so the group fits in 0.8 units
    offsets   = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * width

    fig, ax = plt.subplots(figsize=(22, 7))
    fig.suptitle("Base vs Fine-Tuned Model Comparison — All Metrics",
                 fontsize=15, fontweight="bold")

    for (label, values, colour), offset in zip(series, offsets):
        bars = ax.bar(x + offset, values, width, label=label, color=colour,
                      edgecolor="white", linewidth=0.5, alpha=0.9)
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=6.5, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = Path(output_dir) / "comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Bar chart saved           : {out_path.resolve()}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — CONFUSION MATRIX COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def save_confusion_matrix_comparison(
    models: list,
    results_dir: str,
    base_results_dir: str,
    output_dir: str,
    log: logging.Logger,
) -> None:
    """
    Load true_label / predicted_label from evaluation_results.csv for every
    model × variant pair, recompute confusion matrices, and save a subplot
    grid (rows=models, cols=base|fine-tuned).
    """
    log.info("Creating confusion matrix comparison...")

    # ── Collect all data ─────────────────────────────────────────────────────
    data = {}   # {(model, variant): (y_true, y_pred)}
    for model in models:
        ft_csv   = Path(results_dir)      / model / "evaluation_results.csv"
        base_csv = Path(base_results_dir) / model / "base_model_evaluation_results.csv"

        if ft_csv.exists():
            df = pd.read_csv(ft_csv, usecols=["true_label", "predicted_label"])
            data[(model, "fine-tuned")] = (
                df["true_label"].astype(str).tolist(),
                df["predicted_label"].astype(str).tolist(),
            )
        else:
            log.warning(f"  {model} fine-tuned evaluation_results.csv not found.")

        if base_csv.exists():
            df = pd.read_csv(base_csv, usecols=["true_label", "predicted_label"])
            data[(model, "base")] = (
                df["true_label"].astype(str).tolist(),
                df["predicted_label"].astype(str).tolist(),
            )
        else:
            log.info(f"  {model} base evaluation_results.csv not found (skipping base CM).")

    if not data:
        log.warning("No prediction data found; skipping confusion matrix comparison.")
        return

    # Unified label set across all models + variants
    all_labels = sorted({
        lbl
        for y_true, y_pred in data.values()
        for lbl in set(y_true) | set(y_pred)
    })
    log.info(f"  Confusion matrix labels: {all_labels}")

    # ── Build subplot grid: rows=models, cols=[base, fine-tuned] ─────────────
    variants = ["base", "fine-tuned"]
    n_rows   = len(models)
    n_cols   = len(variants)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5, n_rows * 4.5),
        squeeze=False,
    )
    fig.suptitle("Confusion Matrix Comparison — Base vs Fine-Tuned",
                 fontsize=14, fontweight="bold", y=1.01)

    for row_i, model in enumerate(models):
        for col_i, variant in enumerate(variants):
            ax = axes[row_i][col_i]
            key = (model, variant)

            if key not in data:
                ax.axis("off")
                ax.set_title(f"{model}\n({variant})", fontsize=9)
                continue

            y_true, y_pred = data[key]
            cm   = confusion_matrix(y_true, y_pred, labels=all_labels)
            # Row-normalise (avoid division by zero)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_norm  = cm / row_sums

            # Build annotation: "N\n(p%)" per cell
            annot = np.empty_like(cm, dtype=object)
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    annot[r, c] = f"{cm[r, c]}\n({cm_norm[r, c] * 100:.1f}%)"

            sns.heatmap(
                cm_norm,
                ax=ax,
                annot=annot,
                fmt="",
                cmap="Blues",
                xticklabels=all_labels,
                yticklabels=all_labels,
                vmin=0, vmax=1,
                linewidths=0.5,
                cbar=False,
            )
            ax.set_title(f"{model} — {variant}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("True", fontsize=9)
            ax.tick_params(axis="both", labelsize=8)

    plt.tight_layout()
    out_path = Path(output_dir) / "confusion_matrix_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Confusion matrix grid saved: {out_path.resolve()}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — ARGUMENT PARSING & MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare fine-tuned vs base-model results across all models "
                    "found in Fine-Tune/results/."
    )
    p.add_argument(
        "--results-dir",
        default=None,
        help="Fine-Tune results directory (default: results/)",
    )
    p.add_argument(
        "--base-results-dir",
        default=None,
        help="Base-LLM-Evaluation results directory "
             "(default: ../Base-LLM-Evaluation/results)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to write comparison outputs (default: results/)",
    )
    p.add_argument(
        "--models",
        default=None,
        nargs="+",
        metavar="MODEL",
        help="Restrict comparison to these model names (default: all discovered)",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Print table only; skip CSV and PNG output",
    )
    return p.parse_args()


def apply_args(cfg: dict, args) -> dict:
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.base_results_dir:
        cfg["base_results_dir"] = args.base_results_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.models:
        cfg["models"] = args.models
    if args.no_save:
        cfg["save_outputs"] = False
    return cfg


def main():
    args = parse_args()
    cfg  = apply_args(CONFIG, args)

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    log = setup_logging(cfg["output_dir"])

    log.info(f"PID {os.getpid()} — compare_results.py starting")
    log.info(f"Results dir      : {Path(cfg['results_dir']).resolve()}")
    log.info(f"Base results dir : {Path(cfg['base_results_dir']).resolve()}")
    log.info(f"Output dir       : {Path(cfg['output_dir']).resolve()}")

    # ── Discover or use specified models ─────────────────────────────────────
    if cfg["models"]:
        models = cfg["models"]
        log.info(f"Using specified models: {models}")
    else:
        models = discover_models(cfg["results_dir"], log)

    if not models:
        log.error("No models found. Exiting.")
        sys.exit(1)

    # ── Build comparison table ────────────────────────────────────────────────
    table = build_comparison_table(
        models, cfg["results_dir"], cfg["base_results_dir"], log
    )

    # ── Print ─────────────────────────────────────────────────────────────────
    print_summary(table, models, log)

    # ── Save ──────────────────────────────────────────────────────────────────
    if cfg["save_outputs"]:
        log.info("=" * 70)
        log.info("STEP 2 — Saving outputs")
        log.info("=" * 70)

        save_csv(table, cfg["output_dir"], log)

        save_chart(table, models, cfg["output_dir"], log)

        save_confusion_matrix_comparison(
            models,
            cfg["results_dir"],
            cfg["base_results_dir"],
            cfg["output_dir"],
            log,
        )

        log.info("=" * 70)
        log.info("Done.  All outputs written to: "
                 f"{Path(cfg['output_dir']).resolve()}")
        log.info("=" * 70)
    else:
        log.info("--no-save: skipping file output.")


if __name__ == "__main__":
    main()
