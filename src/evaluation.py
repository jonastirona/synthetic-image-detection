"""
Evaluation script

Reads results/predictions.csv and produces:
  results/metrics/summary.csv          — key accuracy/F1 numbers
  results/metrics/classification.txt   — full sklearn classification report
  results/figures/confusion_overall.png
  results/figures/confusion_by_prompt.png
  results/figures/accuracy_by_prompt.png
  results/figures/f1_by_class.png
  results/figures/confidence_analysis.png
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

ROOT = Path(__file__).parent.parent
PREDICTIONS = ROOT / "results/predictions.csv"
FIGURES = ROOT / "results/figures"
METRICS = ROOT / "results/metrics"
FIGURES.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)

CLASSES = ["real", "ai_generated", "deepfake"]
PROMPTS = ["zero_shot", "structured", "few_shot", "cot"]
FORMATS = ["label_only", "reasoning"]

PALETTE = {"real": "#4C9BE8", "ai_generated": "#E8944C", "deepfake": "#6DBF6D"}
PROMPT_COLORS = ["#5B8DB8", "#E8864C", "#6DBF6D", "#B85B8D"]

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


# ── Load data ─────────────────────────────────────────────────────────

def load() -> list[dict]:
    with open(PREDICTIONS) as f:
        rows = list(csv.DictReader(f))
    # Drop rows with unparsed labels
    rows = [r for r in rows if r["predicted_label"]]
    print(f"Loaded {len(rows)} rows ({480 - len(rows)} unparsed dropped)")
    return rows


# ── Confusion matrix helper ───────────────────────────────────────────

def plot_cm(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(
        cm_pct, annot=True, fmt=".0f", cmap="Blues",
        xticklabels=["real", "ai_gen", "deepfake"],
        yticklabels=["real", "ai_gen", "deepfake"],
        vmin=0, vmax=100, ax=ax, cbar=False,
        annot_kws={"size": 10},
    )
    # Overlay raw counts in smaller text
    for i in range(3):
        for j in range(3):
            ax.text(j + 0.5, i + 0.75, f"n={cm[i,j]}",
                    ha="center", va="center", fontsize=7, color="gray")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11, fontweight="bold")


# ── Figure 1: Overall confusion matrix ───────────────────────────────

def fig_confusion_overall(rows):
    y_true = [r["true_label"] for r in rows]
    y_pred = [r["predicted_label"] for r in rows]

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_cm(ax, y_true, y_pred, "Overall Confusion Matrix (%, n=480)")
    fig.tight_layout()
    out = FIGURES / "confusion_overall.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 2: Confusion matrix per prompt ────────────────────────────

def fig_confusion_by_prompt(rows):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    for ax, prompt in zip(axes, PROMPTS):
        subset = [r for r in rows if r["prompt_type"] == prompt]
        y_true = [r["true_label"] for r in subset]
        y_pred = [r["predicted_label"] for r in subset]
        acc = accuracy_score(y_true, y_pred) * 100
        plot_cm(ax, y_true, y_pred, f"{prompt}  (acc={acc:.1f}%)")
    fig.suptitle("Confusion Matrices by Prompt Type", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = FIGURES / "confusion_by_prompt.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 3: Accuracy by prompt × format ────────────────────────────

def fig_accuracy_by_prompt(rows):
    x = np.arange(len(PROMPTS))
    width = 0.35
    accs = {}
    for fmt in FORMATS:
        accs[fmt] = []
        for prompt in PROMPTS:
            subset = [r for r in rows if r["prompt_type"] == prompt and r["output_format"] == fmt]
            y_true = [r["true_label"] for r in subset]
            y_pred = [r["predicted_label"] for r in subset]
            accs[fmt].append(accuracy_score(y_true, y_pred) * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, accs["label_only"], width, label="label_only", color="#5B8DB8")
    bars2 = ax.bar(x + width/2, accs["reasoning"],  width, label="reasoning",  color="#E8864C")

    # Chance line
    ax.axhline(33.3, color="gray", linestyle="--", linewidth=1, label="chance (33%)")

    ax.set_xticks(x)
    ax.set_xticklabels(PROMPTS)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 75)
    ax.set_title("Accuracy by Prompt Type and Output Format", fontweight="bold")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out = FIGURES / "accuracy_by_prompt.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 4: F1 by class × prompt ───────────────────────────────────

def fig_f1_by_class(rows):
    x = np.arange(len(PROMPTS))
    width = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(9, 5))
    for cls, offset, color in zip(CLASSES, offsets, PALETTE.values()):
        f1s = []
        for prompt in PROMPTS:
            subset = [r for r in rows if r["prompt_type"] == prompt]
            y_true = [r["true_label"] for r in subset]
            y_pred = [r["predicted_label"] for r in subset]
            report = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
            f1s.append(report[cls]["f1-score"] * 100)
        bars = ax.bar(x + offset, f1s, width, label=cls, color=color)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(PROMPTS)
    ax.set_ylabel("F1 Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("F1 Score by Class and Prompt Type", fontweight="bold")
    ax.legend(title="Class")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))

    fig.tight_layout()
    out = FIGURES / "f1_by_class.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 5: Confidence analysis ────────────────────────────────────

def fig_confidence(rows):
    reasoning_rows = [
        r for r in rows
        if r["output_format"] == "reasoning" and r["confidence"]
    ]
    if not reasoning_rows:
        print("No confidence data — skipping confidence figure")
        return

    correct   = [int(r["confidence"]) for r in reasoning_rows if r["predicted_label"] == r["true_label"]]
    incorrect = [int(r["confidence"]) for r in reasoning_rows if r["predicted_label"] != r["true_label"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Boxplot
    ax = axes[0]
    ax.boxplot([correct, incorrect], tick_labels=["Correct", "Incorrect"], patch_artist=True,
               boxprops=dict(facecolor="#5B8DB8", alpha=0.7),
               medianprops=dict(color="black", linewidth=2))
    ax.set_ylabel("Confidence (0–100)")
    ax.set_title("Confidence: Correct vs Incorrect", fontweight="bold")
    ax.set_ylim(0, 105)

    # Histogram overlay
    ax = axes[1]
    bins = range(0, 105, 5)
    ax.hist(correct,   bins=bins, alpha=0.6, label=f"Correct (n={len(correct)})",   color="#5B8DB8")
    ax.hist(incorrect, bins=bins, alpha=0.6, label=f"Incorrect (n={len(incorrect)})", color="#E8864C")
    ax.set_xlabel("Confidence (0–100)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution", fontweight="bold")
    ax.legend()

    fig.suptitle("Confidence Calibration", fontweight="bold")
    fig.tight_layout()
    out = FIGURES / "confidence_analysis.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")

    print(f"  Correct   — mean: {np.mean(correct):.1f}, median: {np.median(correct):.0f}")
    print(f"  Incorrect — mean: {np.mean(incorrect):.1f}, median: {np.median(incorrect):.0f}")


# ── Metrics: summary CSV + classification report ──────────────────────

def save_metrics(rows):
    y_true = [r["true_label"] for r in rows]
    y_pred = [r["predicted_label"] for r in rows]

    # Classification report
    report_str = classification_report(y_true, y_pred, labels=CLASSES, zero_division=0)
    txt_path = METRICS / "classification.txt"
    txt_path.write_text(report_str)
    print(f"\nClassification report:\n{report_str}")

    # Summary CSV
    summary_rows = []

    # Overall
    summary_rows.append({
        "group": "overall", "name": "all",
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0),
    })

    # By prompt
    for prompt in PROMPTS:
        s = [r for r in rows if r["prompt_type"] == prompt]
        yt, yp = [r["true_label"] for r in s], [r["predicted_label"] for r in s]
        summary_rows.append({
            "group": "prompt", "name": prompt,
            "accuracy": accuracy_score(yt, yp),
            "macro_f1": f1_score(yt, yp, average="macro", labels=CLASSES, zero_division=0),
        })

    # By format
    for fmt in FORMATS:
        s = [r for r in rows if r["output_format"] == fmt]
        yt, yp = [r["true_label"] for r in s], [r["predicted_label"] for r in s]
        summary_rows.append({
            "group": "format", "name": fmt,
            "accuracy": accuracy_score(yt, yp),
            "macro_f1": f1_score(yt, yp, average="macro", labels=CLASSES, zero_division=0),
        })

    # By class (recall = per-class accuracy)
    report_dict = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    for cls in CLASSES:
        summary_rows.append({
            "group": "class", "name": cls,
            "accuracy": report_dict[cls]["recall"],
            "macro_f1": report_dict[cls]["f1-score"],
        })

    csv_path = METRICS / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "name", "accuracy", "macro_f1"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    rows = load()
    if not rows:
        print("No rows to evaluate.")
        sys.exit(1)

    print("\nGenerating figures...")
    fig_confusion_overall(rows)
    fig_confusion_by_prompt(rows)
    fig_accuracy_by_prompt(rows)
    fig_f1_by_class(rows)
    fig_confidence(rows)

    print("\nSaving metrics...")
    save_metrics(rows)

    print("\nDone. All outputs in results/")


if __name__ == "__main__":
    main()
