"""
Post-Training Comparison Visualizer
=====================================
Loads runs/space_debris/baseline/results.csv and runs/space_debris/p2head/results.csv,
produces side-by-side training curves (mAP50, Loss) and a final metrics bar chart.
Run AFTER training completes.

Usage:
    python3 src/train/visualize_results.py
"""

import os
import sys
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless rendering for saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR = Path(__file__).parent.parent.parent
RUNS_DIR = BASE_DIR / "runs" / "space_debris"


def load_csv(run_name: str) -> list[dict]:
    csv_path = RUNS_DIR / run_name / "results.csv"
    if not csv_path.exists():
        print(f"  ⚠️  {csv_path} not found — training may not be complete.")
        return []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows


def parse_col(rows: list[dict], key: str) -> list[float]:
    out = []
    for r in rows:
        for k in r:
            if k.strip() == key.strip():
                try:
                    out.append(float(r[k].strip()))
                except ValueError:
                    out.append(0.0)
                break
    return out


def plot_comparison():
    print("📊 Loading training results...")
    baseline_rows = load_csv("baseline")
    p2head_rows   = load_csv("p2head")

    if not baseline_rows and not p2head_rows:
        print("❌ No results found yet. Run after training completes.")
        return

    epochs_b = list(range(1, len(baseline_rows) + 1))
    epochs_p = list(range(1, len(p2head_rows)   + 1))

    # ── Extract curves ────────────────────────────────────────────────────────
    b_map50    = parse_col(baseline_rows, "metrics/mAP50(B)")
    b_map5095  = parse_col(baseline_rows, "metrics/mAP50-95(B)")
    b_prec     = parse_col(baseline_rows, "metrics/precision(B)")
    b_rec      = parse_col(baseline_rows, "metrics/recall(B)")
    b_box_loss = parse_col(baseline_rows, "train/box_loss")
    b_cls_loss = parse_col(baseline_rows, "train/cls_loss")

    p_map50    = parse_col(p2head_rows, "metrics/mAP50(B)")
    p_map5095  = parse_col(p2head_rows, "metrics/mAP50-95(B)")
    p_prec     = parse_col(p2head_rows, "metrics/precision(B)")
    p_rec      = parse_col(p2head_rows, "metrics/recall(B)")
    p_box_loss = parse_col(p2head_rows, "train/box_loss")
    p_cls_loss = parse_col(p2head_rows, "train/cls_loss")

    BLUE   = "#2196F3"
    ORANGE = "#FF5722"

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "🛰️ UZAY ENKAZI TESPİT: Baseline YOLO11n  vs  P2-Head YOLO11n",
        fontsize=15, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1. mAP50 curves ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if b_map50: ax1.plot(epochs_b, b_map50, color=BLUE,   lw=2, label="Baseline (P3-P5)")
    if p_map50: ax1.plot(epochs_p, p_map50, color=ORANGE, lw=2, label="P2-Head (P2-P5)")
    ax1.set_title("Val mAP@50 Eğitim Eğrisi", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("mAP@50")
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_ylim(0, 1.05)

    # ── 2. mAP50-95 curves ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if b_map5095: ax2.plot(epochs_b, b_map5095, color=BLUE,   lw=2, label="Baseline")
    if p_map5095: ax2.plot(epochs_p, p_map5095, color=ORANGE, lw=2, label="P2-Head")
    ax2.set_title("Val mAP@50-95 Eğitim Eğrisi", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("mAP@50-95")
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 1.05)

    # ── 3. Train Box Loss ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if b_box_loss: ax3.plot(epochs_b, b_box_loss, color=BLUE,   lw=2, label="Baseline")
    if p_box_loss: ax3.plot(epochs_p, p_box_loss, color=ORANGE, lw=2, label="P2-Head")
    ax3.set_title("Train Box Loss", fontweight="bold")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Loss")
    ax3.legend(); ax3.grid(True, alpha=0.3)

    # ── 4. Precision curves ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    if b_prec: ax4.plot(epochs_b, b_prec, color=BLUE,   lw=2, label="Baseline Precision")
    if p_prec: ax4.plot(epochs_p, p_prec, color=ORANGE, lw=2, label="P2-Head Precision")
    if b_rec:  ax4.plot(epochs_b, b_rec,  color=BLUE,   lw=2, ls="--", label="Baseline Recall")
    if p_rec:  ax4.plot(epochs_p, p_rec,  color=ORANGE, lw=2, ls="--", label="P2-Head Recall")
    ax4.set_title("Precision & Recall", fontweight="bold")
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("Score")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3); ax4.set_ylim(0, 1.05)

    # ── 5. Final metrics bar chart ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1:])
    json_path = RUNS_DIR / "comparison_report.json"
    if json_path.exists():
        with open(json_path) as f:
            report = json.load(f)

        bm = report.get("baseline", {})
        pm = report.get("p2head", {})

        metric_names = ["mAP@50", "mAP@50-95", "Precision", "Recall"]
        metric_keys  = ["mAP50",  "mAP50_95",  "precision", "recall"]
        b_vals = [bm.get(k, 0) for k in metric_keys]
        p_vals = [pm.get(k, 0) for k in metric_keys]

        x = np.arange(len(metric_names))
        width = 0.35
        bars1 = ax5.bar(x - width/2, b_vals, width, label="Baseline (P3-P5)", color=BLUE,   alpha=0.85)
        bars2 = ax5.bar(x + width/2, p_vals, width, label="P2-Head (P2-P5)",  color=ORANGE, alpha=0.85)

        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                ax5.text(bar.get_x() + bar.get_width()/2., h + 0.01, f"{h:.3f}",
                         ha="center", va="bottom", fontsize=9, color=BLUE, fontweight="bold")
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax5.text(bar.get_x() + bar.get_width()/2., h + 0.01, f"{h:.3f}",
                         ha="center", va="bottom", fontsize=9, color=ORANGE, fontweight="bold")

        ax5.set_title("Final Metrik Karşılaştırması (En İyi Epoch)", fontweight="bold")
        ax5.set_xticks(x)
        ax5.set_xticklabels(metric_names)
        ax5.set_ylabel("Metrik Değeri")
        ax5.set_ylim(0, 1.15)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis="y")

        # Delta annotations
        for i, (bv, pv) in enumerate(zip(b_vals, p_vals)):
            delta = pv - bv
            color = "green" if delta >= 0 else "red"
            sign  = "+" if delta >= 0 else ""
            ax5.text(i, 1.10, f"Δ {sign}{delta:.3f}",
                     ha="center", fontsize=9, color=color, fontweight="bold")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = RUNS_DIR / "comparison_chart.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Karşılaştırma grafiği kaydedildi: {out_path}")

    # ── P/R Curve comparison ──────────────────────────────────────────────────
    # Check for Ultralytics-generated PR curve images
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle("Precision-Recall Eğrisi Karşılaştırması", fontsize=14, fontweight="bold")
    for ax, run_name, color, label in [
        (axes[0], "baseline", BLUE,   "Baseline (P3-P5)"),
        (axes[1], "p2head",   ORANGE, "P2-Head (P2-P5)"),
    ]:
        pr_img_path = RUNS_DIR / run_name / "PR_curve.png"
        if pr_img_path.exists():
            import matplotlib.image as mpimg
            img = mpimg.imread(str(pr_img_path))
            ax.imshow(img)
            ax.set_title(f"{label}", fontweight="bold", color=color)
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, f"PR Curve\n({run_name})\nEğitim tamamlandıktan\nsonra mevcut olacak",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title(label, color=color, fontweight="bold")

    pr_out = RUNS_DIR / "pr_curve_comparison.png"
    fig2.savefig(pr_out, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"✅ PR eğrisi grafiği kaydedildi: {pr_out}")


if __name__ == "__main__":
    plot_comparison()
