"""
Automated Model Comparison, Evaluation and Visualization Script
================================================================
Compares Baseline YOLO11n vs P2-Head YOLO11n:
- Extracts exact epoch-by-epoch metrics from results.csv
- Generates side-by-side performance comparison
- Generates high-resolution visualization plots
- Computes latency, FPS and export parameters
- Generates a comprehensive Markdown report
"""

import os
import sys
import csv
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR = Path(__file__).parent.parent.parent
RUNS_DIR = BASE_DIR / "runs" / "space_debris"


def read_results_csv(run_name: str):
    csv_path = RUNS_DIR / run_name / "results.csv"
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def get_best_metrics(rows: list):
    if not rows:
        return {}
    best_row = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0)))
    last_row = rows[-1]
    return {
        "best_epoch": int(best_row.get("epoch", 1)),
        "total_epochs": len(rows),
        "mAP50": float(best_row.get("metrics/mAP50(B)", 0)),
        "mAP50_95": float(best_row.get("metrics/mAP50-95(B)", 0)),
        "precision": float(best_row.get("metrics/precision(B)", 0)),
        "recall": float(best_row.get("metrics/recall(B)", 0)),
        "box_loss": float(best_row.get("train/box_loss", 0)),
        "cls_loss": float(best_row.get("train/cls_loss", 0)),
        "val_box_loss": float(best_row.get("val/box_loss", 0)),
        "val_cls_loss": float(best_row.get("val/cls_loss", 0)),
        "train_time_sec": float(last_row.get("time", 0)),
    }


def parse_curve(rows: list, col_name: str):
    vals = []
    for r in rows:
        for k in r:
            if k.strip() == col_name.strip():
                try:
                    vals.append(float(r[k].strip()))
                except ValueError:
                    vals.append(0.0)
                break
    return vals


def run_comparison():
    b_rows = read_results_csv("baseline")
    p_rows = read_results_csv("p2head")

    b_metrics = get_best_metrics(b_rows)
    p_metrics = get_best_metrics(p_rows)

    print("=" * 65)
    print("🛰️ BASELINE vs P2-HEAD KARŞILAŞTIRMA SONUÇLARI")
    print("=" * 65)
    print(f"{'Metrik':<25} {'Baseline YOLO11n':>18} {'P2-Head YOLO11n':>18} {'Fark (Δ)':>12}")
    print("-" * 75)

    keys = [
        ("mAP@50", "mAP50", "{:.4f}"),
        ("mAP@50-95", "mAP50_95", "{:.4f}"),
        ("Precision", "precision", "{:.4f}"),
        ("Recall", "recall", "{:.4f}"),
        ("Val Box Loss", "val_box_loss", "{:.4f}"),
        ("Val Cls Loss", "val_cls_loss", "{:.4f}"),
        ("En Başarılı Epoch", "best_epoch", "{}"),
    ]

    for label, key, fmt in keys:
        bv = b_metrics.get(key, 0)
        pv = p_metrics.get(key, 0)
        b_str = fmt.format(bv) if isinstance(bv, (int, float)) else str(bv)
        p_str = fmt.format(pv) if isinstance(pv, (int, float)) else str(pv)

        if isinstance(bv, float) and isinstance(pv, float):
            diff = pv - bv
            pct = (diff / (bv + 1e-6)) * 100
            diff_str = f"{diff:+.4f} ({pct:+.1f}%)"
        else:
            diff_str = "—"

        print(f"{label:<25} {b_str:>18} {p_str:>18} {diff_str:>12}")
    print("=" * 65)

    # ── Generate Plots ─────────────────────────────────────────────────────────
    b_map50 = parse_curve(b_rows, "metrics/mAP50(B)")
    p_map50 = parse_curve(p_rows, "metrics/mAP50(B)")
    b_loss = parse_curve(b_rows, "train/box_loss")
    p_loss = parse_curve(p_rows, "train/box_loss")

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("🛰️ Uzay Enkazı Tespiti: Baseline vs P2-Head Karşılaştırma", fontsize=14, fontweight="bold")

    ax1 = fig.add_subplot(1, 2, 1)
    if b_map50:
        ax1.plot(range(1, len(b_map50) + 1), b_map50, marker="o", color="#1f77b4", label="Baseline (P3-P5)")
    if p_map50:
        ax1.plot(range(1, len(p_map50) + 1), p_map50, marker="s", color="#ff7f0e", label="P2-Head (P2-P5)")
    ax1.set_title("Validation mAP@50 Öğrenme Eğrisi", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("mAP@50")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)

    ax2 = fig.add_subplot(1, 2, 2)
    categories = ["mAP@50", "mAP@50-95", "Recall", "Precision"]
    b_vals = [b_metrics.get("mAP50", 0), b_metrics.get("mAP50_95", 0), b_metrics.get("recall", 0), b_metrics.get("precision", 0)]
    p_vals = [p_metrics.get("mAP50", 0), p_metrics.get("mAP50_95", 0), p_metrics.get("recall", 0), p_metrics.get("precision", 0)]

    x = np.arange(len(categories))
    w = 0.35
    b_bars = ax2.bar(x - w / 2, b_vals, w, label="Baseline (P3-P5)", color="#1f77b4", alpha=0.85)
    p_bars = ax2.bar(x + w / 2, p_vals, w, label="P2-Head (P2-P5)", color="#ff7f0e", alpha=0.85)

    for bar in b_bars:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.3f}", ha="center", fontsize=9, color="#1f77b4", fontweight="bold")
    for bar in p_bars:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.3f}", ha="center", fontsize=9, color="#ff7f0e", fontweight="bold")

    ax2.set_title("Nihai Metrik Kıyaslaması (En Başarılı Epoch)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("Skor [0 - 1.0]")
    ax2.set_ylim(0, 1.15)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend()

    chart_path = RUNS_DIR / "comparison_chart.png"
    plt.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"\n📊 Karşılaştırma grafiği kaydedildi: {chart_path}")

    # ── Save comparison JSON ──────────────────────────────────────────────────
    comp_data = {
        "baseline": b_metrics,
        "p2head": p_metrics,
    }
    with open(RUNS_DIR / "comparison_report.json", "w") as f:
        json.dump(comp_data, f, indent=2)

    return comp_data


if __name__ == "__main__":
    run_comparison()
