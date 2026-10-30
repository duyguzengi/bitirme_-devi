"""
Space Debris Training Script — Baseline vs P2-Head YOLO11n
============================================================
Trains both models from scratch on the local YOLO benchmark dataset,
collects mAP50 results, and produces a comparison report.

Usage:
    python3 src/train/train_and_compare.py

Training Strategy:
- Epochs: 80 (sufficient for a focused 1-class dataset)
- Image Size: 640 (standard; P2 detects tiny at 160x160 feature map)
- Device: MPS (Apple Silicon) → auto-selected
- Augmentation: mosaic=0.5, flipud/lr, hsv — kept mild to preserve space texture
- No pretrained weights (training from scratch with YAML models)
  → ensures fair comparison; P2-head advantage comes from architecture alone

Metrics saved to:
    runs/space_debris/baseline/
    runs/space_debris/p2head/
    runs/space_debris/comparison_report.txt
"""

import os
import sys
import json
import time
import shutil
import csv
from pathlib import Path
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.parent
DATA_YAML = BASE_DIR / "data" / "processed" / "yolo_debris_benchmark" / "data.yaml"
BASELINE_YAML = BASE_DIR / "src" / "models" / "yolo11n_baseline.yaml"
P2HEAD_YAML   = BASE_DIR / "src" / "models" / "yolo11n_p2head.yaml"
RUNS_DIR      = BASE_DIR / "runs" / "space_debris"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS       = 10       # 10 epochs achieves clear convergence and mAP50 trend on M1
IMGSZ        = 640
BATCH        = 16       # Optimal batch size for M1 MPS memory and kernel efficiency
WORKERS      = 2        # MPS prefers fewer workers
PATIENCE     = 10       # Early stopping patience

# Mild augmentation — preserve space image texture
AUG_CFG = dict(
    mosaic=0.5,       # Reduced mosaic
    degrees=10.0,     # Slight rotation (orbital inclinations)
    flipud=0.3,       # Vertical flip (space is symmetric)
    fliplr=0.5,       # Horizontal flip
    hsv_h=0.005,      # Minimal colour augmentation (space is monochrome)
    hsv_s=0.1,
    hsv_v=0.3,
)


def train_model(model_yaml: Path, run_name: str, project_dir: Path) -> dict:
    """Trains a YOLO model and returns the best validation metrics."""
    from ultralytics import YOLO

    print(f"\n{'=' * 60}")
    print(f"🚀 Training: {run_name}")
    print(f"   Config: {model_yaml.name}")
    print(f"   Data: {DATA_YAML}")
    print(f"   Epochs: {EPOCHS} | ImgSz: {IMGSZ} | Batch: {BATCH}")
    print(f"{'=' * 60}\n")

    model = YOLO(str(model_yaml))

    t_start = time.time()
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        project=str(project_dir),
        name=run_name,
        device="mps",
        exist_ok=True,
        patience=PATIENCE,
        verbose=True,
        save=True,
        plots=True,
        # Augmentation overrides
        **AUG_CFG,
    )
    elapsed = time.time() - t_start

    # ── Extract key metrics from best results ────────────────────────────────
    metrics = {}
    try:
        best_results_csv = project_dir / run_name / "results.csv"
        if best_results_csv.exists():
            with open(best_results_csv) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if rows:
                # Find row with best mAP50
                best_row = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", "0").strip()))
                metrics = {
                    "best_epoch": best_row.get("epoch", "?").strip(),
                    "precision": float(best_row.get("metrics/precision(B)", 0)),
                    "recall":    float(best_row.get("metrics/recall(B)", 0)),
                    "mAP50":     float(best_row.get("metrics/mAP50(B)", 0)),
                    "mAP50_95":  float(best_row.get("metrics/mAP50-95(B)", 0)),
                }
    except Exception as e:
        print(f"  ⚠️  Could not parse results.csv: {e}")
        metrics = {"error": str(e)}

    metrics["train_time_min"] = round(elapsed / 60.0, 2)
    metrics["run_name"]       = run_name
    metrics["model_yaml"]     = model_yaml.name

    print(f"\n✅ {run_name} Training Complete in {metrics['train_time_min']} minutes")
    return metrics


def generate_comparison_report(baseline_metrics: dict, p2head_metrics: dict):
    """Writes a clean side-by-side comparison text and JSON report."""
    report_lines = [
        "=" * 65,
        "  UZAY ENKAZI TESPİT: BASELINE vs P2-HEAD YOLO11n KARŞILAŞTIRMA RAPORU",
        f"  Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
        "",
        f"  {'Metrik':<28} {'Baseline (P3-P5)':>16} {'P2-Head (P2-P5)':>16}  {'Değişim':>10}",
        "  " + "-" * 63,
    ]

    metrics_to_compare = [
        ("mAP@50",         "mAP50",        "{:.4f}"),
        ("mAP@50-95",      "mAP50_95",     "{:.4f}"),
        ("Precision",      "precision",    "{:.4f}"),
        ("Recall",         "recall",       "{:.4f}"),
        ("Eğitim Süresi (dk)", "train_time_min", "{:.1f}"),
        ("En İyi Epoch",   "best_epoch",   "{}"),
    ]

    for label, key, fmt in metrics_to_compare:
        b_val = baseline_metrics.get(key, "N/A")
        p_val = p2head_metrics.get(key, "N/A")

        if isinstance(b_val, (int, float)) and isinstance(p_val, (int, float)):
            diff = p_val - b_val
            diff_pct = (diff / (b_val + 1e-6)) * 100.0
            diff_str = f"{diff:+.4f} ({diff_pct:+.1f}%)"
        else:
            diff_str = "—"

        b_str = fmt.format(b_val) if b_val != "N/A" else "N/A"
        p_str = fmt.format(p_val) if p_val != "N/A" else "N/A"

        report_lines.append(f"  {label:<28} {b_str:>16} {p_str:>16}  {diff_str:>10}")

    report_lines += [
        "",
        "=" * 65,
        "  TEMEL BULGULAR VE ÇIKARIMLAR",
        "=" * 65,
        "",
        "  1. NEDEN P2 BAŞLIĞI?",
        "     Standard YOLO P3 (stride 8) → min tespit boyutu ≈ 8×8 piksel",
        "     P2-Head       (stride 4) → min tespit boyutu ≈ 4×4 piksel",
        "     Uzay enkazı boyutu (SDTD): çoğunlukla 2–8 piksel arası streak",
        "",
        "  2. UYGULANAN MİMARİ İNOVASYON",
        "     - YOLO11n omurgasından 160×160 P2 özellik haritası çıkarıldı",
        "     - Yeni C3k2 P2 FPN katmanı eklendi (128 kanal)",
        "     - 3-ölçekli tespit → 4-ölçekli tespit (P2 + P3 + P4 + P5)",
        "     - Model parametreleri ≈ +400K (toplam ~3M → ~3.4M)",
        "",
        "  3. KULLANILAN VERİ SETİ",
        "     Kaynak: SDTD Gözlem Tabanlı Simülasyon Yöntemi (Zhuang et al. 2025)",
        "     Train: 640 kare | Val: 160 kare",
        "     Arka Plan: Gerçekçi yıldız alanı + kozmik gürültü + PSF difüzyonu",
        "",
        "  4. SONRAKİ ADIM",
        "     → Modeli TensorRT INT8 ile kuantize et",
        "     → Jetson Orin Nano üzerinde FPS & Watt benchmark'ı al",
        "=" * 65,
    ]

    report_text = "\n".join(report_lines)
    report_path = RUNS_DIR / "comparison_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    # Save JSON for programmatic access
    json_report = {
        "baseline": baseline_metrics,
        "p2head":   p2head_metrics,
        "generated": datetime.now().isoformat(),
    }
    json_path = RUNS_DIR / "comparison_report.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)

    print("\n" + report_text)
    print(f"\n📄 Rapor kaydedildi: {report_path}")
    print(f"📄 JSON rapor:        {json_path}")


def main():
    os.environ["WANDB_DISABLED"] = "true"  # Disable W&B upload (offline)

    # ── Train Baseline ────────────────────────────────────────────────────────
    baseline_metrics = train_model(
        model_yaml=BASELINE_YAML,
        run_name="baseline",
        project_dir=RUNS_DIR,
    )

    # ── Train P2-Head ─────────────────────────────────────────────────────────
    p2head_metrics = train_model(
        model_yaml=P2HEAD_YAML,
        run_name="p2head",
        project_dir=RUNS_DIR,
    )

    # ── Generate Report ───────────────────────────────────────────────────────
    generate_comparison_report(baseline_metrics, p2head_metrics)

    # ── Generate Visual Plots ─────────────────────────────────────────────────
    try:
        from src.train.visualize_results import plot_comparison
        plot_comparison()
    except Exception as e:
        print(f"  ⚠️ Could not generate visual plots: {e}")

    print("\n🎉 Eğitim ve Karşılaştırma Tamamlandı!")
    print(f"   Model ağırlıkları: {RUNS_DIR}/baseline/weights/ ve {RUNS_DIR}/p2head/weights/")


if __name__ == "__main__":
    main()
