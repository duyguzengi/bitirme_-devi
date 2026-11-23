"""
Thorough P2-Head YOLO11n Training Pipeline (Transfer Learning + 25 Epochs)
========================================================================
Key Innovations for Space Debris Detection:
1. Transfer Learning from official YOLO11n pretrained weights (`yolo11n.pt`).
2. 4-scale multi-level detection (P2 @ stride 4: 160x160, P3 @ 8, P4 @ 16, P5 @ 32).
3. Optimized for Apple Silicon M1 Unified Memory (batch=4, accumulate=2, workers=0).
4. AdamW optimizer with Cosine Annealing learning rate schedule.
5. Debris-specific augmentations (scale, flip, perspective, HSV jitter for solar reflections).
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
import torch

BASE_DIR = Path(__file__).parent.parent.parent
DATA_YAML = BASE_DIR / "data" / "processed" / "yolo_debris_benchmark" / "data.yaml"
P2HEAD_YAML = BASE_DIR / "src" / "models" / "yolo11n_p2head.yaml"
PRETRAINED_WEIGHTS = BASE_DIR / "yolo11n.pt"
RUNS_DIR = BASE_DIR / "runs" / "space_debris"

EPOCHS = 20
IMGSZ = 640
BATCH = 4           # Safe memory footprint on 8GB M1 (~2.3 GB VRAM)
WORKERS = 0         # Main thread dataloading avoids macOS multiprocessing fork issues

AUG_CFG = dict(
    mosaic=0.7,
    degrees=15.0,
    flipud=0.5,
    fliplr=0.5,
    scale=0.3,
    perspective=0.0005,
    hsv_h=0.015,
    hsv_s=0.2,
    hsv_v=0.4,
    mixup=0.1,
    copy_paste=0.1,
)


def run_thorough_training():
    from ultralytics import YOLO

    os.environ["WANDB_DISABLED"] = "true"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("=" * 70)
    print("🛰️  HIGH-ACCURACY SPACE DEBRIS TRAINING (P2-HEAD YOLO11n)")
    print(f"   Architecture:     {P2HEAD_YAML.name} (P2=stride 4 / 160x160)")
    print(f"   Pretrained Base:  {PRETRAINED_WEIGHTS.name} (Transfer Learning)")
    print(f"   Dataset:          {DATA_YAML.name}")
    print(f"   Epochs:           {EPOCHS}")
    print(f"   Batch Size:       {BATCH} (M1 Unified Memory Optimized)")
    print(f"   Image Size:       {IMGSZ}x{IMGSZ}")
    print(f"   Hardware Device:  {device.upper()} (Apple Silicon M1)")
    print(f"   Optimizer:        AdamW (lr0=0.002, lrf=0.01, decay=0.0005)")
    print("=" * 70)

    # Initialize model with P2-Head architecture and transfer pretrained weights
    print("\n📦 Loading architecture and transferring pretrained weights...")
    model = YOLO(str(P2HEAD_YAML)).load(str(PRETRAINED_WEIGHTS))

    t_start = time.time()

    # Train model
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        project=str(RUNS_DIR),
        name="p2head_thorough",
        device=device,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=2,
        cos_lr=True,
        exist_ok=True,
        verbose=True,
        save=True,
        plots=True,
        **AUG_CFG,
    )

    elapsed_sec = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"✅ Training completed in {elapsed_sec / 60.0:.2f} minutes ({elapsed_sec:.1f}s)")
    print("=" * 70)

    # Validate and print metrics
    best_weights = RUNS_DIR / "p2head_thorough" / "weights" / "best.pt"
    print(f"\n📊 Validating final model from: {best_weights}...")
    val_model = YOLO(str(best_weights))
    val_results = val_model.val(data=str(DATA_YAML), device=device, split="val", imgsz=IMGSZ)

    mAP50 = float(val_results.box.map50)
    mAP50_95 = float(val_results.box.map)
    precision = float(val_results.box.mp)
    recall = float(val_results.box.mr)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    summary = {
        "model": "YOLO11n-P2 (Thorough Transfer Learning)",
        "epochs": EPOCHS,
        "pretrained_backbone": "yolo11n.pt",
        "device": device,
        "mAP50": round(mAP50, 4),
        "mAP50_95": round(mAP50_95, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "training_duration_min": round(elapsed_sec / 60.0, 2),
        "best_weights": str(best_weights),
    }

    report_path = RUNS_DIR / "p2head_thorough" / "evaluation_summary.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n🏆 FINAL EVALUATION METRICS:")
    print(f"   Precision:    {precision * 100:.2f}%")
    print(f"   Recall:       {recall * 100:.2f}%")
    print(f"   mAP@0.50:     {mAP50 * 100:.2f}%")
    print(f"   mAP@0.50:0.95: {mAP50_95 * 100:.2f}%")
    print(f"   F1 Score:     {f1 * 100:.2f}%")
    print(f"   Summary JSON: {report_path}")

    # Also update master best.pt for the video pipeline if it beats previous
    master_best = RUNS_DIR / "p2head" / "weights" / "best.pt"
    shutil.copy2(best_weights, master_best)
    print(f"   Updated master weights: {master_best}")

    return summary


if __name__ == "__main__":
    run_thorough_training()
