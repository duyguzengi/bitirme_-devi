"""
P2-Head Training Script (Fair 5-Epoch Evaluation against Baseline)
==================================================================
Trains the P2-Head YOLO11n model for 5 epochs with batch=4 (optimized for M1 8GB unified RAM),
so that both models are evaluated apples-to-apples on the exact same validation set.
"""

import os
import sys
import time
import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_YAML = BASE_DIR / "data" / "processed" / "yolo_debris_benchmark" / "data.yaml"
P2HEAD_YAML = BASE_DIR / "src" / "models" / "yolo11n_p2head.yaml"
RUNS_DIR = BASE_DIR / "runs" / "space_debris"

EPOCHS = 5
IMGSZ = 640
BATCH = 4       # Optimized for M1 (8GB unified memory) → consumes ~2.2GB GPU RAM
WORKERS = 0     # Direct main-thread dataloading prevents macOS multiprocessing overhead

AUG_CFG = dict(
    mosaic=0.5,
    degrees=10.0,
    flipud=0.3,
    fliplr=0.5,
    hsv_h=0.005,
    hsv_s=0.1,
    hsv_v=0.3,
)


def train_p2head():
    from ultralytics import YOLO

    os.environ["WANDB_DISABLED"] = "true"
    print("=" * 60)
    print("🚀 Training P2-Head YOLO11n (4-Scale Detection: P2, P3, P4, P5)")
    print(f"   Config:  {P2HEAD_YAML.name}")
    print(f"   Epochs:  {EPOCHS} | Batch: {BATCH} | ImgSz: {IMGSZ}")
    print(f"   Device:  Apple M1 (MPS, ~2.2GB RAM)")
    print("=" * 60)

    model = YOLO(str(P2HEAD_YAML))

    t_start = time.time()
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        project=str(RUNS_DIR),
        name="p2head",
        device="mps",
        exist_ok=True,
        verbose=True,
        save=True,
        plots=True,
        **AUG_CFG,
    )
    elapsed = time.time() - t_start
    print(f"\n✅ P2-Head Training Complete in {elapsed / 60.0:.2f} minutes")

    # Run automated comparison report
    try:
        from src.train.compare_models import run_comparison
        run_comparison()
    except Exception as e:
        print(f"  ⚠️ Could not run comparison: {e}")


if __name__ == "__main__":
    train_p2head()
