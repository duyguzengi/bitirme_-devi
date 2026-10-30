"""
Dataset Acquisition and Management Module.

Provides registry, URLs, download helpers, and local synthetic bootstrapping for:
1. SDTD (Space Debris Tracking Dataset - Shanghai AI Lab / ZJU 2025)
2. SPEED+ (Next-Gen Spacecraft Pose Estimation Dataset - Stanford SLAB & ESA)
3. Local Synthetic Debris Benchmark Generator (immediate plug-and-play dataset)
"""

import os
import json
import argparse
from typing import Dict
from .synthetic_debris_generator import SpaceDebrisSimulator
from .label_converter import export_yolo_annotation_file, export_mot_gt_file
import cv2


DATASET_REGISTRY = {
    "SDTD": {
        "title": "Space Debris Tracking Dataset (SDTD)",
        "authors": "Guohang Zhuang, Weixi Song, Wanli Ouyang et al. (2025)",
        "source": "Zwicky Transient Facility (ZTF) and Antarctic Station Observations",
        "sequences": 18040,
        "frames": 62562,
        "targets": 250000,
        "github_reference": "https://github.com/ShanghaiAI-Lab/SDTD (or paper release link)",
        "description": "Astronomical video sequences with dense debris, skylight, and star background noise.",
    },
    "SPEED+": {
        "title": "SPEED+ (Next-Generation Spacecraft Pose Estimation Dataset)",
        "authors": "Stanford Space Rendezvous Laboratory (SLAB) & European Space Agency (ESA)",
        "zenodo_doi": "10.5281/zenodo.5587976",
        "url": "https://zenodo.org/record/5587976",
        "description": "Photorealistic synthetic and hardware-in-the-loop (lightbox/sunlamp) images of the Prisma satellite with harsh space illumination.",
    },
}


def print_dataset_info():
    """Prints registry information for the target datasets."""
    print("=" * 60)
    print("🛰️ SPACE DEBRIS DATASET REGISTRY & DOWNLOAD GUIDE")
    print("=" * 60)
    for name, info in DATASET_REGISTRY.items():
        print(f"\n[{name}] - {info['title']}")
        print(f"  Description: {info['description']}")
        if "url" in info:
            print(f"  URL / DOI:   {info['url']}")
        if "github_reference" in info:
            print(f"  Reference:   {info['github_reference']}")
    print("=" * 60)


def generate_local_benchmark_dataset(
    output_dir: str = "data/processed/yolo_debris_benchmark",
    num_train_sequences: int = 10,
    num_val_sequences: int = 3,
    frames_per_sequence: int = 10,
    img_size: int = 640,
):
    """
    Bootstraps a fully standardized local YOLOv8/v11 & MOT benchmark dataset
    using the SDTD observation-based physics generator.

    Creates standard structure:
    output_dir/
      images/train/
      images/val/
      labels/train/
      labels/val/
      data.yaml
    """
    print(f"\n🚀 Bootstrapping Local Space Debris Benchmark in: {output_dir}")

    for split, count in [("train", num_train_sequences), ("val", num_val_sequences)]:
        img_split_dir = os.path.join(output_dir, "images", split)
        lbl_split_dir = os.path.join(output_dir, "labels", split)
        os.makedirs(img_split_dir, exist_ok=True)
        os.makedirs(lbl_split_dir, exist_ok=True)

        seq_counter = 0
        for seq_i in range(count):
            simulator = SpaceDebrisSimulator(
                width=img_size,
                height=img_size,
                num_stars=random_int(40, 100),
                psf_sigma=1.2,
            )
            frames, annotations = simulator.generate_sequence(
                num_frames=frames_per_sequence,
                num_debris=random_int(1, 3),
            )

            for f_idx, (frame, annos) in enumerate(zip(frames, annotations)):
                img_name = f"seq{seq_i:03d}_frame{f_idx:03d}.jpg"
                txt_name = f"seq{seq_i:03d}_frame{f_idx:03d}.txt"

                img_path = os.path.join(img_split_dir, img_name)
                txt_path = os.path.join(lbl_split_dir, txt_name)

                # Save frame image
                cv2.imwrite(img_path, frame)

                # Save YOLO annotations
                export_yolo_annotation_file(txt_path, annos, img_size, img_size)

            seq_counter += 1

        print(f"  ✅ Completed split '{split}': {count} sequences ({count * frames_per_sequence} frames)")

    # Create data.yaml for YOLO training
    yaml_content = f"""# Space Debris YOLO Dataset Configuration
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

# Classes
names:
  0: space_debris
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"  📄 Generated YOLO data.yaml: {yaml_path}")
    print(f"🎉 Dataset generation complete!\n")


def random_int(a, b):
    import random
    return random.randint(a, b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Downloader & Bootstrap Tool")
    parser.add_argument("--info", action="store_true", help="Print dataset sources and information")
    parser.add_argument("--bootstrap", action="store_true", help="Generate local YOLO benchmark dataset")
    parser.add_argument("--output", type=str, default="data/processed/yolo_debris_benchmark", help="Output directory")

    args = parser.parse_args()
    if args.info:
        print_dataset_info()
    if args.bootstrap:
        generate_local_benchmark_dataset(output_dir=args.output)
