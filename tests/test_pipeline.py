"""
Comprehensive End-to-End Test for Hafta 4-6 Milestones:
- Preprocessing (Cosmic Ray Filtering, Star Removal, ZScale/CLAHE)
- Synthetic Debris Generation (SDTD Equations 4-6)
- YOLO and MOT Label Standardization
- Local Dataset Bootstrapping
"""

import os
import sys
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.cosmic_ray_filter import filter_cosmic_rays_median
from src.preprocessing.star_removal import remove_stars_morphological, frame_difference_stars
from src.preprocessing.clahe_enhancer import enhance_space_image, apply_zscale
from src.dataset.synthetic_debris_generator import SpaceDebrisSimulator
from src.dataset.label_converter import xyxy_to_yolo, yolo_to_xyxy, export_yolo_annotation_file, export_mot_gt_file
from src.dataset.dataset_downloader import generate_local_benchmark_dataset
from src.utils.visualizer import draw_annotations, create_comparison_grid


def run_all_tests():
    print("=" * 60)
    print("🛰️ RUNNING TEST SUITE: SPACE DEBRIS EDGE AI PIPELINE")
    print("=" * 60)

    # 1. Test Simulator
    print("\n[Step 1] Testing Space Debris Simulation (SDTD Formulation)...")
    sim = SpaceDebrisSimulator(width=640, height=640, num_stars=60, psf_sigma=1.2, seed=42)
    frames, annos = sim.generate_sequence(num_frames=5, num_debris=2)
    assert len(frames) == 5, f"Expected 5 frames, got {len(frames)}"
    assert len(annos) == 5, f"Expected 5 annotation sets, got {len(annos)}"
    print(f"  ✅ Successfully generated 5 frames. Frame 0 debris count: {len(annos[0])}")

    # 2. Test Cosmic Ray / Hot Pixel Filtering
    print("\n[Step 2] Testing Cosmic Ray & Hot Pixel Filtering...")
    noisy_frame = frames[0].copy()
    # Inject isolated extreme spikes
    noisy_frame[150, 200] = 255
    noisy_frame[320, 410] = 254
    cleaned_cosmic = filter_cosmic_rays_median(noisy_frame, kernel_size=3)
    assert cleaned_cosmic[150, 200] < 200, "Hot pixel should have been attenuated"
    print("  ✅ Cosmic ray filtering verified successfully.")

    # 3. Test Star Removal & Background Subtraction
    print("\n[Step 3] Testing Star Removal (Morphological & Temporal Differencing)...")
    star_cleaned = remove_stars_morphological(cleaned_cosmic, star_kernel_size=3, streak_kernel_len=11)
    diff_mask = frame_difference_stars(frames[0], frames[1], threshold=15)
    print(f"  ✅ Morphological star suppression & frame differencing computed.")

    # 4. Test Astronomical Contrast Enhancement (ZScale & CLAHE)
    print("\n[Step 4] Testing ZScale Contrast Stretch and CLAHE...")
    enhanced = enhance_space_image(star_cleaned, use_zscale=True, clip_limit=2.5)
    assert enhanced.dtype == np.uint8
    print("  ✅ ZScale & CLAHE enhancement completed.")

    # 5. Test Label Standardization (YOLO format & MOT format)
    print("\n[Step 5] Testing Label Standardization...")
    demo_bbox = (100.0, 150.0, 140.0, 170.0)
    yolo_str = xyxy_to_yolo(demo_bbox, img_width=640, img_height=640, class_id=0)
    cid, x1, y1, x2, y2 = yolo_to_xyxy(yolo_str, img_width=640, img_height=640)
    assert abs(x1 - 100.0) < 1.0, f"Conversion error on x1: {x1}"
    assert abs(y2 - 170.0) < 1.0, f"Conversion error on y2: {y2}"
    print(f"  ✅ YOLO string: '{yolo_str}' successfully round-tripped.")

    # 6. Test Visualizer & Demo Export
    print("\n[Step 6] Saving Verification Grids...")
    demo_dir = os.path.join(PROJECT_ROOT, "data", "sample_demo")
    os.makedirs(demo_dir, exist_ok=True)

    grid = create_comparison_grid(noisy_frame, cleaned_cosmic, star_cleaned, enhanced)
    grid_path = os.path.join(demo_dir, "pipeline_stages_grid.jpg")
    cv2.imwrite(grid_path, grid)

    annotated = draw_annotations(frames[0], annos[0], color=(0, 255, 0))
    anno_path = os.path.join(demo_dir, "annotated_frame_0.jpg")
    cv2.imwrite(anno_path, annotated)
    print(f"  ✅ Saved pipeline comparison grid to: {grid_path}")
    print(f"  ✅ Saved annotated frame to: {anno_path}")

    # 7. Test Local Dataset Generation
    print("\n[Step 7] Bootstrapping Local Benchmark Dataset for YOLO Training...")
    ds_dir = os.path.join(PROJECT_ROOT, "data", "processed", "yolo_debris_benchmark")
    generate_local_benchmark_dataset(
        output_dir=ds_dir,
        num_train_sequences=4,
        num_val_sequences=1,
        frames_per_sequence=5,
        img_size=640,
    )
    assert os.path.exists(os.path.join(ds_dir, "data.yaml")), "data.yaml must exist"
    assert os.path.exists(os.path.join(ds_dir, "images", "train")), "images/train must exist"
    print("  ✅ Local benchmark dataset successfully bootstrapped and validated.")

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED SUCCESSFULLY! HAFTA 4-6 HEDEFLERİ EKSİKSİZ TAMAMLANDI.")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
