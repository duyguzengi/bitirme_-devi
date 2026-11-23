"""
Real Space Video Testing & Verification Script
==============================================
Runs the unified P2-Head ByteTrack + Kalman + Collision Risk (TTC) pipeline
on real NASA space footage:
1. NASA Dellingr CubeSat Deployment in Earth Orbit (data/real_cubesat_deploy_nasa.mp4)
2. NASA OCSD CubeSat Orbital Rendezvous (data/real_space_video_1.mp4)
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.tracking.video_pipeline import SpaceDebrisVideoPipeline

WEIGHTS_THOROUGH = BASE_DIR / "runs" / "space_debris" / "p2head_thorough" / "weights" / "best.pt"
WEIGHTS_FALLBACK = BASE_DIR / "runs" / "space_debris" / "p2head" / "weights" / "best.pt"
OUTPUT_DIR = BASE_DIR / "output"


def run_real_video_tests(video_choice: int = 1, max_frames: int = 250):
    weights = str(WEIGHTS_THOROUGH if WEIGHTS_THOROUGH.exists() else WEIGHTS_FALLBACK)
    print(f"🎬 Initializing Real Space Video Test using weights: {weights}")

    pipeline = SpaceDebrisVideoPipeline(
        weights_path=weights,
        conf_thresh=0.15,
        iou_thresh=0.45,
        device="mps",
        fps=30.0,
        satellite_id="NASA-ORBIT-SURVEILLANCE",
    )

    if video_choice == 1:
        video_in = str(BASE_DIR / "data" / "real_cubesat_deploy_nasa.mp4")
        video_out = str(OUTPUT_DIR / "tracked_cubesat_deploy_demo.mp4")
    elif video_choice == 2:
        video_in = str(BASE_DIR / "data" / "real_space_video_1.mp4")
        video_out = str(OUTPUT_DIR / "tracked_ocsd_orbital_demo.mp4")
    else:
        video_in = str(BASE_DIR / "data" / "real_space_video_2.mp4")
        video_out = str(OUTPUT_DIR / "tracked_dart_rendezvous_demo.mp4")

    print(f"   Input video:  {video_in}")
    print(f"   Output video: {video_out}")
    print(f"   Max frames:   {max_frames}")

    telemetry = pipeline.run_on_video(
        video_path=video_in,
        output_video_path=video_out,
        max_frames=max_frames,
        save_keyframes=True,
        output_dir=str(OUTPUT_DIR),
    )

    print("\n✅ Real video processing completed!")
    print(f"   Total frames processed: {len(telemetry)}")
    return video_out, telemetry


if __name__ == "__main__":
    choice = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    run_real_video_tests(choice, frames)
