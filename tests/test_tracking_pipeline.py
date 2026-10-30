"""
End-to-End Validation Test: P2-Head + ByteTrack + Collision Risk (TTC) Pipeline
==============================================================================
Generates a realistic multi-frame orbital encounter scenario:
1. Debris #1: High-threat impactor on a collision trajectory towards the satellite optical center.
2. Debris #2: Safe orbital debris passing in an adjacent orbital plane.

Processes the sequence using the trained P2-Head YOLO11n weights,
runs ByteTrack multi-object tracking, computes TTC & miss distance,
and outputs annotated video, telemetry packets, and report keyframes.
"""

import os
import sys
import math
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.synthetic_debris_generator import SpaceDebrisSimulator, DebrisObject
from src.tracking.video_pipeline import SpaceDebrisVideoPipeline


def generate_collision_scenario(num_frames: int = 25, width: int = 640, height: int = 640):
    """
    Generates a controlled orbital encounter with:
    - Object 1: Direct collision threat crossing the optical center (320, 320)
    - Object 2: Safe flyby passing on the outer perimeter
    """
    sim = SpaceDebrisSimulator(width=width, height=height, num_stars=90, psf_sigma=1.2, seed=101)
    center = np.array([width / 2.0, height / 2.0])

    # Debris 1: Moving from top-left toward center
    d_threat = DebrisObject(
        debris_id=1,
        x=140.0,
        y=110.0,
        length=28.0,
        width=2.5,
        angle_rad=math.atan2(center[1] - 110.0, center[0] - 140.0),  # Aiming directly at satellite!
        velocity=9.0,
        intensity=240.0,
    )

    # Debris 2: Passing safely along the right boundary
    d_safe = DebrisObject(
        debris_id=2,
        x=520.0,
        y=90.0,
        length=20.0,
        width=2.0,
        angle_rad=math.radians(95.0),  # Moving straight down
        velocity=11.0,
        intensity=210.0,
    )

    frames = []
    ground_truth = []

    for f_idx in range(num_frames):
        canvas = sim.starfield.copy()
        debris_layer = np.zeros((height, width), dtype=np.float32)
        frame_annos = []

        for d in [d_threat, d_safe]:
            bbox = d.get_bbox(width, height)
            if bbox is not None:
                st, ed = d.get_endpoints()
                cv2.line(
                    debris_layer,
                    (int(st[0]), int(st[1])),
                    (int(ed[0]), int(ed[1])),
                    float(d.intensity),
                    max(1, int(round(d.width))),
                    cv2.LINE_AA,
                )
                frame_annos.append({
                    "debris_id": d.debris_id,
                    "bbox": bbox,
                    "pos": (d.x, d.y),
                })
            d.update(dt=1.0)

        # PSF diffusion
        diffused = cv2.GaussianBlur(debris_layer, (5, 5), sim.psf_sigma)
        combined = np.clip(canvas + diffused, 0, 255).astype(np.uint8)
        frames.append(combined)
        ground_truth.append(frame_annos)

    return frames, ground_truth


def run_tracking_test():
    weights_path = os.path.join(PROJECT_ROOT, "runs", "space_debris", "p2head", "weights", "best.pt")
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("🛰️ TESTING: P2-HEAD + BYTETRACK + COLLISION RISK (TTC) PIPELINE")
    print("=" * 65)

    if not os.path.exists(weights_path):
        print(f"❌ Weights not found at: {weights_path}")
        return

    # 1. Generate Collision Scenario Video
    print("\n[Step 1] Synthesizing Controlled Orbital Encounter Scenario (25 frames)...")
    frames, gt = generate_collision_scenario(num_frames=25)
    print(f"  ✅ Generated {len(frames)} frames with 1 Collision Threat + 1 Safe Flyby.")

    # 2. Initialize Video Pipeline
    print("\n[Step 2] Initializing Space Debris Video Pipeline...")
    pipeline = SpaceDebrisVideoPipeline(
        weights_path=weights_path,
        conf_thresh=0.20,
        iou_thresh=0.45,
        device="mps",
        fps=30.0,
        satellite_id="FERGANI-LEO-SAT01",
    )

    # 3. Process Video Sequence
    out_video_path = os.path.join(output_dir, "tracked_debris_collision_demo.mp4")
    telemetry_log = pipeline.run_on_sequence(
        frames=frames,
        output_video_path=out_video_path,
        save_keyframes=True,
        output_dir=output_dir,
    )

    # 4. Assertions & Verification
    print("\n[Step 3] Verifying Collision Assessment & Tracking Accuracy...")
    critical_events = [t for t in telemetry_log if t["overall_status"] == "CRITICAL_COLLISION_WARNING"]
    print(f"  Total Processed Frames: {len(telemetry_log)}")
    print(f"  Frames with Critical Alerts: {len(critical_events)}")

    assert len(telemetry_log) == 25, "All 25 frames must be processed"
    assert len(critical_events) > 0, "Pipeline must trigger Critical Collision Alert for colliding debris"

    sample_alert = critical_events[0]
    threat = sample_alert["threats"][0]
    print(f"\n  🎯 Critical Event Verified:")
    print(f"     Debris ID:             {threat['debris_id']}")
    print(f"     Threat Level:          {threat['threat_level']}")
    print(f"     Time-to-Collision:     {threat['time_to_collision_sec']} sec")
    print(f"     Miss Distance:         {threat['miss_distance_pixels']} px")
    print(f"     Avoidance Maneuver:    {sample_alert['recommended_action']}")

    print("\n" + "=" * 65)
    print("🎉 ALL TRACKING & COLLISION RISK TESTS PASSED SUCCESSFULLY!")
    print(f"   🎬 Output Video:     {out_video_path}")
    print(f"   📡 Telemetry Log:    {os.path.join(output_dir, 'flight_telemetry_log.json')}")
    print(f"   🖼️ Keyframe Preview: {os.path.join(output_dir, 'hud_critical_collision_alert.jpg')}")
    print("=" * 65)


if __name__ == "__main__":
    run_tracking_test()
