"""
Unified Space Debris Tracking and Collision Avoidance Video Pipeline.
=====================================================================
Integrates:
1. P2-Head YOLO11n Object Detector (runs/space_debris/p2head/weights/best.pt)
2. SpaceByteTracker (Kalman Filter + 2-Stage Association)
3. CollisionRiskAnalyzer (TTC, Miss Distance, CPA, Avoidance Maneuver)
4. Spacecraft Cockpit HUD / Telemetry Visualizer
"""

import os
import cv2
import json
import time
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from .byte_tracker import SpaceByteTracker, SDebrisTrack
from .collision_risk import CollisionRiskAnalyzer, CollisionRiskLevel


class SpaceDebrisVideoPipeline:
    """End-to-End Tracking & Collision Avoidance Video Processor."""

    def __init__(
        self,
        weights_path: str,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        device: str = "mps",
        fps: float = 30.0,
        satellite_id: str = "FERGANI-LEO-01",
    ):
        from ultralytics import YOLO

        print(f"🛰️ Initializing Space Debris Pipeline...")
        print(f"   Model Weights: {weights_path}")
        print(f"   Inference Device: {device}")

        self.model = YOLO(weights_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.fps = fps
        self.satellite_id = satellite_id

        self.tracker = SpaceByteTracker(
            track_thresh=conf_thresh,
            low_thresh=0.08,
            match_thresh=0.80,
            frame_rate=int(fps),
        )
        self.risk_analyzer = CollisionRiskAnalyzer(
            frame_width=640,
            frame_height=640,
            fps=fps,
            safety_radius_px=75.0,
            caution_radius_px=150.0,
            critical_ttc_sec=4.5,
            warning_ttc_sec=10.0,
        )

        self.telemetry_history: List[Dict] = []

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, Dict]:
        """
        Processes a single video frame through detection, tracking, risk analysis, and HUD rendering.
        """
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        h, w = frame.shape[:2]
        if self.risk_analyzer.width != w or self.risk_analyzer.height != h:
            self.risk_analyzer.width = w
            self.risk_analyzer.height = h
            self.risk_analyzer.center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
            self.risk_analyzer.safety_radius = min(w, h) * 0.12
            self.risk_analyzer.caution_radius = min(w, h) * 0.25

        # ── 1. Inference with P2-Head Model ──────────────────────────────────
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False,
            imgsz=640,
        )[0]

        # Format detections as [x1, y1, x2, y2, conf, class_id]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        if len(boxes) > 0:
            dets = np.column_stack([boxes, scores, classes])
        else:
            dets = np.empty((0, 6))

        # ── 2. ByteTrack Update ──────────────────────────────────────────────
        active_tracks = self.tracker.update(dets)

        # ── 3. Collision Risk Analysis ────────────────────────────────────────
        track_risks = []
        for track in active_tracks:
            risk_info = self.risk_analyzer.analyze_track(track)
            track_risks.append(risk_info)

        # Generate telemetry packet
        telemetry = self.risk_analyzer.generate_telemetry_packet(
            satellite_id=self.satellite_id,
            frame_idx=frame_idx,
            tracks_risk_list=track_risks,
        )
        self.telemetry_history.append(telemetry)

        # ── 4. Render Cockpit HUD ─────────────────────────────────────────────
        annotated_frame = self._render_hud(frame, active_tracks, track_risks, telemetry)

        return annotated_frame, telemetry

    def _render_hud(
        self,
        frame: np.ndarray,
        tracks: List[SDebrisTrack],
        risks: List[Dict],
        telemetry: Dict,
    ) -> np.ndarray:
        """Renders tactical space situational awareness HUD overlay."""
        hud = frame.copy()
        h, w = hud.shape[:2]
        cx, cy = int(w / 2), int(h / 2)

        # ── Reticle & Safety Exclusion Zones ─────────────────────────────────
        # Optical center crosshair
        cv2.line(hud, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(hud, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(hud, (cx, cy), 4, (0, 255, 255), 1, cv2.LINE_AA)

        # Caution zone (yellow dashed)
        cv2.circle(hud, (cx, cy), int(self.risk_analyzer.caution_radius), (0, 200, 255), 1, cv2.LINE_AA)
        # Critical safety bubble (red circle)
        cv2.circle(hud, (cx, cy), int(self.risk_analyzer.safety_radius), (0, 0, 255), 1, cv2.LINE_AA)

        # ── Draw Tracks and Vectors ──────────────────────────────────────────
        risk_dict = {r["track_id"]: r for r in risks}

        for track in tracks:
            tid = track.track_id
            r_info = risk_dict.get(tid, {})
            risk_level = r_info.get("risk_level", CollisionRiskLevel.APPROACHING)
            is_crit = r_info.get("is_critical", False)

            # Color scheme based on risk
            if is_crit:
                box_color = (0, 0, 255)       # Red (Critical)
            elif risk_level == CollisionRiskLevel.WARNING:
                box_color = (0, 165, 255)     # Orange (Warning)
            elif risk_level == CollisionRiskLevel.DEPARTING:
                box_color = (200, 200, 200)   # Gray (Departing)
            else:
                box_color = (0, 255, 0)       # Green (Safe)

            # Bounding box
            tlbr = track.to_tlbr()
            x1, y1, x2, y2 = map(int, tlbr)
            cv2.rectangle(hud, (x1, y1), (x2, y2), box_color, 1 if not is_crit else 2)

            # Trajectory History Tail (motion history line)
            hist = track.history
            if len(hist) > 1:
                pts = np.array(hist, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(hud, [pts], False, box_color, 1, cv2.LINE_AA)

            # Velocity vector arrow
            cur_pos = (int(hist[-1][0]), int(hist[-1][1]))
            vx, vy = track.get_velocity()
            arrow_end = (int(cur_pos[0] + vx * 10.0), int(cur_pos[1] + vy * 10.0))
            cv2.arrowedLine(hud, cur_pos, arrow_end, box_color, 1, tipLength=0.3)

            # Projected CPA point (if critical)
            if is_crit and "cpa_pos" in r_info:
                cpa_pt = (int(r_info["cpa_pos"][0]), int(r_info["cpa_pos"][1]))
                cv2.drawMarker(hud, cpa_pt, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 8, 1)

            # Text label
            ttc_val = r_info.get("t_cpa_sec", 0.0)
            miss_val = r_info.get("miss_dist_px", 0.0)
            spd_val = r_info.get("speed_px_sec", 0.0)

            label = f"ID:{tid} TTC:{ttc_val:.1f}s"
            if is_crit:
                label += " [CRITICAL!]"
            cv2.putText(
                hud,
                label,
                (x1, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                box_color,
                1,
                cv2.LINE_AA,
            )

        # ── Top Banner Alert ─────────────────────────────────────────────────
        status = telemetry["overall_status"]
        if status == "CRITICAL_COLLISION_WARNING":
            banner_color = (0, 0, 200)
            banner_text = f"🚨 CRITICAL COLLISION ALERT - {telemetry['recommended_action']}"
        elif status == "COLLISION_WARNING_MONITORING":
            banner_color = (0, 140, 255)
            banner_text = "⚠️ WARNING: OBJECT IN CAUTION CORRIDOR - ATTITUDE STANDBY"
        else:
            banner_color = (0, 120, 0)
            banner_text = f"🛰️ ORBITAL SURVEILLANCE NOMINAL | TRACKS: {len(tracks)} | PATH CLEAR"

        cv2.rectangle(hud, (0, 0), (w, 28), banner_color, -1)
        cv2.putText(hud, banner_text, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Bottom Telemetry Overlay ─────────────────────────────────────────
        overlay_box = np.zeros((70, 280, 3), dtype=np.uint8)
        cv2.rectangle(overlay_box, (0, 0), (280, 70), (20, 20, 20), -1)
        cv2.rectangle(overlay_box, (0, 0), (280, 70), (60, 60, 60), 1)

        t_lines = [
            f"SAT: {self.satellite_id} | FPS: {self.fps:.0f}",
            f"STATUS: {status}",
            f"TRACKS: {telemetry['active_tracks_count']} | THREATS: {telemetry['critical_threats_count']}",
        ]
        for idx, line in enumerate(t_lines):
            c = (0, 255, 255) if "CRITICAL" not in line else (0, 0, 255)
            cv2.putText(overlay_box, line, (8, 18 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1, cv2.LINE_AA)

        # Blend onto bottom-left
        hud[h - 78 : h - 8, 8 : 288] = overlay_box

        return hud

    def run_on_sequence(
        self,
        frames: List[np.ndarray],
        output_video_path: Optional[str] = None,
        save_keyframes: bool = True,
        output_dir: str = "output",
    ) -> List[Dict]:
        """
        Runs the full tracking and collision pipeline over a sequence of frames.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.tracker.reset()
        self.telemetry_history = []

        annotated_frames = []
        writer = None

        if output_video_path:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, self.fps, (w, h))

        print(f"\n🚀 Running Video Tracking & Collision Pipeline on {len(frames)} frames...")
        critical_frame_saved = False
        nominal_frame_saved = False

        for f_idx, frame in enumerate(frames):
            ann_frame, telemetry = self.process_frame(frame, f_idx)
            annotated_frames.append(ann_frame)

            if writer:
                writer.write(ann_frame)

            # Save sample keyframes for visual reports
            if save_keyframes:
                if telemetry["critical_threats_count"] > 0 and not critical_frame_saved:
                    path_crit = os.path.join(output_dir, "hud_critical_collision_alert.jpg")
                    cv2.imwrite(path_crit, ann_frame)
                    critical_frame_saved = True
                    print(f"  🚨 Saved Critical Alert Frame: {path_crit}")

                if telemetry["critical_threats_count"] == 0 and not nominal_frame_saved and f_idx > 3:
                    path_nom = os.path.join(output_dir, "hud_nominal_surveillance.jpg")
                    cv2.imwrite(path_nom, ann_frame)
                    nominal_frame_saved = True
                    print(f"  ✅ Saved Nominal Frame: {path_nom}")

        if writer:
            writer.release()
            print(f"  🎬 Exported Tracked Video: {output_video_path}")

        # Save flight telemetry log
        telemetry_file = os.path.join(output_dir, "flight_telemetry_log.json")
        with open(telemetry_file, "w") as f:
            json.dump(self.telemetry_history, f, indent=2)
        print(f"  📡 Generated Flight Telemetry Log: {telemetry_file}")

        return self.telemetry_history

    def run_on_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        save_keyframes: bool = True,
        output_dir: str = "output",
        target_fps: Optional[float] = None,
    ) -> List[Dict]:
        """
        Processes a full video file in a memory-efficient streaming manner.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.tracker.reset()
        self.telemetry_history = []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.fps = target_fps if target_fps else src_fps
        self.tracker.frame_rate = int(self.fps)
        self.risk_analyzer.fps = self.fps

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_to_process = min(total_frames, max_frames) if max_frames else total_frames
        print(f"\n🚀 Processing video stream: {video_path}")
        print(f"   Resolution: {w}x{h} | Source FPS: {src_fps:.1f} | Frames to process: {frames_to_process}")

        writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, self.fps, (w, h))

        f_idx = 0
        critical_saved = False
        nominal_saved = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and f_idx >= max_frames):
                break

            ann_frame, telemetry = self.process_frame(frame, f_idx)

            if writer:
                writer.write(ann_frame)

            if save_keyframes:
                if telemetry["critical_threats_count"] > 0 and not critical_saved:
                    path_crit = os.path.join(output_dir, "real_video_hud_critical.jpg")
                    cv2.imwrite(path_crit, ann_frame)
                    critical_saved = True
                    print(f"  🚨 Saved Critical Alert Keyframe: {path_crit}")
                elif telemetry["active_tracks_count"] > 0 and not nominal_saved and f_idx > 5:
                    path_nom = os.path.join(output_dir, "real_video_hud_tracking.jpg")
                    cv2.imwrite(path_nom, ann_frame)
                    nominal_saved = True
                    print(f"  🛰️ Saved Active Tracking Keyframe: {path_nom}")

            f_idx += 1
            if f_idx % 50 == 0 or f_idx == frames_to_process:
                print(f"   Processed frame {f_idx}/{frames_to_process} ({f_idx*100.0/frames_to_process:.1f}%) | Active tracks: {telemetry['active_tracks_count']}")

        cap.release()
        if writer:
            writer.release()
            print(f"  🎬 Exported Tracked Video to: {output_video_path}")

        telemetry_file = os.path.join(output_dir, "real_video_telemetry_log.json")
        with open(telemetry_file, "w") as f:
            json.dump(self.telemetry_history, f, indent=2)
        print(f"  📡 Generated Flight Telemetry Log: {telemetry_file}")

        return self.telemetry_history
