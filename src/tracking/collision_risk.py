"""
Collision Risk and Time-to-Collision (TTC) Assessment Module.
==============================================================
Provides real-time space situational awareness (SSA) for on-orbit satellites.

Key Aerospace & CV Computations:
1. Closest Point of Approach (t_CPA):
      t_CPA = - (r . v) / ||v||^2
2. Minimum Miss Distance (d_miss):
      d_miss = || p(t_CPA) - c_opt ||
3. Optical Expansion Rate (TTC):
      tau = height / v_height
4. Collision Risk Levels:
      CRITICAL_ALERT, WARNING, APPROACHING_SAFE, DEPARTING_CLEAR
5. Autonomous Avoidance Maneuver Vector (delta-V direction)
6. CCSDS / Fergani Telemetry Packet Generator
"""

import math
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class CollisionRiskLevel:
    CRITICAL_ALERT = "CRITICAL_COLLISION_ALERT"
    WARNING = "COLLISION_WARNING"
    APPROACHING = "APPROACHING_SAFE"
    DEPARTING = "DEPARTING_CLEAR"
    STATIONARY = "STATIONARY"


class CollisionRiskAnalyzer:
    """
    Analyzes tracked debris trajectories relative to the host satellite optical center
    to evaluate collision threats and compute Time-to-Collision (TTC).
    """

    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 640,
        fps: float = 30.0,
        safety_radius_px: float = 80.0,
        caution_radius_px: float = 160.0,
        critical_ttc_sec: float = 5.0,
        warning_ttc_sec: float = 12.0,
    ):
        self.width = frame_width
        self.height = frame_height
        self.fps = fps
        self.center = np.array([frame_width / 2.0, frame_height / 2.0], dtype=np.float32)

        self.safety_radius = safety_radius_px
        self.caution_radius = caution_radius_px
        self.critical_ttc = critical_ttc_sec
        self.warning_ttc = warning_ttc_sec

    def analyze_track(self, track) -> Dict:
        """
        Analyzes a single SDebrisTrack object.
        Returns a dictionary containing risk metrics and telemetry.
        """
        tlwh = track.to_tlwh()
        pos = np.array([tlwh[0] + tlwh[2] / 2.0, tlwh[1] + tlwh[3] / 2.0], dtype=np.float32)
        r = pos - self.center  # Vector from satellite center to debris

        # Velocity in pixels/sec
        vx, vy = track.get_velocity()
        # Scale by fps
        v_px_sec = np.array([vx * self.fps, vy * self.fps], dtype=np.float32)
        speed = float(np.linalg.norm(v_px_sec))

        current_dist = float(np.linalg.norm(r))

        # Check for near-zero velocity
        if speed < 1.0:
            return {
                "track_id": track.track_id,
                "current_dist_px": round(current_dist, 1),
                "speed_px_sec": round(speed, 1),
                "t_cpa_sec": 999.0,
                "miss_dist_px": round(current_dist, 1),
                "risk_level": CollisionRiskLevel.STATIONARY,
                "is_critical": False,
                "cpa_pos": pos.tolist(),
                "maneuver_vector": [0.0, 0.0],
            }

        # ── Compute Closest Point of Approach (CPA) ──────────────────────────
        # t_cpa = - (r . v) / ||v||^2
        dot_product = float(np.dot(r, v_px_sec))
        t_cpa = -dot_product / (speed**2 + 1e-6)

        # Projected position at CPA
        if t_cpa > 0:
            cpa_pos = pos + v_px_sec * t_cpa
            miss_dist = float(np.linalg.norm(cpa_pos - self.center))
        else:
            cpa_pos = pos
            miss_dist = current_dist

        # ── Categorize Risk Level ─────────────────────────────────────────────
        is_critical = False
        if t_cpa <= 0:
            risk_level = CollisionRiskLevel.DEPARTING
        else:
            if miss_dist <= self.safety_radius and t_cpa <= self.critical_ttc:
                risk_level = CollisionRiskLevel.CRITICAL_ALERT
                is_critical = True
            elif miss_dist <= self.caution_radius and t_cpa <= self.warning_ttc:
                risk_level = CollisionRiskLevel.WARNING
            else:
                risk_level = CollisionRiskLevel.APPROACHING

        # ── Optimal Evasive Maneuver Vector (Perpendicular to relative velocity)
        if speed > 0:
            # Perpendicular vector in 2D: [-vy, vx]
            perp = np.array([-v_px_sec[1], v_px_sec[0]], dtype=np.float32) / speed
            # Direct away from optical center
            test_point = self.center + perp * 50.0
            if np.linalg.norm(test_point - pos) < current_dist:
                perp = -perp
            maneuver_vector = perp.tolist()
        else:
            maneuver_vector = [0.0, 0.0]

        return {
            "track_id": track.track_id,
            "current_dist_px": round(current_dist, 1),
            "speed_px_sec": round(speed, 1),
            "velocity_px_sec": [round(float(v_px_sec[0]), 1), round(float(v_px_sec[1]), 1)],
            "t_cpa_sec": round(float(t_cpa), 2) if t_cpa > 0 else 0.0,
            "miss_dist_px": round(miss_dist, 1),
            "risk_level": risk_level,
            "is_critical": is_critical,
            "cpa_pos": [round(float(cpa_pos[0]), 1), round(float(cpa_pos[1]), 1)],
            "maneuver_vector": [round(float(maneuver_vector[0]), 3), round(float(maneuver_vector[1]), 3)],
        }

    def generate_telemetry_packet(
        self,
        satellite_id: str,
        frame_idx: int,
        tracks_risk_list: List[Dict],
    ) -> Dict:
        """
        Creates a compact on-board telemetry packet (~500 bytes - 1 KB).
        """
        critical_threats = [t for t in tracks_risk_list if t["is_critical"]]
        warning_threats = [t for t in tracks_risk_list if t["risk_level"] == CollisionRiskLevel.WARNING]

        overall_status = "SAFE"
        recommended_action = "CONTINUE_NOMINAL_ATTITUDE"

        if critical_threats:
            overall_status = "CRITICAL_COLLISION_WARNING"
            most_urgent = min(critical_threats, key=lambda t: t["t_cpa_sec"])
            mx, my = most_urgent["maneuver_vector"]
            recommended_action = f"EXECUTE_IMPULSIVE_BURN_DIRECTION_[{mx:+.2f},{my:+.2f}]"
        elif warning_threats:
            overall_status = "COLLISION_WARNING_MONITORING"
            recommended_action = "CHARGE_PROPULSION_PREPARE_MANEUVER"

        packet = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "satellite_id": satellite_id,
            "frame_index": frame_idx,
            "overall_status": overall_status,
            "active_tracks_count": len(tracks_risk_list),
            "critical_threats_count": len(critical_threats),
            "recommended_action": recommended_action,
            "threats": [
                {
                    "debris_id": t["track_id"],
                    "threat_level": t["risk_level"],
                    "time_to_collision_sec": t["t_cpa_sec"],
                    "miss_distance_pixels": t["miss_dist_px"],
                    "relative_velocity_pixels_sec": t.get("velocity_px_sec", [0, 0]),
                    "evasive_vector": t["maneuver_vector"],
                }
                for t in tracks_risk_list
                if t["risk_level"] in [CollisionRiskLevel.CRITICAL_ALERT, CollisionRiskLevel.WARNING]
            ],
        }
        return packet
