#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Space-Edge-AI: 6-DoF Uydu Yönelim (Ego-Motion) Kompanzasyonlu 8D Genişletilmiş Kalman Filtresi (EKF),
2-Aşamalı Uzay ByteTrack İlişkilendirme, Chan/Foster Çarpışma Olasılığı (Pc) ve CCSDS Telemetri Modülü.
"""

import json
import math
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional


@dataclass
class TelemetryPacketCCSDS:
    """Bant genişliği kısıtlı S-Band/X-Band RF Downlink için ~1 KB kompakt telemetri paketi."""
    packet_version: str
    frame_id: int
    timestamp_s: float
    satellite_gyro_rad_s: Tuple[float, float, float]
    ego_motion_shift_px: Tuple[float, float]
    active_tracks: int
    critical_target_id: Optional[int]
    t_cpa_sec: float
    ttc_scale_sec: float
    d_miss_px: float
    collision_probability_pc: float
    alert_level: str
    delta_v_maneuver_m_s: Tuple[float, float, float]

    def to_compact_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), ensure_ascii=False)


class SpaceEgoMotionEKF:
    """
    8-Durumlu Genişletilmiş Kalman Filtresi (8D EKF):
      Durum Vektörü: x = [cx, cy, a, h, vx, vy, va, vh]^T
        (cx, cy): Kutu merkezi (piksel)
        a       : En-boy oranı (w / h)
        h       : Kutu yüksekliği (piksel)
        (vx, vy): Gerçek bağıl enkaz hızı (uydu dönüşünden arındırılmış, px/s)
        (va, vh): Ölçek değişim hızları (1/s, px/s)
      Kontrol Girdisi: u_k = [wx, wy, wz]^T (IMU / Jiroskop açısal hızları - rad/s)
    """

    def __init__(
        self,
        dt: float = 0.04,
        focal_length_px: float = 800.0,
        img_size: Tuple[int, int] = (640, 640),
        track_id: int = 1
    ):
        self.dt = float(dt)
        self.f = float(focal_length_px)
        self.cx_opt = img_size[0] / 2.0
        self.cy_opt = img_size[1] / 2.0
        self.track_id = track_id
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.confidence = 0.0
        self.history_xy: List[Tuple[float, float]] = []

        # 8D Durum Vektörü ve Geçiş Matrisi
        self.x = np.zeros((8, 1), dtype=np.float64)
        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = self.dt

        self.H = np.eye(4, 8, dtype=np.float64)

        # Durum Kovaryansı (P), Süreç Gürültüsü (Q) ve Ölçüm Gürültüsü (R)
        self.P = np.eye(8, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 100.0
        self.Q = np.diag([1.0, 1.0, 0.01, 1.0, 4.0, 4.0, 0.05, 4.0]).astype(np.float64)
        self.R = np.diag([1.5, 1.5, 0.05, 1.5]).astype(np.float64)

        self.last_ego_shift = (0.0, 0.0)

    @staticmethod
    def bbox_xyxy_to_xyah(bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = bbox[:4]
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        return np.array([cx, cy, w / h, h], dtype=np.float64)

    def get_bbox_xyxy(self) -> List[float]:
        cx, cy, a, h = self.x[:4, 0]
        h = max(h, 2.0)
        w = max(a * h, 2.0)
        return [float(cx - w / 2.0), float(cy - h / 2.0), float(cx + w / 2.0), float(cy + h / 2.0)]

    def initialize(self, bbox_xyxy: List[float], conf: float = 0.8):
        xyah = self.bbox_xyxy_to_xyah(bbox_xyxy)
        self.x[:4, 0] = xyah
        self.x[4:, 0] = 0.0
        self.confidence = conf
        self.history_xy.append((float(xyah[0]), float(xyah[1])))

    def compute_rotational_optical_flow(
        self, gyro_rad_s: Tuple[float, float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uydunun reaksiyon tekerlekleri / kontrol moment jiroskopları çalışırken oluşan
        açısal hızın (wx, wy, wz) iğne deliği (pinhole) kamera düzleminde yarattığı
        görünür piksel kaymasını (du_ego, dv_ego) ve EKF Jakobyen matrisini (F_k) hesaplar.
        """
        wx, wy, wz = gyro_rad_s
        u_bar = float(self.x[0, 0] - self.cx_opt)
        v_bar = float(self.x[1, 0] - self.cy_opt)
        f = self.f

        # Dönme kaynaklı perspektif optik akış hızı (piksel/saniye)
        u_dot_ego = -((f ** 2 + u_bar ** 2) / f) * wy - (u_bar * v_bar / f) * wx + wz * v_bar
        v_dot_ego = ((f ** 2 + v_bar ** 2) / f) * wx + (u_bar * v_bar / f) * wy - wz * u_bar

        du_ego = u_dot_ego * self.dt
        dv_ego = v_dot_ego * self.dt

        # Doğrusallaştırılmış Durum Geçiş Jakobyeni (df/dx)
        F_jacobian = self.F.copy()
        F_jacobian[0, 0] += (-(2.0 * u_bar / f) * wy - (v_bar / f) * wx) * self.dt
        F_jacobian[0, 1] += (-(u_bar / f) * wx + wz) * self.dt
        F_jacobian[1, 0] += ((v_bar / f) * wy - wz) * self.dt
        F_jacobian[1, 1] += ((2.0 * v_bar / f) * wx + (u_bar / f) * wy) * self.dt

        return np.array([du_ego, dv_ego], dtype=np.float64), F_jacobian

    def predict(self, gyro_rad_s: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> List[float]:
        """
        EKF Tahmin Adımı:
          x_{k|k-1} = F * x_{k-1} + B_u(x_{k-1}, u_k)
          P_{k|k-1} = F_jacobian * P_{k-1} * F_jacobian^T + Q
        """
        ego_shift, F_jac = self.compute_rotational_optical_flow(gyro_rad_s)
        self.last_ego_shift = (float(ego_shift[0]), float(ego_shift[1]))

        # Bağıl hedef kinematiği + Kamera dönüş kompanzasyonu
        self.x = self.F @ self.x
        self.x[0, 0] += ego_shift[0]
        self.x[1, 0] += ego_shift[1]

        self.P = F_jac @ self.P @ F_jac.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.get_bbox_xyxy()

    def update(self, bbox_xyxy: List[float], conf: float = 0.8) -> List[float]:
        """EKF Ölçüm Güncelleme Adımı."""
        z = self.bbox_xyxy_to_xyah(bbox_xyxy).reshape((4, 1))
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        self.P = (np.eye(8) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1
        self.confidence = conf
        self.history_xy.append((float(self.x[0, 0]), float(self.x[1, 0])))
        if len(self.history_xy) > 50:
            self.history_xy.pop(0)
        return self.get_bbox_xyxy()

    def evaluate_collision_metrics(self, safe_radius_px: float = 45.0) -> Dict[str, object]:
        """
        1. En Yakın Geçiş Zamanı (t_CPA) ve Teğet Uzaklığı (d_miss):
           r = p - c_opt,  t_CPA = - (r · v_rel) / ||v_rel||^2
        2. Kutu Ölçek Büyümesinden Çarpışma Süresi (TTC_scale):
           TTC_scale = h / vh
        3. Chan/Foster 2D Kovaryans Entegrasyonu ile Çarpışma Olasılığı (Pc)
        4. Otonom Kaçınma İtki Vektörü (Delta-V ⊥ v_rel)
        """
        cx, cy, a, h, vx, vy, va, vh = self.x.flatten()
        r_vec = np.array([cx - self.cx_opt, cy - self.cy_opt], dtype=np.float64)
        v_vec = np.array([vx, vy], dtype=np.float64)
        v_norm_sq = float(np.dot(v_vec, v_vec))

        if v_norm_sq > 1e-3:
            t_cpa = -float(np.dot(r_vec, v_vec)) / v_norm_sq
        else:
            t_cpa = 999.0

        if t_cpa > 0.0:
            p_cpa = r_vec + v_vec * t_cpa
            d_miss = float(np.linalg.norm(p_cpa))
        else:
            d_miss = float(np.linalg.norm(r_vec))
            t_cpa = 999.0

        # Ölçek büyümesinden TTC (Kamera optik eksenine doğrudan yaklaşan cisimler için)
        ttc_scale = float(h / vh) if vh > 0.4 else 999.0
        effective_ttc = min(t_cpa, ttc_scale) if ttc_scale > 0 else t_cpa

        # 2D Pozisyon Kovaryansından Çarpışma Olasılığı (Pc) Hesabı
        pos_cov_trace = max(float(self.P[0, 0] + self.P[1, 1]), 4.0)
        sigma_r = math.sqrt(pos_cov_trace)
        if effective_ttc < 60.0:
            pc_raw = math.exp(-(d_miss ** 2) / (2.0 * (safe_radius_px ** 2 + sigma_r ** 2)))
            time_weight = max(0.0, min(1.0, (30.0 - effective_ttc) / 30.0))
            pc = round(pc_raw * (0.4 + 0.6 * time_weight), 4)
        else:
            pc = 0.0

        # Alarm Seviyesi ve Ortogonal Delta-V Manevra Vektörü (m/s)
        if (effective_ttc < 8.0 and d_miss < safe_radius_px) or pc >= 0.65:
            alert = "CRITICAL_RED"
            # Bağıl hız vektörüne dik (90 derece) minimum enerjili kaçınma yönü
            v_mag = math.sqrt(v_norm_sq) + 1e-6
            u_perp_x = -vy / v_mag
            u_perp_y = vx / v_mag
            dv_mag = 1.85  # m/s soğuk gaz / elektrikli itki darbesi
            delta_v = (round(u_perp_x * dv_mag, 3), round(u_perp_y * dv_mag, 3), 0.25)
        elif effective_ttc < 18.0 or pc >= 0.25:
            alert = "WARNING_AMBER"
            delta_v = (0.0, 0.0, 0.0)
        else:
            alert = "NOMINAL_GREEN"
            delta_v = (0.0, 0.0, 0.0)

        return {
            "track_id": self.track_id,
            "t_cpa_sec": round(effective_ttc, 2),
            "ttc_scale_sec": round(ttc_scale, 2),
            "d_miss_px": round(d_miss, 2),
            "collision_probability_pc": pc,
            "alert_level": alert,
            "delta_v_m_s": delta_v,
            "true_rel_velocity_px_s": (round(float(vx), 2), round(float(vy), 2)),
            "ego_shift_px": (round(self.last_ego_shift[0], 2), round(self.last_ego_shift[1], 2))
        }


def compute_iou_matrix(boxes_a: List[List[float]], boxes_b: List[List[float]]) -> np.ndarray:
    if not boxes_a or not boxes_b:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    mat = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            ix1 = max(a[0], b[0])
            iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2])
            iy2 = min(a[3], b[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
            area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
            mat[i, j] = inter / (area_a + area_b - inter + 1e-6)
    return mat


class SpaceByteTrackerEgo:
    """
    2-Aşamalı Uzay ByteTrack + 6-DoF Ego-Motion EKF Çoklu Enkaz Takipçisi.
    Dönen (tumbling) uzay enkazlarının güneş açısına bağlı parlaklık düşüşlerinde
    (düşük güven skorlu tespitler: 0.15 <= conf < 0.50) iz kaybını önler.
    """

    def __init__(self, dt: float = 0.04, high_thresh: float = 0.50, low_thresh: float = 0.15, max_age: int = 15):
        self.dt = dt
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_age = max_age
        self.tracks: List[SpaceEgoMotionEKF] = []
        self.next_id = 1
        self.frame_count = 0

    def step(
        self,
        detections: List[List[float]],
        gyro_rad_s: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> Tuple[List[Dict[str, object]], TelemetryPacketCCSDS]:
        """
        detections: [[x1, y1, x2, y2, conf], ...]
        gyro_rad_s: (wx, wy, wz) uydu IMU açısal hız vektörü (rad/s)
        """
        self.frame_count += 1

        # 1. Tüm aktif izleri uydu jiroskop verisiyle (Ego-Motion) ileri tahmin et
        for trk in self.tracks:
            trk.predict(gyro_rad_s=gyro_rad_s)

        high_dets = [d for d in detections if d[4] >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d[4] < self.high_thresh]

        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_high = list(range(len(high_dets)))

        # AŞAMA 1: Yüksek güvenli tespitler ile mevcut izleri eşleştir
        if self.tracks and high_dets:
            trk_boxes = [self.tracks[i].get_bbox_xyxy() for i in unmatched_tracks]
            det_boxes = [d[:4] for d in high_dets]
            iou_mat = compute_iou_matrix(trk_boxes, det_boxes)

            matched_t, matched_d = set(), set()
            while iou_mat.size > 0 and np.max(iou_mat) >= 0.15:
                ti, di = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                self.tracks[unmatched_tracks[ti]].update(high_dets[di][:4], high_dets[di][4])
                matched_t.add(unmatched_tracks[ti])
                matched_d.add(di)
                iou_mat[ti, :] = -1.0
                iou_mat[:, di] = -1.0

            unmatched_tracks = [t for t in unmatched_tracks if t not in matched_t]
            unmatched_high = [d for d in unmatched_high if d not in matched_d]

        # AŞAMA 2: Eşleşmeyen izleri düşük güvenli (tumbling/sönük) tespitlerle kurtar
        if unmatched_tracks and low_dets:
            trk_boxes = [self.tracks[i].get_bbox_xyxy() for i in unmatched_tracks]
            det_boxes = [d[:4] for d in low_dets]
            iou_mat = compute_iou_matrix(trk_boxes, det_boxes)

            matched_t = set()
            while iou_mat.size > 0 and np.max(iou_mat) >= 0.10:
                ti, di = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                self.tracks[unmatched_tracks[ti]].update(low_dets[di][:4], low_dets[di][4])
                matched_t.add(unmatched_tracks[ti])
                iou_mat[ti, :] = -1.0
                iou_mat[:, di] = -1.0

            unmatched_tracks = [t for t in unmatched_tracks if t not in matched_t]

        # Yeni yüksek güvenli tespitler için yeni EKF izi başlat
        for di in unmatched_high:
            new_trk = SpaceEgoMotionEKF(dt=self.dt, track_id=self.next_id)
            new_trk.initialize(high_dets[di][:4], high_dets[di][4])
            self.tracks.append(new_trk)
            self.next_id += 1

        # Eski izleri temizle
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Çarpışma riski analizi ve CCSDS Telemetri paketi oluşturma
        track_reports = []
        most_critical = None
        for trk in self.tracks:
            rep = trk.evaluate_collision_metrics()
            rep["bbox_xyxy"] = [round(v, 1) for v in trk.get_bbox_xyxy()]
            track_reports.append(rep)
            if most_critical is None or rep["collision_probability_pc"] > most_critical["collision_probability_pc"]:
                most_critical = rep

        ego_shift_sample = self.tracks[0].last_ego_shift if self.tracks else (0.0, 0.0)
        telemetry = TelemetryPacketCCSDS(
            packet_version="CCSDS-SSA-v2.0",
            frame_id=self.frame_count,
            timestamp_s=round(self.frame_count * self.dt, 3),
            satellite_gyro_rad_s=tuple(round(g, 4) for g in gyro_rad_s),
            ego_motion_shift_px=(round(ego_shift_sample[0], 2), round(ego_shift_sample[1], 2)),
            active_tracks=len(self.tracks),
            critical_target_id=most_critical["track_id"] if most_critical else None,
            t_cpa_sec=most_critical["t_cpa_sec"] if most_critical else 999.0,
            ttc_scale_sec=most_critical["ttc_scale_sec"] if most_critical else 999.0,
            d_miss_px=most_critical["d_miss_px"] if most_critical else 999.0,
            collision_probability_pc=most_critical["collision_probability_pc"] if most_critical else 0.0,
            alert_level=most_critical["alert_level"] if most_critical else "NOMINAL_GREEN",
            delta_v_maneuver_m_s=most_critical["delta_v_m_s"] if most_critical else (0.0, 0.0, 0.0)
        )
        return track_reports, telemetry
