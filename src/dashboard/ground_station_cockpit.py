#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Space-Edge-AI: Yer İstasyonu Grafiksel Kokpit Paneli (Ground Station Mission Control Dashboard)
Çalıştırmak için:
    streamlit run src/dashboard/ground_station_cockpit.py
"""

import json
import sys
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.ego_motion_ekf import SpaceByteTrackerEgo

import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Space-Edge-AI | Yer İstasyonu Kokpiti",
    page_icon="🛰️",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background-color: #070b14; color: #e2f1ff; }
div[data-testid="stMetricValue"] { color: #00f5d4; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

st.title("🛰️ SPACE-EDGE-AI: LEO OTONOM UZAY ENKAZI TAKİP VE KAÇINMA YER İSTASYONU")
st.caption("P2-Head YOLOv11n (INT8) + 6-DoF Ego-Motion 8D EKF + Uzay ByteTrack + CCSDS Telemetri Kokpiti")

# Yan Panel: Uydu Yönelim (IMU/Gyro) ve Donanım Kontrolleri
st.sidebar.header("⚙️ Uydu Aviyonik & GNC Kontrolü")
wx = st.sidebar.slider("Pitch Açısal Hızı ω_x (rad/s)", -0.30, 0.30, 0.06, 0.01)
wy = st.sidebar.slider("Yaw Açısal Hızı ω_y (rad/s)", -0.30, 0.30, -0.04, 0.01)
wz = st.sidebar.slider("Roll Açısal Hızı ω_z (rad/s)", -0.30, 0.30, 0.02, 0.01)
sim_step = st.sidebar.slider("Yörünge Yaklaşma Zaman Adımı (Kare)", 1, 25, 18)
hw_mode = st.sidebar.selectbox(
    "On-Board Çıkarım Donanımı",
    [
        "Jetson Orin Nano INT8 (22.4 ms | 44.6 FPS | 7.4W)",
        "Apple M1 CoreML ANE INT8 (24.8 ms | 40.3 FPS | 6.8W)",
        "Apple M1 CPU FP32 Baz (116.6 ms | 8.6 FPS | 15.2W)"
    ]
)

# EKF + ByteTrack Simülasyonunu Seçilen Adıma Kadar Koştur
tracker = SpaceByteTrackerEgo(dt=0.04)
for step_idx in range(sim_step):
    # Merkeze (320, 320) doğru hızla yaklaşan ve büyüyen kritik enkaz + ikinci teğet enkaz
    cx1 = 160.0 + step_idx * 7.8
    cy1 = 140.0 + step_idx * 8.2
    s1 = 6.0 + step_idx * 1.1
    cx2 = 520.0 - step_idx * 4.5
    cy2 = 120.0 + step_idx * 6.0
    dets = [
        [cx1 - s1, cy1 - s1, cx1 + s1, cy1 + s1, 0.91],
        [cx2 - 5, cy2 - 5, cx2 + 5, cy2 + 5, 0.78]
    ]
    reports, telemetry = tracker.step(dets, gyro_rad_s=(wx, wy, wz))

# Üst Telemetri Göstergeleri
m1, m2, m3, m4, m5 = st.columns(5)
if "Jetson" in hw_mode:
    m1.metric("Çıkarım Hızı (INT8)", "44.6 FPS", "+36.0 FPS (vs FP32)")
    m2.metric("Boru Hattı Gecikmesi", "22.4 ms", "-94.2 ms (Hedef ≤60ms)")
elif "ANE" in hw_mode:
    m1.metric("Çıkarım Hızı (M1 ANE)", "40.3 FPS", "+31.7 FPS")
    m2.metric("Boru Hattı Gecikmesi", "24.8 ms", "-91.8 ms")
else:
    m1.metric("Çıkarım Hızı (FP32)", "8.6 FPS", "Uydu Sınırı Altında")
    m2.metric("Boru Hattı Gecikmesi", "116.6 ms", "+56.6 ms Aşım")

m3.metric("En Yakın Geçiş (t_CPA)", f"{telemetry.t_cpa_sec:.2f} sn", delta="-0.8 sn", delta_color="inverse")
m4.metric("Çarpışma Olasılığı (Pc)", f"%{telemetry.collision_probability_pc * 100:.1f}", telemetry.alert_level)
m5.metric("Ego-Motion Kompanzasyon", f"{telemetry.ego_motion_shift_px[0]:.1f}, {telemetry.ego_motion_shift_px[1]:.1f} px", "IMU Aktif")

col_cam, col_radar = st.columns([1.15, 1])

with col_cam:
    st.subheader("📷 On-Board Star Tracker HUD & EKF Takip Katmanı")
    hud_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(hud_img, (320, 240), 45, (0, 180, 255), 1)
    cv2.drawMarker(hud_img, (320, 240), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)

    for rep in reports:
        x1, y1, x2, y2 = [int(v * 0.75) if idx % 2 == 1 else int(v) for idx, v in enumerate(rep["bbox_xyxy"])]
        color = (255, 40, 40) if rep["alert_level"] == "CRITICAL_RED" else (0, 255, 160)
        cv2.rectangle(hud_img, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8), color, 2)
        cv2.putText(
            hud_img,
            f"ID:{rep['track_id']} CPA:{rep['t_cpa_sec']}s Pc:{rep['collision_probability_pc']:.2f}",
            (max(10, x1 - 20), max(20, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1
        )
    st.image(hud_img, caption="P2-Head YOLOv11n + 6-DoF Ego-Motion EKF Optik Çıktısı", use_container_width=True)

    if st.button("🚨 ACİL OTONOM KAÇINMA MANEVRASI (DELTA-V İTKİ ATEŞLE)", use_container_width=True):
        dv = telemetry.delta_v_maneuver_m_s
        st.error(f"⚡ AVİYONİK İTKİ KOMUTU GÖNDERİLDİ: ΔV = [{dv[0]} m/s Radial, {dv[1]} m/s In-Track, {dv[2]} m/s Cross-Track]")

with col_radar:
    st.subheader("🌐 3D Bağıl Yörünge Radarı (Hill-Clohessy-Wiltshire Çerçevesi)")
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers+text",
        marker=dict(size=10, color="#00f5d4", symbol="diamond"),
        text=["KENDİ UYDUMUZ"], name="LEO Uydu"
    ))
    for rep in reports:
        bx1, by1, bx2, by2 = rep["bbox_xyxy"]
        rx = ((bx1 + bx2) / 2.0 - 320.0) * 0.25
        ry = ((by1 + by2) / 2.0 - 320.0) * 0.25
        rz = max(2.0, rep["t_cpa_sec"] * 3.5)
        fig.add_trace(go.Scatter3d(
            x=[rx], y=[ry], z=[rz], mode="markers+text",
            marker=dict(size=8, color="red" if rep["alert_level"] == "CRITICAL_RED" else "orange"),
            text=[f"ENKAZ #{rep['track_id']} (CPA:{rep['t_cpa_sec']}s)"],
            name=f"Enkaz #{rep['track_id']}"
        ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-80, 80], title="Radial (m)"),
            yaxis=dict(range=[-80, 80], title="In-Track (m)"),
            zaxis=dict(range=[0, 120], title="Bağıl Mesafe (m)"),
            bgcolor="#070b14"
        ),
        paper_bgcolor="#070b14", font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=10), height=400
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("📡 CCSDS Kompakt RF Downlink Telemetri Paketi (~1 KB JSON)")
st.code(telemetry.to_compact_json(), language="json")
