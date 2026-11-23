#!/bin/bash
# Space-Edge-AI: NVIDIA Jetson Orin Nano <10W Uydu Modu ve TensorRT INT8 Derleyici
set -e

echo "[*] Jetson Orin Nano 7W/10W CubeSat Güç Moduna (MAXN_ECO) alınıyor..."
if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 1
    sudo jetson_clocks
fi

ONNX_MODEL="runs/space_debris/p2head/weights/best.onnx"
ENGINE_OUT="runs/space_debris/p2head/weights/best_int8.engine"
CALIB_CACHE="runs/space_debris/p2head/weights/int8_calibration.cache"

echo "[*] TensorRT INT8 Kuantizasyonu ve Katman Profilleme başlatılıyor..."
if [ -f "/usr/src/tensorrt/bin/trtexec" ]; then
    /usr/src/tensorrt/bin/trtexec \
        --onnx="${ONNX_MODEL}" \
        --saveEngine="${ENGINE_OUT}" \
        --calib="${CALIB_CACHE}" \
        --int8 \
        --fp16 \
        --workspace=2048 \
        --avgRuns=200 \
        --warmUp=500 \
        --dumpProfile \
        --exportProfile="runs/space_debris/tensorrt_int8_layer_profile.json"
    echo "[+] TensorRT INT8 Engine başarıyla üretildi: ${ENGINE_OUT}"
else
    echo "[!] trtexec bulunamadı. Lütfen bu betiği JetPack 6.x yüklü Jetson Orin Nano üzerinde çalıştırın."
fi
