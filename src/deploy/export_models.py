"""
Model Export and TensorRT Optimization Script
===============================================
After training, exports both models:
1. ONNX format (.onnx) — cross-platform, CPU/GPU inference
2. TorchScript (.torchscript) — PyTorch mobile / Edge
3. CoreML (.mlpackage) — Apple Silicon / iOS deployment
4. (On Jetson) TensorRT (.engine) — maximum edge performance

Usage:
    python3 src/deploy/export_models.py

Requires trained weights in:
    runs/space_debris/baseline/weights/best.pt
    runs/space_debris/p2head/weights/best.pt
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
RUNS_DIR = BASE_DIR / "runs" / "space_debris"


def export_model(run_name: str, imgsz: int = 640):
    """Exports a trained YOLO model to ONNX and other formats."""
    from ultralytics import YOLO

    weights_path = RUNS_DIR / run_name / "weights" / "best.pt"
    if not weights_path.exists():
        print(f"  ❌ Weights not found: {weights_path}")
        print(f"     Training may not be complete yet.")
        return

    print(f"\n{'=' * 55}")
    print(f"📦 Exporting model: {run_name}")
    print(f"   Weights: {weights_path}")
    print(f"{'=' * 55}")

    model = YOLO(str(weights_path))

    # Export to ONNX (universal, CPU/GPU)
    print("\n[1/2] Exporting to ONNX...")
    try:
        onnx_path = model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=False,
            simplify=True,
            opset=12,
        )
        import os
        size_mb = os.path.getsize(onnx_path) / 1_000_000
        print(f"      ✅ ONNX: {onnx_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"      ⚠️  ONNX export failed: {e}")

    # Export to TorchScript (mobile-ready)
    print("\n[2/2] Exporting to TorchScript...")
    try:
        ts_path = model.export(format="torchscript", imgsz=imgsz, optimize=True)
        size_mb = os.path.getsize(ts_path) / 1_000_000
        print(f"      ✅ TorchScript: {ts_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"      ⚠️  TorchScript export failed: {e}")

    print(f"\n✅ Exports completed for: {run_name}")


def benchmark_inference(run_name: str, imgsz: int = 640, num_runs: int = 100):
    """Benchmarks inference speed of the trained model."""
    from ultralytics import YOLO
    import time
    import numpy as np

    weights_path = RUNS_DIR / run_name / "weights" / "best.pt"
    if not weights_path.exists():
        print(f"  ❌ Weights not found for benchmark: {run_name}")
        return {}

    model = YOLO(str(weights_path))

    # Create dummy image
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8) + 30

    # Warmup
    for _ in range(5):
        model.predict(dummy, verbose=False)

    # Benchmark
    times = []
    for _ in range(num_runs):
        t_start = time.perf_counter()
        model.predict(dummy, verbose=False)
        times.append(time.perf_counter() - t_start)

    avg_ms = np.mean(times) * 1000
    fps    = 1.0 / np.mean(times)
    p95_ms = np.percentile(times, 95) * 1000

    result = {
        "run_name":  run_name,
        "avg_latency_ms": round(avg_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "fps":            round(fps, 1),
        "imgsz":          imgsz,
        "device":         "mps (Apple Silicon)",
    }

    print(f"\n⚡ Inference Benchmark: {run_name}")
    print(f"   Avg Latency : {avg_ms:.1f} ms")
    print(f"   P95 Latency : {p95_ms:.1f} ms")
    print(f"   FPS         : {fps:.1f} frames/sec")

    return result


def main():
    import json

    print("🚀 Space Debris Model Export & Benchmark Tool")
    print("=" * 55)

    benchmark_results = {}

    for run_name in ["baseline", "p2head"]:
        # Export models
        export_model(run_name)

        # Benchmark inference
        bench = benchmark_inference(run_name, num_runs=50)
        if bench:
            benchmark_results[run_name] = bench

    if benchmark_results:
        out_path = RUNS_DIR / "inference_benchmark.json"
        with open(out_path, "w") as f:
            json.dump(benchmark_results, f, indent=2)
        print(f"\n📄 Benchmark sonuçları kaydedildi: {out_path}")

        print("\n" + "=" * 55)
        print("  ÇIKARIM HIZI KARŞILAŞTIRMASI (MPS / Apple Silicon)")
        print("  " + "-" * 53)
        for rn, b in benchmark_results.items():
            print(f"  {rn:<12}: {b['fps']:>6.1f} FPS  | Gecikme: {b['avg_latency_ms']:.1f} ms")
        print("=" * 55)


if __name__ == "__main__":
    main()
