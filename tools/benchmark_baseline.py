#!/usr/bin/env python3
"""
Export the transformer baseline to CoreML and benchmark inference throughput.

Usage:
    python tools/export_baseline.py [--out build/baseline.mlpackage]
    python tools/benchmark_baseline.py [--model build/baseline.mlpackage] [--n 200]

This produces the speed comparison needed for the paper:
    ByteGrid-44M  (hand-written MIL, private ANE runtime)  →  11.2 ms/pass
    Transformer-44M (CoreML export, system-managed routing) →  ? ms/pass

Compute unit options:
    --compute-units cpuAndNeuralEngine   # force ANE (may fall back)
    --compute-units cpuAndGPU            # force GPU  
    --compute-units all                  # system chooses (default)
"""
import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def export(out_path: Path):
    import torch
    import coremltools as ct
    from training.baseline_transformer_pt import make_model, INPUT_CHANNELS, SEQ

    print("Building model...")
    model = make_model("cpu")
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n:,} ({n/1e6:.1f}M)")

    # Trace with a dummy input
    dummy = torch.randn(1, INPUT_CHANNELS, 1, SEQ, dtype=torch.float32)
    print("Tracing model with torch.jit.trace ...")
    traced = torch.jit.trace(model, dummy)

    # Convert to CoreML
    print("Converting to CoreML (fp16)...")
    ml_model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x_ane", shape=dummy.shape, dtype=float)],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ml_model.save(str(out_path))
    print(f"Saved: {out_path}")
    return out_path


def benchmark(model_path: Path, n_passes: int, compute_units: str):
    import coremltools as ct
    import numpy as np

    cu_map = {
        "all":                ct.ComputeUnit.ALL,
        "cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
        "cpuAndGPU":          ct.ComputeUnit.CPU_AND_GPU,
        "cpuOnly":            ct.ComputeUnit.CPU_ONLY,
    }
    cu = cu_map.get(compute_units, ct.ComputeUnit.ALL)
    print(f"Loading {model_path} with compute_units={compute_units} ...")
    model = ct.models.MLModel(str(model_path), compute_units=cu)

    INPUT_CHANNELS, SEQ = 320, 256
    dummy = np.random.randn(1, INPUT_CHANNELS, 1, SEQ).astype(np.float32)
    feed  = {"x_ane": dummy}

    # Warmup
    print("Warming up (20 passes)...")
    for _ in range(20):
        model.predict(feed)

    # Timed passes
    print(f"Benchmarking {n_passes} passes...")
    times = []
    for _ in range(n_passes):
        t0 = time.perf_counter()
        model.predict(feed)
        times.append((time.perf_counter() - t0) * 1000)   # ms

    times.sort()
    trim = int(len(times) * 0.1)
    trimmed = times[trim:-trim] if trim > 0 else times

    mean_ms   = sum(trimmed) / len(trimmed)
    median_ms = trimmed[len(trimmed)//2]
    p90_ms    = trimmed[int(len(trimmed)*0.9)]
    passes_s  = 1000.0 / mean_ms
    bytes_s   = passes_s * SEQ

    SEQ_BYTES = SEQ   # 256 bytes per pass
    throughput_mbs = passes_s * SEQ_BYTES / 1e6

    print()
    print("─" * 50)
    print(f"  Compute units  : {compute_units}")
    print(f"  Passes         : {n_passes}  (trimmed {trim*2})")
    print(f"  Mean           : {mean_ms:.2f} ms/pass")
    print(f"  Median         : {median_ms:.2f} ms/pass")
    print(f"  P90            : {p90_ms:.2f} ms/pass")
    print(f"  Passes/s       : {passes_s:.1f}")
    print(f"  Throughput     : {throughput_mbs:.1f} MB/s")
    print("─" * 50)
    print()
    print("ByteGrid-44M (ANE private runtime, hand-written MIL):")
    print("  Mean : 11.21 ms/pass  |  Passes/s: 89.2  |  Throughput: 22.8 MB/s")
    print()

    ratio = mean_ms / 11.21
    if ratio >= 1:
        print(f"  Transformer is {ratio:.1f}× SLOWER than ByteGrid on this hardware.")
    else:
        print(f"  Transformer is {1/ratio:.1f}× FASTER than ByteGrid on this hardware.")

    import json
    result = {
        "model": str(model_path),
        "compute_units": compute_units,
        "n_passes": n_passes,
        "mean_ms": round(mean_ms, 3),
        "median_ms": round(median_ms, 3),
        "p90_ms": round(p90_ms, 3),
        "passes_per_s": round(passes_s, 2),
        "throughput_mb_s": round(throughput_mbs, 2),
        "bytegrid_mean_ms": 11.21,
        "speedup_bytegrid_over_transformer": round(mean_ms / 11.21, 2),
    }
    out_json = model_path.parent / "baseline_benchmark.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"Results saved: {out_json}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Export + benchmark transformer baseline")
    sub = parser.add_subparsers(dest="cmd")

    p_export = sub.add_parser("export", help="Export PyTorch model to CoreML")
    p_export.add_argument("--out", default="build/baseline_transformer.mlpackage")

    p_bench = sub.add_parser("bench", help="Benchmark CoreML model throughput")
    p_bench.add_argument("--model", default="build/baseline_transformer.mlpackage")
    p_bench.add_argument("--n", type=int, default=200, help="Number of timed passes")
    p_bench.add_argument(
        "--compute-units",
        default="cpuAndNeuralEngine",
        choices=["all", "cpuAndNeuralEngine", "cpuAndGPU", "cpuOnly"],
        help="CoreML compute unit routing",
    )

    args = parser.parse_args()

    if args.cmd == "export":
        export(Path(args.out))
    elif args.cmd == "bench":
        benchmark(Path(args.model), args.n, args.compute_units)
    else:
        # Run both by default
        out = Path("build/baseline_transformer.mlpackage")
        export(out)
        print()
        for cu in ["cpuAndNeuralEngine", "cpuAndGPU"]:
            benchmark(out, 200, cu)
            print()


if __name__ == "__main__":
    main()
