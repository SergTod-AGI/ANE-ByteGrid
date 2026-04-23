#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

SPLIT_RE = re.compile(
    r"split loss: train_ce=([0-9.]+) eval_ce=([0-9.]+) gap_eval_minus_train=([+-][0-9.]+)"
)
EVAL_RE = re.compile(
    r"ANE staged evaluate succeeded: .* avg=([0-9.]+) ms/pass .*passes/s=([0-9.]+)(?: .*tokens/s=([0-9.]+))?"
)
GUARD_RE = re.compile(r"guardrails: .* stable=(YES|NO)")


def parse_ane_output(text: str) -> Optional[Dict]:
    split_m = SPLIT_RE.search(text)
    eval_m = EVAL_RE.search(text)
    guard_m = GUARD_RE.search(text)
    if split_m is None or eval_m is None or guard_m is None:
        return None
    return {
        "train_ce": float(split_m.group(1)),
        "eval_ce": float(split_m.group(2)),
        "gap": float(split_m.group(3)),
        "avg_ms": float(eval_m.group(1)),
        "passes_per_s": float(eval_m.group(2)),
        "tokens_per_s": float(eval_m.group(3)) if eval_m.group(3) else None,
        "stable": guard_m.group(1) == "YES",
    }


def summarize_rows(rows: List[Dict]) -> Dict:
    ok_rows = [r for r in rows if r.get("ok")]
    summary = {
        "n": len(rows),
        "ok_n": len(ok_rows),
        "stable_yes": sum(1 for r in ok_rows if r.get("stable")),
    }
    if not ok_rows:
        summary.update(
            {
                "eval_ce_mean": math.nan,
                "eval_ce_sd": math.nan,
                "avg_ms_mean": math.nan,
                "avg_ms_sd": math.nan,
                "passes_per_s_mean": math.nan,
                "tokens_per_s_mean": math.nan,
                "stability_ratio": 0.0,
            }
        )
        return summary
    evals = [r["eval_ce"] for r in ok_rows]
    ms = [r["avg_ms"] for r in ok_rows]
    pps = [r["passes_per_s"] for r in ok_rows]
    tps = [r["tokens_per_s"] for r in ok_rows if r.get("tokens_per_s") is not None]
    summary.update(
        {
            "eval_ce_mean": statistics.fmean(evals),
            "eval_ce_sd": statistics.stdev(evals) if len(evals) > 1 else 0.0,
            "avg_ms_mean": statistics.fmean(ms),
            "avg_ms_sd": statistics.stdev(ms) if len(ms) > 1 else 0.0,
            "passes_per_s_mean": statistics.fmean(pps),
            "tokens_per_s_mean": statistics.fmean(tps) if tps else math.nan,
            "stability_ratio": (summary["stable_yes"] / summary["ok_n"]) if summary["ok_n"] else 0.0,
        }
    )
    return summary


def run_ane(root: Path, train_bin: Path, seeds: List[int], warmup: int, iters: int, timeout_s: int) -> Dict:
    rows = []
    for seed in seeds:
        env = os.environ.copy()
        env.update(
            {
                "ANE_BG_LOG_SURFACE_STATS": "0",
                "ANE_BG_EXPERIMENT_LOG": "0",
                "ANE_BG_WARMUP": str(warmup),
                "ANE_BG_ITERS": str(iters),
                "ANE_BG_TRAIN_SEED": str(seed),
                "ANE_BG_EVAL_SEED": str(12000 + seed),
            }
        )
        proc = subprocess.run(
            [str(train_bin)],
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        parsed = parse_ane_output(text)
        row = {
            "ok": proc.returncode == 0 and parsed is not None,
            "seed": seed,
            "returncode": proc.returncode,
        }
        if parsed:
            row.update(parsed)
        else:
            row["stderr_tail"] = text[-2000:]
        rows.append(row)
    return {"available": True, "rows": rows, "summary": summarize_rows(rows)}


def _pattern_tokens(seq: int, seed: int) -> List[int]:
    alphabet = b"ane-bytegrid-44m|"
    alen = len(alphabet)
    out = []
    for token in range(seq):
        base = alphabet[(token + (seed % alen)) % alen]
        mix = (token * 29) ^ ((seed * 131) + (seed >> 1))
        out.append((base ^ (mix & 0x3F)) & 0xFF)
    return out


def run_gpu(
    seeds: List[int],
    seq: int,
    gpu_warmup: int,
    gpu_iters: int,
    gpu_device: str,
) -> Dict:
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:
        return {"available": False, "reason": f"torch_unavailable: {exc}"}

    if gpu_device == "cuda" and not torch.cuda.is_available():
        return {"available": False, "reason": "cuda_unavailable"}

    device = torch.device(gpu_device if gpu_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)
    rows = []

    stem_w = torch.randn(512, 320, 1, device=device, dtype=torch.float32) * 0.02
    local_w = torch.randn(512, 512, 1, device=device, dtype=torch.float32) * 0.02
    global_w = torch.randn(512, 512, 1, device=device, dtype=torch.float32) * 0.02
    wv = torch.randn(1024, 512, 1, device=device, dtype=torch.float32) * 0.02
    wg = torch.randn(1024, 512, 1, device=device, dtype=torch.float32) * 0.02
    wo = torch.randn(512, 1024, 1, device=device, dtype=torch.float32) * 0.02
    head = torch.randn(256, 512, 1, device=device, dtype=torch.float32) * 0.02

    def model_forward(x: "torch.Tensor") -> "torch.Tensor":
        h = F.conv1d(x, stem_w)
        for _ in range(24):
            h = h + 0.5 * F.conv1d(h, local_w)
            h = h + 0.5 * F.conv1d(h, global_w)
            a = F.conv1d(h, wv)
            b = F.conv1d(h, wg)
            h = h + 0.05 * F.conv1d(F.silu(a) * torch.sigmoid(b), wo)
        return F.conv1d(h, head)

    for seed in seeds:
        tokens = _pattern_tokens(seq, seed)
        x = torch.zeros(1, 320, seq, device=device, dtype=torch.float32)
        for t, tok in enumerate(tokens):
            x[0, tok, t] = 1.0

        with torch.no_grad():
            for _ in range(gpu_warmup):
                _ = model_forward(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            start = time.perf_counter()
            out = None
            for _ in range(gpu_iters):
                out = model_forward(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

            avg_ms = (elapsed * 1000.0) / max(1, gpu_iters)
            passes = 1000.0 / avg_ms if avg_ms > 0.0 else 0.0
            tokens_per_s = passes * seq
            target = torch.tensor(tokens[1:] + [tokens[-1]], device=device)
            logits = out[0, :, :].transpose(0, 1)
            loss = torch.nn.functional.cross_entropy(logits[:-1, :], target[:-1], reduction="mean")
            stable = torch.isfinite(out).all().item() and torch.isfinite(loss).item()

        rows.append(
            {
                "ok": True,
                "seed": seed,
                "eval_ce": float(loss.item()),
                "avg_ms": float(avg_ms),
                "passes_per_s": float(passes),
                "tokens_per_s": float(tokens_per_s),
                "stable": bool(stable),
            }
        )

    return {
        "available": True,
        "device": str(device),
        "rows": rows,
        "summary": summarize_rows(rows),
        "note": "GPU path is a PyTorch conv micro-harness approximation for reference benchmarking.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ANE vs GPU benchmark with matched summary schema")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--train-bin", default="build/train", help="ANE train binary")
    parser.add_argument("--mode", choices=["ane", "gpu", "both"], default="both")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1, help="ANE warmup")
    parser.add_argument("--iters", type=int, default=3, help="ANE timed iterations")
    parser.add_argument("--gpu-warmup", type=int, default=5)
    parser.add_argument("--gpu-iters", type=int, default=20)
    parser.add_argument("--gpu-device", default="auto", help="auto|cuda|cpu")
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--out", default="build/ane_gpu_benchmark.json")
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    train_bin = (root / args.train_bin).resolve()
    out_path = (root / args.out).resolve()
    summary_path = (
        Path(args.summary_out).resolve()
        if args.summary_out
        else out_path.with_name(out_path.stem + ".summary.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seeds))
    result = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "seeds": args.seeds,
            "schema": "matched_benchmark_v1",
        },
        "framing": "Compare latency stability and throughput characteristics; not intended as raw FLOPS-equivalence claim.",
    }

    if args.mode in ("ane", "both"):
        if not train_bin.exists():
            result["ane"] = {"available": False, "reason": f"missing_train_binary:{train_bin}"}
        else:
            result["ane"] = run_ane(root, train_bin, seeds, args.warmup, args.iters, args.timeout_s)
    if args.mode in ("gpu", "both"):
        result["gpu"] = run_gpu(
            seeds=seeds,
            seq=args.seq,
            gpu_warmup=args.gpu_warmup,
            gpu_iters=args.gpu_iters,
            gpu_device=args.gpu_device,
        )

    compact = {
        "metadata": result["metadata"],
        "ane_summary": result.get("ane", {}).get("summary"),
        "gpu_summary": result.get("gpu", {}).get("summary"),
        "ane_available": result.get("ane", {}).get("available", False),
        "gpu_available": result.get("gpu", {}).get("available", False),
    }

    out_path.write_text(json.dumps(result, indent=2) + "\n")
    summary_path.write_text(json.dumps(compact, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
