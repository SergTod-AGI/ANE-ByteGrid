#!/usr/bin/env python3
import argparse
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SPLIT_RE = re.compile(
    r"split loss: train_ce=([0-9.]+) eval_ce=([0-9.]+) gap_eval_minus_train=([+-][0-9.]+)"
)
EVAL_RE = re.compile(
    r"ANE staged evaluate succeeded: .* avg=([0-9.]+) ms/pass .*passes/s=([0-9.]+)(?: .*tokens/s=([0-9.]+))?"
)
GUARD_RE = re.compile(r"guardrails: .* stable=(YES|NO)")

PRESETS = {
    "quick": {"demo_seeds": 10, "search_seeds": 4, "gate_seeds": 10, "batches": 2},
    "golden": {"demo_seeds": 20, "search_seeds": 6, "gate_seeds": 20, "batches": 2},
    "stability": {"demo_seeds": 20, "search_seeds": 6, "gate_seeds": 20, "batches": 2},
}


@dataclass
class Candidate:
    name: str
    env: Dict[str, str]


def parse_train_output(text: str) -> Optional[Dict]:
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


def _trim_by_latency(ok_rows: List[Dict], trim_frac: float) -> List[Dict]:
    if trim_frac <= 0.0 or len(ok_rows) < 3:
        return ok_rows
    trim_each = int(len(ok_rows) * trim_frac)
    if trim_each <= 0:
        return ok_rows
    if (trim_each * 2) >= len(ok_rows):
        return ok_rows
    ordered = sorted(ok_rows, key=lambda r: float(r["avg_ms"]))
    return ordered[trim_each : len(ordered) - trim_each]


def summarize_rows(rows: List[Dict], latency_trim_frac: float = 0.0) -> Dict:
    ok_rows = [r for r in rows if r.get("ok")]
    latency_rows = _trim_by_latency(ok_rows, latency_trim_frac)
    summary = {
        "n": len(rows),
        "ok_n": len(ok_rows),
        "stable_yes": sum(1 for r in ok_rows if r.get("stable")),
        "latency_trim_frac": latency_trim_frac,
        "latency_rows_n": len(latency_rows),
        "latency_trimmed_n": max(0, len(ok_rows) - len(latency_rows)),
    }
    if not ok_rows:
        summary.update(
            {
                "eval_ce_mean": math.nan,
                "eval_ce_sd": math.nan,
                "gap_mean": math.nan,
                "avg_ms_mean": math.nan,
                "avg_ms_sd": math.nan,
                "passes_per_s_mean": math.nan,
                "tokens_per_s_mean": math.nan,
            }
        )
        return summary

    evals = [r["eval_ce"] for r in ok_rows]
    gaps = [r["gap"] for r in ok_rows]
    ms = [r["avg_ms"] for r in latency_rows] if latency_rows else [r["avg_ms"] for r in ok_rows]
    pps = [r["passes_per_s"] for r in latency_rows] if latency_rows else [r["passes_per_s"] for r in ok_rows]
    tps_src = latency_rows if latency_rows else ok_rows
    tps = [r["tokens_per_s"] for r in tps_src if r.get("tokens_per_s") is not None]
    summary.update(
        {
            "eval_ce_mean": statistics.fmean(evals),
            "eval_ce_sd": statistics.stdev(evals) if len(evals) > 1 else 0.0,
            "gap_mean": statistics.fmean(gaps),
            "avg_ms_mean": statistics.fmean(ms),
            "avg_ms_sd": statistics.stdev(ms) if len(ms) > 1 else 0.0,
            "passes_per_s_mean": statistics.fmean(pps),
            "tokens_per_s_mean": statistics.fmean(tps) if tps else math.nan,
        }
    )
    return summary


def _command_output(args: List[str]) -> Optional[str]:
    try:
        result = subprocess.run(args, text=True, capture_output=True, check=True, timeout=3)
        return result.stdout.strip()
    except Exception:
        return None


def _detect_hardware_tag() -> str:
    machine = platform.machine()
    mac_ver = platform.mac_ver()[0] or "unknown"
    cpu = _command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
    if cpu:
        return f"{machine}|macOS-{mac_ver}|{cpu}"
    return f"{machine}|macOS-{mac_ver}"


def _detect_commitish(root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            text=True,
            capture_output=True,
            timeout=3,
            check=True,
        )
        out = result.stdout.strip()
        return out if out else None
    except Exception:
        return None


def _default_candidates() -> List[Candidate]:
    candidates = []
    for value in [0.99, 0.985, 0.98, 0.975, 0.97, 0.965, 0.96]:
        candidates.append(Candidate(f"head_logit_{value:.3f}", {"ANE_BG_HEAD_LOGIT_SCALE": f"{value:.3f}"}))
    for value in [0.99, 0.98]:
        candidates.append(Candidate(f"head_weight_{value:.3f}", {"ANE_BG_HEAD_WEIGHT_SCALE": f"{value:.3f}"}))
    for weight, logit in [(0.99, 0.99), (0.99, 0.98), (0.98, 0.98), (0.98, 0.97)]:
        candidates.append(
            Candidate(
                f"head_hybrid_w{weight:.3f}_l{logit:.3f}",
                {
                    "ANE_BG_HEAD_WEIGHT_SCALE": f"{weight:.3f}",
                    "ANE_BG_HEAD_LOGIT_SCALE": f"{logit:.3f}",
                },
            )
        )
    candidates.append(
        Candidate(
            "residual_1.15_1.42_0.03",
            {
                "ANE_BG_ALPHA_LOCAL_MUL": "1.15",
                "ANE_BG_ALPHA_GLOBAL_MUL": "1.42",
                "ANE_BG_ALPHA_MLP_MUL": "0.03",
            },
        )
    )
    return candidates


def _stability_candidates() -> List[Candidate]:
    candidates: List[Candidate] = []
    for weight in [0.98, 0.985, 0.99]:
        for logit in [0.96, 0.965, 0.97, 0.975]:
            candidates.append(
                Candidate(
                    f"head_hybrid_w{weight:.3f}_l{logit:.3f}",
                    {
                        "ANE_BG_HEAD_WEIGHT_SCALE": f"{weight:.3f}",
                        "ANE_BG_HEAD_LOGIT_SCALE": f"{logit:.3f}",
                    },
                )
            )
    return candidates


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = q * (len(ordered) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _latency_diagnostics(rows: List[Dict], baseline_mean: Optional[float] = None, baseline_sd: Optional[float] = None) -> Dict:
    ok_rows = [r for r in rows if r.get("ok")]
    ms = [float(r["avg_ms"]) for r in ok_rows if "avg_ms" in r]
    if not ms:
        return {
            "n_ok": 0,
            "avg_ms_p50": math.nan,
            "avg_ms_p90": math.nan,
            "slow_outlier_threshold_ms": math.nan,
            "slow_outlier_count": 0,
        }
    mean = statistics.fmean(ms)
    sd = statistics.stdev(ms) if len(ms) > 1 else 0.0
    ref_mean = baseline_mean if baseline_mean is not None else mean
    ref_sd = baseline_sd if baseline_sd is not None else sd
    threshold = ref_mean + (2.0 * ref_sd)
    outliers = sum(1 for v in ms if v > threshold)
    return {
        "n_ok": len(ms),
        "avg_ms_p50": _percentile(ms, 0.5),
        "avg_ms_p90": _percentile(ms, 0.9),
        "slow_outlier_threshold_ms": threshold,
        "slow_outlier_count": outliers,
    }


def _run_train(
    root: Path,
    train_bin: Path,
    train_seed: int,
    eval_seed: int,
    env_overrides: Dict[str, str],
    warmup: int,
    iters: int,
    timeout_s: int,
) -> Dict:
    env = os.environ.copy()
    env.update(
        {
            "ANE_BG_LOG_SURFACE_STATS": "0",
            "ANE_BG_EXPERIMENT_LOG": "0",
            "ANE_BG_WARMUP": str(warmup),
            "ANE_BG_ITERS": str(iters),
            "ANE_BG_TRAIN_SEED": str(train_seed),
            "ANE_BG_EVAL_SEED": str(eval_seed),
        }
    )
    env.update(env_overrides)
    proc = subprocess.run(
        [str(train_bin)],
        cwd=str(root),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    parsed = parse_train_output(text)
    row = {
        "ok": proc.returncode == 0 and parsed is not None,
        "returncode": proc.returncode,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
    }
    if parsed is None:
        row["stderr_tail"] = text[-2500:]
        return row
    row.update(parsed)
    return row


def _aggregate_repeated_seed_rows(seed: int, eval_seed: int, rows: List[Dict], aggregate: str) -> Dict:
    aggregate = aggregate.lower()
    if not rows:
        return {
            "ok": False,
            "train_seed": seed,
            "eval_seed": eval_seed,
            "repeat_n": 0,
            "repeat_agg": aggregate,
            "stderr_tail": "no_repeat_rows",
        }
    if len(rows) == 1:
        row = dict(rows[0])
        row["repeat_n"] = 1
        row["repeat_agg"] = aggregate
        row["repeat_rows"] = rows
        return row

    ok_rows = [r for r in rows if r.get("ok")]
    out = {
        "ok": len(ok_rows) == len(rows),
        "train_seed": seed,
        "eval_seed": eval_seed,
        "repeat_n": len(rows),
        "repeat_agg": aggregate,
        "repeat_rows": rows,
    }
    if len(ok_rows) != len(rows):
        out["returncode"] = next((r.get("returncode", 1) for r in rows if not r.get("ok")), 1)
        out["stderr_tail"] = next((r.get("stderr_tail", "") for r in rows if not r.get("ok")), "")
        return out

    f = statistics.fmean if aggregate == "mean" else statistics.median
    out.update(
        {
            "returncode": 0,
            "train_ce": float(f([float(r["train_ce"]) for r in ok_rows])),
            "eval_ce": float(f([float(r["eval_ce"]) for r in ok_rows])),
            "gap": float(f([float(r["gap"]) for r in ok_rows])),
            "avg_ms": float(f([float(r["avg_ms"]) for r in ok_rows])),
            "passes_per_s": float(f([float(r["passes_per_s"]) for r in ok_rows])),
            "stable": all(bool(r.get("stable")) for r in ok_rows),
            "avg_ms_samples": [float(r["avg_ms"]) for r in ok_rows],
        }
    )
    tokens = [float(r["tokens_per_s"]) for r in ok_rows if r.get("tokens_per_s") is not None]
    out["tokens_per_s"] = float(f(tokens)) if tokens else None
    return out


def _run_seed_set(
    root: Path,
    train_bin: Path,
    env_overrides: Dict[str, str],
    seeds: List[int],
    eval_seed_offset: int,
    warmup: int,
    iters: int,
    timeout_s: int,
    seed_repeats: int = 1,
    repeat_aggregate: str = "mean",
    cooldown_ms: int = 0,
    latency_trim_frac: float = 0.0,
) -> Tuple[List[Dict], Dict]:
    rows = []
    for seed in seeds:
        eval_seed = eval_seed_offset + seed
        repeated_rows = []
        for rep in range(max(1, seed_repeats)):
            repeated_rows.append(
                _run_train(
                    root=root,
                    train_bin=train_bin,
                    train_seed=seed,
                    eval_seed=eval_seed,
                    env_overrides=env_overrides,
                    warmup=warmup,
                    iters=iters,
                    timeout_s=timeout_s,
                )
            )
            if cooldown_ms > 0 and rep < (max(1, seed_repeats) - 1):
                time.sleep(float(cooldown_ms) / 1000.0)
        rows.append(
            _aggregate_repeated_seed_rows(
                seed=seed,
                eval_seed=eval_seed,
                rows=repeated_rows,
                aggregate=repeat_aggregate,
            )
        )
        if cooldown_ms > 0:
            time.sleep(float(cooldown_ms) / 1000.0)
    return rows, summarize_rows(rows, latency_trim_frac=latency_trim_frac)


def _candidate_hard_filter(summary: Dict, baseline_summary: Dict, variance_multiplier: float) -> List[str]:
    reasons = []
    if summary["ok_n"] != summary["n"]:
        reasons.append("non_ok_runs")
    if summary["stable_yes"] != summary["ok_n"]:
        reasons.append("unstable_runs")
    if summary["avg_ms_mean"] > baseline_summary["avg_ms_mean"]:
        reasons.append("avg_ms_mean_regression")
    if summary["avg_ms_sd"] > baseline_summary["avg_ms_sd"] * variance_multiplier:
        reasons.append("avg_ms_sd_regression")
    return reasons


def _run_progression(
    root: Path,
    train_bin: Path,
    steps: int,
    progression_seeds: int,
    progression_start: float,
    progression_delta: float,
    latency_variance_multiplier: float,
    warmup: int,
    iters: int,
    timeout_s: int,
    seed_repeats: int,
    repeat_aggregate: str,
    cooldown_ms: int,
    latency_trim_frac: float,
) -> Dict:
    seed_list = list(range(progression_seeds))
    checkpoints = []
    for step in range(steps + 1):
        value = progression_start + (progression_delta * step)
        env = {"ANE_BG_HEAD_LOGIT_SCALE": f"{value:.3f}"}
        rows, summary = _run_seed_set(
            root=root,
            train_bin=train_bin,
            env_overrides=env,
            seeds=seed_list,
            eval_seed_offset=9000,
            warmup=warmup,
            iters=iters,
            timeout_s=timeout_s,
            seed_repeats=seed_repeats,
            repeat_aggregate=repeat_aggregate,
            cooldown_ms=cooldown_ms,
            latency_trim_frac=latency_trim_frac,
        )
        checkpoints.append(
            {
                "step": step,
                "head_logit_scale": value,
                "summary": summary,
                "rows": rows,
            }
        )

    base = checkpoints[0]["summary"]
    final = checkpoints[-1]["summary"]
    eval_curve = [c["summary"]["eval_ce_mean"] for c in checkpoints]
    monotonic_non_increasing = all(eval_curve[i] <= eval_curve[i - 1] for i in range(1, len(eval_curve)))
    net_quality_gain = final["eval_ce_mean"] < base["eval_ce_mean"]
    no_latency_regression = final["avg_ms_mean"] <= base["avg_ms_mean"]
    variance_ok = final["avg_ms_sd"] <= base["avg_ms_sd"] * latency_variance_multiplier
    all_stable = all(
        c["summary"]["ok_n"] == c["summary"]["n"] and c["summary"]["stable_yes"] == c["summary"]["ok_n"]
        for c in checkpoints
    )
    progression_success = (
        all_stable
        and no_latency_regression
        and variance_ok
        and (monotonic_non_increasing or net_quality_gain)
    )
    return {
        "config": {
            "steps": steps,
            "progression_seeds": progression_seeds,
            "head_logit_start": progression_start,
            "head_logit_delta": progression_delta,
            "latency_variance_multiplier": latency_variance_multiplier,
        },
        "checkpoints": checkpoints,
        "criteria": {
            "monotonic_non_increasing_eval_ce": monotonic_non_increasing,
            "net_quality_gain": net_quality_gain,
            "no_latency_mean_regression": no_latency_regression,
            "latency_variance_within_policy": variance_ok,
            "all_stable_ok": all_stable,
            "progression_success": progression_success,
        },
    }


def _validate_required_sections(result: Dict) -> Dict:
    required = ["metadata", "demo_criterion", "search", "gate", "progression"]
    missing = [key for key in required if key not in result]
    return {
        "required_sections": required,
        "missing_sections": missing,
        "ok": len(missing) == 0,
    }


def _self_test() -> int:
    sample = (
        "guardrails: hidden_naninf=0 hidden_sat=0.000000 hidden_var=0.096769 output_naninf=0 output_sat=0.000000 "
        "output_var=1.334426 stable=YES\n"
        "split loss: train_ce=6.1440 eval_ce=6.2039 gap_eval_minus_train=+0.0599 train_ppx=465.89 eval_ppx=494.67\n"
        "ANE staged evaluate succeeded: warmup=1 iters=1 avg=11.994 ms/pass throughput=1065.66 MB/s "
        "passes/s=83.38 tokens/s=21344 est_conv=9.752 GMAC/pass est_tflops=1.626 kernels=26\n"
    )
    parsed = parse_train_output(sample)
    if parsed is None:
        print("self-test failed: parse_train_output returned None")
        return 1
    if not parsed["stable"]:
        print("self-test failed: expected stable=YES")
        return 1
    if abs(parsed["avg_ms"] - 11.994) > 1e-6:
        print("self-test failed: avg_ms mismatch")
        return 1
    print("phase3_protocol self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3/4 protocol: baseline criterion + constrained search + gate + progression"
    )
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--train-bin", default="build/train", help="Path to train binary")
    parser.add_argument("--out", default="build/phase3_protocol.json", help="Output JSON path")
    parser.add_argument("--summary-out", default=None, help="Compact summary JSON path")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="quick")
    parser.add_argument("--demo-seeds", type=int, default=None)
    parser.add_argument("--search-seeds", type=int, default=None)
    parser.add_argument("--gate-seeds", type=int, default=None)
    parser.add_argument("--batches", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--seed-repeats", type=int, default=None, help="Repeated runs per seed")
    parser.add_argument(
        "--repeat-aggregate",
        choices=["mean", "median"],
        default=None,
        help="Aggregate repeated per-seed runs",
    )
    parser.add_argument(
        "--latency-trim-frac",
        type=float,
        default=0.0,
        help="Trim fraction applied to latency-derived summary metrics",
    )
    parser.add_argument("--cooldown-ms", type=int, default=None, help="Sleep between repeated runs/seeds")
    parser.add_argument(
        "--stability-qualification-runs",
        type=int,
        default=None,
        help="Number of baseline demo reruns required to pass before search/gate",
    )
    parser.add_argument("--search-speed-weight", type=float, default=0.01)
    parser.add_argument("--search-var-weight", type=float, default=0.05)
    parser.add_argument("--min-eval-improve", type=float, default=0.01)
    parser.add_argument("--variance-multiplier", type=float, default=1.2)
    parser.add_argument("--demo-quality-max", type=float, default=6.21)
    parser.add_argument("--demo-latency-max", type=float, default=12.50)
    parser.add_argument("--demo-latency-sd-max", type=float, default=None)
    parser.add_argument("--progression-steps", type=int, default=6)
    parser.add_argument("--progression-seeds", type=int, default=8)
    parser.add_argument("--progression-start", type=float, default=1.00)
    parser.add_argument("--progression-delta", type=float, default=-0.01)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    preset = PRESETS[args.preset]
    demo_seeds = args.demo_seeds if args.demo_seeds is not None else preset["demo_seeds"]
    search_seeds = args.search_seeds if args.search_seeds is not None else preset["search_seeds"]
    gate_seeds = args.gate_seeds if args.gate_seeds is not None else preset["gate_seeds"]
    batches = args.batches if args.batches is not None else preset["batches"]
    seed_repeats = args.seed_repeats if args.seed_repeats is not None else (2 if args.preset == "stability" else 1)
    repeat_aggregate = args.repeat_aggregate if args.repeat_aggregate is not None else ("median" if args.preset == "stability" else "mean")
    cooldown_ms = args.cooldown_ms if args.cooldown_ms is not None else (250 if args.preset == "stability" else 0)
    qualification_runs = (
        args.stability_qualification_runs
        if args.stability_qualification_runs is not None
        else (2 if args.preset == "stability" else 0)
    )

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

    if not train_bin.exists():
        print(f"missing train binary: {train_bin}", file=sys.stderr)
        return 2

    demo_latency_sd_max = args.demo_latency_sd_max
    if demo_latency_sd_max is None:
        demo_latency_sd_max = 0.80 if args.preset == "stability" else 1.00

    metadata = {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "preset": args.preset,
        "hardware_tag": _detect_hardware_tag(),
        "commitish": _detect_commitish(root),
        "execution_hygiene": {
            "seed_repeats": seed_repeats,
            "repeat_aggregate": repeat_aggregate,
            "cooldown_ms": cooldown_ms,
            "latency_trim_frac": args.latency_trim_frac,
            "stability_qualification_runs": qualification_runs,
        },
    }

    baseline = Candidate("baseline", {})
    candidates = _stability_candidates() if args.preset == "stability" else _default_candidates()
    demo_seed_list = list(range(demo_seeds))
    search_seed_list = list(range(search_seeds))
    gate_seed_list = list(range(gate_seeds))

    baseline_demo_rows, baseline_demo_summary = _run_seed_set(
        root=root,
        train_bin=train_bin,
        env_overrides=baseline.env,
        seeds=demo_seed_list,
        eval_seed_offset=1000,
        warmup=args.warmup,
        iters=args.iters,
        timeout_s=args.timeout_s,
        seed_repeats=seed_repeats,
        repeat_aggregate=repeat_aggregate,
        cooldown_ms=cooldown_ms,
        latency_trim_frac=args.latency_trim_frac,
    )
    demo_criteria = {
        "quality_target_eval_ce_mean_max": args.demo_quality_max,
        "latency_target_avg_ms_mean_max": args.demo_latency_max,
        "latency_target_avg_ms_sd_max": demo_latency_sd_max,
        "all_stable_required": True,
    }
    def _demo_pass(summary: Dict) -> bool:
        return (
            summary["ok_n"] == summary["n"]
            and summary["stable_yes"] == summary["ok_n"]
            and summary["eval_ce_mean"] <= args.demo_quality_max
            and summary["avg_ms_mean"] <= args.demo_latency_max
            and summary["avg_ms_sd"] <= demo_latency_sd_max
        )

    demo_pass = _demo_pass(baseline_demo_summary)
    qualification = {"required_runs": qualification_runs, "ran": False, "all_passed": True, "runs": []}
    if qualification_runs > 0:
        qualification["ran"] = True
        qualification["runs"].append(
            {"run_idx": 0, "summary": baseline_demo_summary, "passed": demo_pass, "rows": baseline_demo_rows}
        )
        for run_idx in range(1, qualification_runs):
            q_rows, q_summary = _run_seed_set(
                root=root,
                train_bin=train_bin,
                env_overrides=baseline.env,
                seeds=demo_seed_list,
                eval_seed_offset=1000,
                warmup=args.warmup,
                iters=args.iters,
                timeout_s=args.timeout_s,
                seed_repeats=seed_repeats,
                repeat_aggregate=repeat_aggregate,
                cooldown_ms=cooldown_ms,
                latency_trim_frac=args.latency_trim_frac,
            )
            q_pass = _demo_pass(q_summary)
            qualification["runs"].append(
                {"run_idx": run_idx, "summary": q_summary, "passed": q_pass, "rows": q_rows}
            )
        qualification["all_passed"] = all(r["passed"] for r in qualification["runs"])
        demo_pass = qualification["all_passed"]

    qualification_failed = qualification["ran"] and not qualification["all_passed"]
    if qualification_failed:
        baseline_search_rows = []
        baseline_search_summary = summarize_rows([], latency_trim_frac=args.latency_trim_frac)
    else:
        baseline_search_rows, baseline_search_summary = _run_seed_set(
            root=root,
            train_bin=train_bin,
            env_overrides=baseline.env,
            seeds=search_seed_list,
            eval_seed_offset=3000,
            warmup=args.warmup,
            iters=args.iters,
            timeout_s=args.timeout_s,
            seed_repeats=seed_repeats,
            repeat_aggregate=repeat_aggregate,
            cooldown_ms=cooldown_ms,
            latency_trim_frac=args.latency_trim_frac,
        )

    search_results = []
    survivors = []
    if not qualification_failed:
        for candidate in candidates:
            rows, summary = _run_seed_set(
                root=root,
                train_bin=train_bin,
                env_overrides=candidate.env,
                seeds=search_seed_list,
                eval_seed_offset=3000,
                warmup=args.warmup,
                iters=args.iters,
                timeout_s=args.timeout_s,
                seed_repeats=seed_repeats,
                repeat_aggregate=repeat_aggregate,
                cooldown_ms=cooldown_ms,
                latency_trim_frac=args.latency_trim_frac,
            )
            rejection_reasons = _candidate_hard_filter(
                summary=summary,
                baseline_summary=baseline_search_summary,
                variance_multiplier=args.variance_multiplier,
            )
            passes_filter = len(rejection_reasons) == 0
            score = (
                summary["eval_ce_mean"]
                + (args.search_speed_weight * summary["avg_ms_mean"])
                + (args.search_var_weight * summary["avg_ms_sd"])
                if passes_filter
                else math.inf
            )
            item = {
                "candidate": candidate.name,
                "env": candidate.env,
                "summary": summary,
                "passes_hard_filter": passes_filter,
                "rejection_reasons": rejection_reasons,
                "score": score,
                "rows": rows,
            }
            search_results.append(item)
            if passes_filter:
                survivors.append(item)
    search_results.sort(key=lambda item: item["score"])
    survivors.sort(key=lambda item: item["score"])
    top = survivors[0] if survivors else None

    gate = {"ran": False}
    if qualification_failed:
        gate = {
            "ran": False,
            "reason": "stability_qualification_failed",
            "stability_delta": {
                "avg_ms_mean_delta": math.nan,
                "avg_ms_sd_delta": math.nan,
                "passes_per_s_mean_delta": math.nan,
                "eval_ce_mean_delta": math.nan,
            },
            "hard_filter_policy": [
                "avg_ms_mean must not regress vs baseline",
                f"avg_ms_sd must be <= baseline * {args.variance_multiplier}",
                "all runs must succeed and be stable",
            ],
        }
    elif top:
        batch_size = max(1, gate_seeds // max(1, batches))
        groups = []
        for idx in range(batches):
            start = idx * batch_size
            end = min(gate_seeds, (idx + 1) * batch_size)
            if start < end:
                groups.append(gate_seed_list[start:end])
        if not groups:
            groups = [gate_seed_list]

        base_rows = []
        cand_rows = []
        base_batches = []
        cand_batches = []
        for group in groups:
            b_rows, b_summary = _run_seed_set(
                root=root,
                train_bin=train_bin,
                env_overrides=baseline.env,
                seeds=group,
                eval_seed_offset=5000,
                warmup=args.warmup,
                iters=args.iters,
                timeout_s=args.timeout_s,
                seed_repeats=seed_repeats,
                repeat_aggregate=repeat_aggregate,
                cooldown_ms=cooldown_ms,
                latency_trim_frac=args.latency_trim_frac,
            )
            c_rows, c_summary = _run_seed_set(
                root=root,
                train_bin=train_bin,
                env_overrides=top["env"],
                seeds=group,
                eval_seed_offset=5000,
                warmup=args.warmup,
                iters=args.iters,
                timeout_s=args.timeout_s,
                seed_repeats=seed_repeats,
                repeat_aggregate=repeat_aggregate,
                cooldown_ms=cooldown_ms,
                latency_trim_frac=args.latency_trim_frac,
            )
            base_rows.extend(b_rows)
            cand_rows.extend(c_rows)
            base_batches.append(b_summary)
            cand_batches.append(c_summary)

        base_summary = summarize_rows(base_rows, latency_trim_frac=args.latency_trim_frac)
        cand_summary = summarize_rows(cand_rows, latency_trim_frac=args.latency_trim_frac)
        eval_improve = base_summary["eval_ce_mean"] - cand_summary["eval_ce_mean"]
        no_avg_ms_regress = cand_summary["avg_ms_mean"] <= base_summary["avg_ms_mean"]
        var_eval_ok = cand_summary["eval_ce_sd"] <= base_summary["eval_ce_sd"] * args.variance_multiplier
        var_ms_ok = cand_summary["avg_ms_sd"] <= base_summary["avg_ms_sd"] * args.variance_multiplier
        all_stable_ok = (
            base_summary["ok_n"] == base_summary["n"]
            and cand_summary["ok_n"] == cand_summary["n"]
            and base_summary["stable_yes"] == base_summary["ok_n"]
            and cand_summary["stable_yes"] == cand_summary["ok_n"]
        )
        repeated_batch_eval_consistent = all(
            c["eval_ce_mean"] < b["eval_ce_mean"] for b, c in zip(base_batches, cand_batches)
        )
        promote = all(
            [
                eval_improve >= args.min_eval_improve,
                no_avg_ms_regress,
                var_eval_ok,
                var_ms_ok,
                all_stable_ok,
                repeated_batch_eval_consistent,
            ]
        )
        gate = {
            "ran": True,
            "top_candidate": top["candidate"],
            "top_candidate_env": top["env"],
            "summary": {"baseline": base_summary, "candidate": cand_summary},
            "stability_delta": {
                "avg_ms_mean_delta": cand_summary["avg_ms_mean"] - base_summary["avg_ms_mean"],
                "avg_ms_sd_delta": cand_summary["avg_ms_sd"] - base_summary["avg_ms_sd"],
                "passes_per_s_mean_delta": cand_summary["passes_per_s_mean"] - base_summary["passes_per_s_mean"],
                "eval_ce_mean_delta": cand_summary["eval_ce_mean"] - base_summary["eval_ce_mean"],
            },
            "batches": {"baseline": base_batches, "candidate": cand_batches},
            "criteria": {
                "min_eval_ce_improvement_required": args.min_eval_improve,
                "observed_eval_ce_improvement": eval_improve,
                "min_margin_ok": eval_improve >= args.min_eval_improve,
                "no_avg_ms_mean_regression": no_avg_ms_regress,
                "similar_variance_policy": f"candidate_sd <= baseline_sd * {args.variance_multiplier} for eval_ce and avg_ms",
                "similar_variance_eval_ok": var_eval_ok,
                "similar_variance_avg_ms_ok": var_ms_ok,
                "all_stable_ok": all_stable_ok,
                "repeated_batch_policy": "candidate eval_ce_mean must beat baseline in each batch",
                "repeated_batch_eval_consistent": repeated_batch_eval_consistent,
                "promote_defaults": promote,
            },
            "rows": {"baseline": base_rows, "candidate": cand_rows},
        }
    else:
        gate = {
            "ran": False,
            "reason": "no_candidate_survived_hard_filter",
            "stability_delta": {
                "avg_ms_mean_delta": math.nan,
                "avg_ms_sd_delta": math.nan,
                "passes_per_s_mean_delta": math.nan,
                "eval_ce_mean_delta": math.nan,
            },
            "hard_filter_policy": [
                "avg_ms_mean must not regress vs baseline",
                f"avg_ms_sd must be <= baseline * {args.variance_multiplier}",
                "all runs must succeed and be stable",
            ],
        }

    if qualification_failed:
        progression = {
            "skipped": True,
            "reason": "stability_qualification_failed",
            "criteria": {"progression_success": False},
        }
    else:
        progression = _run_progression(
            root=root,
            train_bin=train_bin,
            steps=args.progression_steps,
            progression_seeds=args.progression_seeds,
            progression_start=args.progression_start,
            progression_delta=args.progression_delta,
            latency_variance_multiplier=args.variance_multiplier,
            warmup=args.warmup,
            iters=args.iters,
            timeout_s=args.timeout_s,
            seed_repeats=seed_repeats,
            repeat_aggregate=repeat_aggregate,
            cooldown_ms=cooldown_ms,
            latency_trim_frac=args.latency_trim_frac,
        )

    result = {
        "metadata": metadata,
        "protocol": {
            "description": "Phase 6 determinism/stability protocol: baseline + qualification gate -> constrained search -> repeated-batch gate -> progression run",
            "version": 4,
        },
        "demo_criterion": {
            "criteria": demo_criteria,
            "baseline_summary": baseline_demo_summary,
            "passed": demo_pass,
            "rows": baseline_demo_rows,
            "stability_qualification": qualification,
        },
        "search": {
            "baseline_summary": baseline_search_summary,
            "baseline_rows": baseline_search_rows,
            "candidates_ranked": search_results,
            "survivor_count": len(survivors),
            "top_candidate": top["candidate"] if top else None,
            "skipped_due_to_qualification_failure": qualification_failed,
        },
        "gate": gate,
        "progression": progression,
        "diagnostics": {
            "demo_baseline": _latency_diagnostics(
                baseline_demo_rows,
                baseline_mean=baseline_demo_summary["avg_ms_mean"],
                baseline_sd=baseline_demo_summary["avg_ms_sd"],
            ),
            "gate_baseline": _latency_diagnostics(gate.get("rows", {}).get("baseline", [])),
            "gate_candidate": _latency_diagnostics(
                gate.get("rows", {}).get("candidate", []),
                baseline_mean=gate.get("summary", {}).get("baseline", {}).get("avg_ms_mean"),
                baseline_sd=gate.get("summary", {}).get("baseline", {}).get("avg_ms_sd"),
            ),
        },
    }
    result["validation"] = _validate_required_sections(result)

    compact = {
        "metadata": metadata,
        "demo_pass": result["demo_criterion"]["passed"],
        "qualification_pass": result["demo_criterion"]["stability_qualification"]["all_passed"],
        "demo_summary": result["demo_criterion"]["baseline_summary"],
        "top_candidate": result["search"]["top_candidate"],
        "survivor_count": result["search"]["survivor_count"],
        "promotion_verdict": (
            result["gate"]["criteria"]["promote_defaults"] if result["gate"].get("ran") else False
        ),
        "gate_ran": result["gate"].get("ran", False),
        "progression_success": result["progression"]["criteria"]["progression_success"],
        "validation_ok": result["validation"]["ok"],
    }

    out_path.write_text(json.dumps(result, indent=2) + "\n")
    summary_path.write_text(json.dumps(compact, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
