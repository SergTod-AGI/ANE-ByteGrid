#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def _row(label: str, summary: Dict[str, Any], notes: str) -> str:
    if not summary:
        return f"| {label} | n/a | n/a | n/a | n/a | n/a | n/a | {notes} |"
    return (
        f"| {label} | {summary.get('n','n/a')} | "
        f"{summary.get('stable_yes','n/a')}/{summary.get('ok_n','n/a')} | "
        f"{_fmt(summary.get('eval_ce_mean'))} ± {_fmt(summary.get('eval_ce_sd'))} | "
        f"{_fmt(summary.get('avg_ms_mean'))} ± {_fmt(summary.get('avg_ms_sd'))} | "
        f"{_fmt(summary.get('passes_per_s_mean'),2)} | "
        f"{_fmt(summary.get('tokens_per_s_mean'),1)} | "
        f"{notes} |"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate markdown Phase 4 report table")
    parser.add_argument("--protocol", default="build/phase3_protocol.json")
    parser.add_argument("--baseline-protocol", default="build/phase3_protocol_golden_post_promotion.json")
    parser.add_argument("--benchmark", default="build/ane_gpu_benchmark.json")
    parser.add_argument("--out", default="build/phase4_report.md")
    args = parser.parse_args()

    protocol_path = Path(args.protocol).resolve()
    baseline_protocol_path = Path(args.baseline_protocol).resolve()
    benchmark_path = Path(args.benchmark).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    baseline_protocol = json.loads(baseline_protocol_path.read_text()) if baseline_protocol_path.exists() else {}
    benchmark = json.loads(benchmark_path.read_text()) if benchmark_path.exists() else {}

    demo_summary = baseline_protocol.get("demo_criterion", {}).get(
        "baseline_summary", protocol.get("demo_criterion", {}).get("baseline_summary", {})
    )
    gate = protocol.get("gate", {})
    gate_candidate_summary = gate.get("summary", {}).get("candidate", {}) if gate.get("ran") else {}
    top_candidate = gate.get("top_candidate", protocol.get("search", {}).get("top_candidate", "n/a"))
    ane_ref_summary = benchmark.get("ane", {}).get("summary", {})
    gpu_ref_summary = benchmark.get("gpu", {}).get("summary", {})

    criteria = gate.get("criteria", {})
    failed = []
    if gate.get("ran"):
        for key in [
            "min_margin_ok",
            "no_avg_ms_mean_regression",
            "similar_variance_eval_ok",
            "similar_variance_avg_ms_ok",
            "all_stable_ok",
            "repeated_batch_eval_consistent",
        ]:
            if criteria.get(key) is False:
                failed.append(key)
    promotion_verdict = bool(criteria.get("promote_defaults", False))
    verdict_note = "Promotion Pass" if promotion_verdict else "No further promotion"
    if failed:
        verdict_note += f" ({', '.join(failed)})"

    lines: List[str] = []
    lines.append("# Stability Report")
    lines.append("")
    lines.append(f"- Promotion verdict: **{verdict_note}**")
    lines.append(f"- Top candidate considered: **{top_candidate}**")
    q = protocol.get("demo_criterion", {}).get("stability_qualification", {})
    if q.get("ran"):
        passed_n = sum(1 for run in q.get("runs", []) if run.get("passed"))
        lines.append(
            f"- Stability qualification: **{'pass' if q.get('all_passed') else 'fail'}** "
            f"({passed_n}/{q.get('required_runs', 0)} passed)"
        )
    lines.append("")
    lines.append("| Lane | Runs | Stable | Eval CE (mean ± sd) | Avg ms/pass (mean ± sd) | Passes/s | Tokens/s | Notes |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    lines.append(_row("ANE Baseline (Stability Baseline)", demo_summary, "Pinned to post-promotion baseline artifact"))
    lines.append(_row(f"ANE Best Candidate ({top_candidate})", gate_candidate_summary, "Promotion gate candidate"))
    lines.append(_row("ANE Reference (Benchmark Harness)", ane_ref_summary, "From ane_gpu_benchmark"))
    lines.append(_row("GPU Reference (Benchmark Harness)", gpu_ref_summary, "PyTorch conv micro-harness"))
    delta = gate.get("stability_delta", {})
    if delta:
        lines.append("")
        lines.append("Stability delta (candidate - baseline):")
        lines.append(
            f"- `avg_ms_mean_delta={_fmt(delta.get('avg_ms_mean_delta'))}`, "
            f"`avg_ms_sd_delta={_fmt(delta.get('avg_ms_sd_delta'))}`, "
            f"`passes_per_s_mean_delta={_fmt(delta.get('passes_per_s_mean_delta'))}`, "
            f"`eval_ce_mean_delta={_fmt(delta.get('eval_ce_mean_delta'))}`"
        )
    diag = protocol.get("diagnostics", {})
    if diag:
        lines.append("")
        lines.append("Latency diagnostics:")
        for key in ["demo_baseline", "gate_baseline", "gate_candidate"]:
            d = diag.get(key, {})
            lines.append(
                f"- `{key}`: p50={_fmt(d.get('avg_ms_p50'))} p90={_fmt(d.get('avg_ms_p90'))} "
                f"slow_outliers={d.get('slow_outlier_count','n/a')} "
                f"(threshold={_fmt(d.get('slow_outlier_threshold_ms'))})"
            )
    lines.append("")
    lines.append("Interpretation: compare quality, latency mean, latency variance, and stability together. ")
    lines.append("Do not interpret this table as a raw FLOPS-equivalence claim.")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
