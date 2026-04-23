#!/usr/bin/env python3
import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional


def _family_from_candidate(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    if "_l" in name and name.startswith("head_hybrid_"):
        return name.split("_l", 1)[0]
    return name


def _load_run(path: Path) -> Dict:
    data = json.loads(path.read_text())
    demo = data.get("demo_criterion", {})
    qual = demo.get("stability_qualification", {})
    gate = data.get("gate", {})
    criteria = gate.get("criteria", {})
    top = gate.get("top_candidate")
    return {
        "path": str(path),
        "qualification_all_passed": bool(qual.get("all_passed", False)),
        "qualification_passed_n": sum(1 for r in qual.get("runs", []) if r.get("passed")),
        "qualification_required_n": int(qual.get("required_runs", 0)),
        "demo_pass": bool(demo.get("passed", False)),
        "gate_ran": bool(gate.get("ran", False)),
        "top_candidate": top,
        "top_candidate_family": _family_from_candidate(top),
        "promote_defaults": bool(criteria.get("promote_defaults", False)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Phase 6 promotion consistency gate across multiple runs")
    parser.add_argument(
        "--glob",
        default="build/phase6_protocol_determinism_trim10_run*.json",
        help="Glob pattern for protocol artifacts",
    )
    parser.add_argument("--min-runs", type=int, default=4)
    parser.add_argument("--required-qualification-rate", type=float, default=1.0)
    parser.add_argument("--required-promotion-passes", type=int, default=3)
    parser.add_argument("--require-same-family", action="store_true")
    parser.add_argument("--out", default="build/phase6_consistency_gate.json")
    args = parser.parse_args()

    paths = []
    for p in sorted(glob.glob(args.glob)):
        path = Path(p).resolve()
        if path.name.endswith(".summary.json"):
            continue
        paths.append(path)
    if len(paths) < args.min_runs:
        print(f"need at least {args.min_runs} runs, found {len(paths)}")
        return 2

    runs = [_load_run(p) for p in paths]
    total = len(runs)
    qual_passed = sum(1 for r in runs if r["qualification_all_passed"])
    gate_ran = sum(1 for r in runs if r["gate_ran"])
    promo_passed = sum(1 for r in runs if r["promote_defaults"])
    qual_rate = (qual_passed / total) if total else 0.0

    family_counts: Dict[str, int] = {}
    promoted_families: List[str] = []
    for run in runs:
        if run["promote_defaults"] and run["top_candidate_family"]:
            fam = run["top_candidate_family"]
            promoted_families.append(fam)
            family_counts[fam] = family_counts.get(fam, 0) + 1
    top_family = None
    top_family_count = 0
    if family_counts:
        top_family, top_family_count = max(family_counts.items(), key=lambda kv: kv[1])

    checks = {
        "enough_runs": total >= args.min_runs,
        "qualification_rate_ok": qual_rate >= args.required_qualification_rate,
        "promotion_pass_count_ok": promo_passed >= args.required_promotion_passes,
        "same_family_ok": (
            (not args.require_same_family)
            or (promo_passed == 0)
            or (top_family_count == promo_passed)
        ),
    }
    gate_pass = all(checks.values())

    out = {
        "config": {
            "glob": args.glob,
            "min_runs": args.min_runs,
            "required_qualification_rate": args.required_qualification_rate,
            "required_promotion_passes": args.required_promotion_passes,
            "require_same_family": args.require_same_family,
        },
        "summary": {
            "total_runs": total,
            "qualification_passed_runs": qual_passed,
            "qualification_pass_rate": qual_rate,
            "gate_ran_runs": gate_ran,
            "promotion_passed_runs": promo_passed,
            "promotion_pass_rate_over_total": (promo_passed / total) if total else 0.0,
            "promotion_pass_rate_over_gate_ran": (promo_passed / gate_ran) if gate_ran else 0.0,
            "top_promoted_family": top_family,
            "top_promoted_family_count": top_family_count,
        },
        "checks": checks,
        "gate_pass": gate_pass,
        "verdict": (
            "promotion_consistency_gate_passed"
            if gate_pass
            else "reopen_evaluation_keep_defaults_frozen"
        ),
        "runs": runs,
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
