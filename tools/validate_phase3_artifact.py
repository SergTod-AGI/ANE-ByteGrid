#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate required sections in phase3 protocol artifact")
    parser.add_argument("--path", default="build/phase3_protocol.json")
    args = parser.parse_args()

    path = Path(args.path).resolve()
    if not path.exists():
        print(f"missing artifact: {path}")
        return 2

    data = json.loads(path.read_text())
    required = ["metadata", "demo_criterion", "search", "gate", "progression", "diagnostics", "validation"]
    missing = [key for key in required if key not in data]
    if missing:
        print(f"invalid artifact, missing sections: {missing}")
        return 1

    gate = data.get("gate", {})
    if "stability_delta" not in gate:
        print("invalid artifact, missing gate.stability_delta")
        return 1
    delta = gate.get("stability_delta", {})
    for key in ["avg_ms_mean_delta", "avg_ms_sd_delta", "passes_per_s_mean_delta", "eval_ce_mean_delta"]:
        if key not in delta:
            print(f"invalid artifact, missing gate.stability_delta.{key}")
            return 1

    diagnostics = data.get("diagnostics", {})
    for section in ["demo_baseline", "gate_baseline", "gate_candidate"]:
        if section not in diagnostics:
            print(f"invalid artifact, missing diagnostics.{section}")
            return 1

    if not data.get("validation", {}).get("ok", False):
        print("artifact validation section reports not ok")
        return 1

    print(f"phase3 artifact looks valid: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
