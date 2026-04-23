# ANE-ByteGrid

ANE-ByteGrid is a byte-level, attention-free encoder designed for Apple Neural Engine (ANE) execution.
The project includes:

- A fixed-shape Objective-C runtime path using MIL + private `_ANEInMemoryModel` APIs.
- A PyTorch mirror used for gradient-based training on MPS GPU.
- Conversion utilities to serialize trained PyTorch weights into ANE blob format.
- Evaluation scripts and a publication draft under `paper/`.

## Final 500k Checkpoint Snapshot

- Model size: 44.4M parameters
- Sequence length: 256 bytes
- Objective: masked byte prediction (15% mask rate)
- Training device: Apple MPS (GPU)
- Inference target: Apple Neural Engine

Held-out metrics at step `500000`:

- In-domain (FineWeb-Edu): `Top-1 68.4%`, `Top-5 69.7%`, `Loss 3.7806`
- Out-of-domain: `Top-1 37.7%`, `Top-5 46.5%`, `Loss 5.4321`
- Combined: `Top-1 55.3%`, `Top-5 60.3%`, `Loss 4.4865`

## Repository Layout

- `training/`: model definitions, data pipeline, training, eval, weight bridge.
- `tests/`: Objective-C probes and unit tests for shape/runtime/MIL parsing.
- `tools/`: benchmarking and analysis scripts.
- `paper/`: manuscript and generated figures.
- `weights/`: ANE-format blobs.
- `weights_probe/`: PyTorch checkpoints.
- `build/`: generated experiment artifacts and reports.

## Quickstart

### 1) Environment

Requirements:

- macOS (Apple Silicon)
- Xcode command line tools
- Python 3.11+ (tested in `.venv`)

Setup:

```sh
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install torch datasets matplotlib
```

### 2) Objective-C Build + Tests

```sh
make
make test
```

### 3) PyTorch Training (MPS)

Smoke run:

```sh
make train-pt-smoke
```

Full run or resume:

```sh
make train-pt
make train-pt-resume
```

### 4) Evaluate a Checkpoint

```sh
.venv/bin/python training/eval_pt.py --checkpoint weights_probe/ckpt_00500000.pt
```

### 5) ANE Runtime Smoke

```sh
make compile-smoke
ANE_PROBE_CASE=all ./build/probe_runtime
```

## Reproducibility Commands

Generate protocol + benchmark artifacts:

```sh
make phase3-protocol
make ane-gpu-bench
make phase4-report
```

Generate training/eval loss figure:

```sh
.venv/bin/python tools/plot_loss_curves.py
```

## Publication Assets

- Architecture document: `ANE_ARCHITECTURE.md`
- Paper source: `paper/paper.tex`
- Loss figure: `paper/fig_loss_curves.pdf`

## Important Notes

- The ANE runtime path uses private APIs (`_ANEInMemoryModel`) and may break across macOS updates.
- Sandboxed environments can block ANE access (`mach-lookup com.apple.appleneuralengine`).
- This model is an encoder (masked prediction), not an autoregressive text generator.

## License

MIT (see `LICENSE`).
    - `build/update_loop_head_logit_speedhard.tsv`: 3/3 steps accepted (`head_logit 1.00 -> 0.97`).
    - `build/update_loop_residuals_speedhard.tsv`: rejected at step 1.
- Promotion gate status (long run):
  - `build/phase3_protocol_golden_hybrid_run1.json` and `build/phase3_protocol_golden_hybrid_run2.json` both produced `promote_defaults=true` on hybrid head candidates (consistent promotion verdict).
  - Defaults were promoted to `head_weight=0.98`, `head_logit=0.97`.
  - Post-promotion verification artifact: `build/phase3_protocol_golden_post_promotion.json`.
- Phase 3 protocol status:
  - `build/phase3_protocol_medium_hybrid.json` and golden artifacts record hybrid-head search/gate behavior with explicit pass/fail criteria.
  - Golden report artifact: `build/phase4_report_golden_post_promotion.md`.

## Next Implementation Steps

1. Run `make phase6-determinism` and `make phase6-validate` to track qualification-gated stability and jitter.
2. Generate `make phase6-report` and archive determinism runs as the new primary trend lane.
3. Continue model development around hybrid-head neighborhood, but require full protocol gate pass before any further default changes.
4. Keep GPU harness as optional reference lane (`make ane-gpu-bench`) when CUDA/PyTorch environment is available.
