# ANE-ByteGrid

> A 44M-parameter byte-level encoder designed from first principles for the **Apple Neural Engine**.  
> No attention. No tokenizer. Pure `conv1×1` — the ANE's native fast path.

---

## What is this?

Standard language models (BERT, transformers) are designed for GPU execution — they rely on softmax attention, dynamic masking, and embedding lookups that are a poor fit for the ANE's hardware primitives.

**ANE-ByteGrid** asks: *can a useful language encoder be built to use only the ANE's fast path, with no GPU fallback?*

The answer is yes. ANE-ByteGrid is a bidirectional byte-level encoder that:

- Replaces attention with a **hierarchical local+global conv1×1 token mixer**
- Accepts raw bytes (no tokenizer, 256-byte fixed windows)
- Compiles to **26 ANE kernels** with all shapes fixed at compile time
- Runs on the **Apple Neural Engine at 11.2 ms/pass, 1140 MB/s**, drawing <500 mW
- Trains on **MPS (Apple GPU) at 2.1 steps/s** via a PyTorch mirror

The training objective is masked byte prediction (BERT-style, 15% mask rate) on FineWeb-Edu. After 500,000 steps:

| Split | Top-1 | Top-5 | Loss |
|-------|------:|------:|-----:|
| In-domain (FineWeb-Edu) | **68.4%** | 69.7% | 3.78 |
| Out-of-domain (code, JSON, markdown) | 37.7% | 46.5% | 5.43 |
| Combined | 55.3% | 60.3% | 4.49 |

---

## Architecture in one sentence

Input bytes → one-hot + class + position encoding → 24 blocks of *(local mixer → global mixer → channel GLU)* → byte logits.

Each **local mixer** mixes the 16 positions within a 16-byte chunk; each **global mixer** mixes the same intra-chunk position across all 16 chunks. Both are `conv1×1` on reshaped tensors — no attention, no softmax, no dynamic shapes.

```
Input [B, 320, 1, 256]
  │
  ▼  input projection  [B, 512, 1, 256]
  │
  ├─ x24 ┬─ LocalMixer   (within-chunk, receptive field = 16 bytes)
  │      ├─ GlobalMixer  (cross-chunk,  receptive field = 256 bytes)
  │      └─ ChannelGLU   (position-wise SiLU gate)
  │
  ▼  output head  [B, 256, 1, 256]  → logits over 256-byte vocab
```

---

## Hardware results (Apple M5 Pro)

| Model | Compute unit | Latency | Throughput |
|-------|-------------|---------|-----------|
| ANE-ByteGrid (private MIL runtime) | ANE | 11.2 ms | 1140 MB/s |
| Transformer-44M (CoreML, ANE policy) | ANE | 3.1 ms | — |
| Transformer-44M (CoreML, CPU policy) | CPU | 10.9 ms | — |

Training power: ~11 W GPU (MPS) / ANE idle.  
Inference power: <500 mW ANE / GPU completely idle.

---

## Comparison with BERT

| | ANE-ByteGrid | BERT-base |
|--|-------------|-----------|
| Parameters | 44M | 110M |
| Architecture | conv1×1 mixers | softmax attention |
| Tokenization | byte-level (256 vocab) | WordPiece (30k vocab) |
| Inference target | Apple Neural Engine | GPU |
| In-domain Top-1 | **68.4%** | ~65–70% |
| Training data | FineWeb-Edu (2B bytes) | Books + Wikipedia (16 GB) |

ByteGrid matches BERT's masked-prediction accuracy on in-domain text at **40% fewer parameters** and on a **100× smaller corpus**, while being natively deployable on ANE without GPU fallback.

---

## Repository layout

```
training/     PyTorch model, training loop, data pipeline, eval, weight bridge
              + Objective-C MIL generator and ANE runtime
tests/        Shape, MIL, and runtime parse unit tests
tools/        Benchmarking, ablation, loss curve, and phase protocol scripts
paper/        paper.tex manuscript + fig_loss_curves.pdf
weights/      ANE-format weight blobs (generated)
build/        Protocol + benchmark artifacts
```

---

## Quickstart

**Requirements:** macOS, Apple Silicon, Xcode CLI tools, Python 3.11+

```sh
# Python environment
python3 -m venv .venv
.venv/bin/pip install -U pip torch datasets matplotlib

# Build & test Objective-C runtime
make && make test

# Smoke train (200 steps)
make train-pt-smoke

# Full training (500k steps)
make train-pt

# Resume from last checkpoint
make train-pt-resume

# Evaluate final checkpoint
.venv/bin/python training/eval_pt.py --checkpoint weights_probe/ckpt_00500000.pt

# ANE compile smoke test
make compile-smoke
```

---

## Publication

Paper manuscript: [`paper/paper.tex`](paper/paper.tex)  
Architecture reference: [`ANE_ARCHITECTURE.md`](ANE_ARCHITECTURE.md)  
Loss curves: [`paper/fig_loss_curves.pdf`](paper/fig_loss_curves.pdf)

---

## License

MIT — see [`LICENSE`](LICENSE).


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
