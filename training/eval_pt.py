"""
eval_pt.py — reference evaluation for ByteGrid-44M checkpoints.

Metrics:
  1. Held-out MLM loss (cross-entropy on masked bytes, fixed seed)
  2. Top-1 and top-5 masked-byte prediction accuracy
  3. Qualitative demo: show model predictions on masked positions

Usage:
    .venv/bin/python training/eval_pt.py
    .venv/bin/python training/eval_pt.py --checkpoint weights_probe/ckpt_00032000.pt
    .venv/bin/python training/eval_pt.py --steps 1000 --seed 123
"""

import argparse, math, os, sys
import torch

# ── fixed reference texts ──────────────────────────────────────────────────
# OUT-OF-DOMAIN: diverse genres, never seen during training.
# Tests generalization beyond the training distribution.
OOD_TEXTS = [
    # English prose
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! The five boxing wizards jump quickly. Sphinx of b",
    # Python code
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in ran"
    "ge(n - 1):\n        a, b = b, a + b\n    return b\n\nprint([fibonacci(i) for i in rang",
    # JSON
    '{"model": "ByteGrid-44M", "params": 44400000, "layers": 24, "hidden": 512, "seq": 256'
    ', "objective": "masked_byte_prediction", "hardware": "Apple Neural Engine", "version"',
    # Mixed punctuation / numbers
    "In 2024, Apple shipped the M4 chip with a 38-TOPS Neural Engine. The ANE supports fp16"
    " and int8 quantized inference at up to 1.1 TB/s memory bandwidth. Peak performance is ",
    # Markdown
    "# ByteGrid-44M\n\nA byte-level bidirectional encoder designed for Apple Neural Engine."
    "\n\n## Architecture\n\n- 24 layers, D=512\n- Local+Global conv mixers\n- No attention\n",
    # Repetitive / compressible
    "aaaaaabbbbbbccccccddddddeeeeeeffffffgggggghhhhhhiiiiiijjjjjjkkkkkkllllllmmmmmmnnnnnnooo"
    "oooopppppppqqqqqqqrrrrrrrsssssssttttttttuuuuuuuuvvvvvvvwwwwwwwxxxxxxxxyyyyyyyyzzzzzzzz",
]

# IN-DOMAIN: FineWeb-Edu-style educational/academic prose.
# Matches the training distribution — a fairer measure of model learning.
INDOMAIN_TEXTS = [
    # Elementary science explanation
    "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide "
    "to produce oxygen and energy in the form of sugar. This process takes place in the chl",
    # High-school history
    "The French Revolution began in 1789 and fundamentally transformed France from a monarc"
    "hy to a republic. The causes included financial crisis, social inequality, and Enlighte",
    # University biology
    "Mitosis is a type of cell division that results in two daughter cells with the same num"
    "ber and type of chromosomes as the parent nucleus. It consists of four phases: prophase",
    # Educational math explanation
    "The Pythagorean theorem states that in a right-angled triangle, the square of the hypot"
    "enuse is equal to the sum of the squares of the other two sides. Written as an equation",
    # Academic writing style
    "This study examines the relationship between sleep duration and academic performance in"
    " undergraduate students. A total of 342 participants completed surveys measuring averag",
    # Encyclopedia-style
    "The water cycle, also known as the hydrological cycle, describes the continuous movement"
    " of water within Earth and its atmosphere. The main stages include evaporation, transpi",
    # Textbook explanation
    "Supply and demand is one of the most fundamental concepts in economics. The law of deman"
    "d states that, all else being equal, as the price of a product increases, the quantity ",
    # Educational narrative
    "Albert Einstein was born on March 14, 1879, in Ulm, in the Kingdom of Württemberg in t"
    "he German Empire. His parents were Hermann Einstein, a salesman and engineer, and Paulin",
]

# Combined list used by default evaluation
REFERENCE_TEXTS = OOD_TEXTS + INDOMAIN_TEXTS

DEMO_TEXT = (
    "The Apple Neural Engine is a dedicated hardware accelerator for machine learning "
    "inference. It is designed to run neural network operations efficiently on-device, "
    "providing high throughput at low power. ByteGrid uses only conv1x1 operations."
)

# ── helpers ────────────────────────────────────────────────────────────────

def text_to_tensor(text: str, seq: int, device) -> torch.Tensor:
    raw = text.encode("utf-8")[:seq]
    arr = [b for b in raw] + [0] * (seq - len(raw))
    return torch.tensor(arr[:seq], dtype=torch.long, device=device)


def make_fixed_masks(seq: int, mask_rate: float, seed: int, n: int, device) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    masks = torch.zeros(n, seq, dtype=torch.bool, device=device)
    n_mask = max(1, int(seq * mask_rate))
    for i in range(n):
        idx = torch.randperm(seq, generator=gen, device=device)[:n_mask]
        masks[i, idx] = True
    return masks


@torch.no_grad()
def evaluate_split(model, device, seq, mask_rate=0.15, seed=999):
    """Evaluate on OOD, in-domain, and combined splits separately."""
    model.eval()

    def _run(texts):
        n = len(texts)
        windows_t = torch.stack(
            [text_to_tensor(t, seq, device) for t in texts], dim=0
        )
        masks_t = make_fixed_masks(seq, mask_rate, seed, n, device)
        batch   = model.encode_bytes_batch(windows_t, masks_t)
        logits  = model(batch)
        loss    = model.compute_loss(logits, windows_t, masks_t)

        pred = logits[:, :, 0, :].permute(0, 2, 1)  # [n, seq, 256]
        top1_correct = top5_correct = total = 0
        for i in range(n):
            pos = masks_t[i].nonzero(as_tuple=True)[0]
            if len(pos) == 0:
                continue
            p = pred[i, pos]
            t = windows_t[i, pos]
            top1_correct += (p.argmax(dim=-1) == t).sum().item()
            top5_idx      = p.topk(5, dim=-1).indices
            top5_correct += (top5_idx == t.unsqueeze(1)).any(dim=1).sum().item()
            total        += len(pos)

        return loss.item(), math.exp(loss.item()), top1_correct/total*100, top5_correct/total*100

    ood      = _run(OOD_TEXTS)
    indomain = _run(INDOMAIN_TEXTS)
    combined = _run(REFERENCE_TEXTS)
    return ood, indomain, combined


@torch.no_grad()
def evaluate(model, device, seq, mask_rate=0.15, seed=999):
    ood, indomain, combined = evaluate_split(model, device, seq, mask_rate, seed)
    return combined  # loss, ppl, top1, top5


@torch.no_grad()
def demo(model, device, seq, mask_rate=0.15, n_show=10):
    model.eval()
    raw = DEMO_TEXT.encode("utf-8")[:seq]
    window = text_to_tensor(DEMO_TEXT, seq, device).unsqueeze(0)  # [1, seq]

    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    n_mask = max(1, int(seq * mask_rate))
    mask_idx = torch.randperm(seq, generator=gen, device=device)[:n_mask]
    mask = torch.zeros(1, seq, dtype=torch.bool, device=device)
    mask[0, mask_idx] = True

    batch  = model.encode_bytes_batch(window, mask)
    logits = model(batch)
    pred   = logits[0, :, 0, :].T  # [seq, 256]

    print("\n── Qualitative demo ─────────────────────────────────────────────")
    print(f"  Input: \"{DEMO_TEXT[:80]}...\"")
    print(f"  Showing first {n_show} masked positions:\n")
    print(f"  {'pos':>4}  {'true':>5}  {'top-1':>5}  {'top-2':>5}  {'top-3':>5}  {'correct'}")
    print(f"  {'---':>4}  {'----':>5}  {'-----':>5}  {'-----':>5}  {'-----':>5}  {'-------'}")

    shown = 0
    for pos in sorted(mask_idx.tolist()):
        if shown >= n_show:
            break
        true_byte = window[0, pos].item()
        top3 = pred[pos].topk(3).indices.tolist()
        def fmt(b):
            c = chr(b) if 32 <= b < 127 else f"\\x{b:02x}"
            return f"{c!r:>5}"
        correct = "✓" if top3[0] == true_byte else " "
        print(f"  {pos:>4}  {fmt(true_byte)}  {fmt(top3[0])}  {fmt(top3[1])}  {fmt(top3[2])}  {correct}")
        shown += 1
    print()


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="Path to .pt checkpoint (default: latest in weights_probe/)")
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--mask-rate", type=float, default=0.15)
    ap.add_argument("--no-demo", action="store_true")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # import here so PYTHONPATH doesn't need to be set
    sys.path.insert(0, os.path.dirname(__file__))
    from model_pt import make_model, SEQ

    model = make_model(device)

    # find checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        probe_dir = os.path.join(os.path.dirname(__file__), "..", "weights_probe")
        ckpts = sorted(
            [f for f in os.listdir(probe_dir) if f.startswith("ckpt_") and f.endswith(".pt")]
        )
        if not ckpts:
            print("No checkpoints found in weights_probe/. Run training first.")
            sys.exit(1)
        ckpt_path = os.path.join(probe_dir, ckpts[-1])

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    step = ckpt.get("step", "?")
    train_loss = ckpt.get("loss", float("nan"))
    print(f"  step={step}  train_loss={train_loss:.4f}\n")

    print("Running held-out evaluation...")
    ood, indomain, combined = evaluate_split(model, device, SEQ, args.mask_rate, args.seed)

    def row(label, r):
        return f"  {label:<22}  {r[0]:>8.4f}  {r[1]:>8.2f}  {r[2]:>9.1f}%  {r[3]:>9.1f}%"

    print(f"\n  {'Split':<22}  {'Loss':>8}  {'PPL':>8}  {'Top-1':>10}  {'Top-5':>10}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")
    print(row(f"In-domain (FineWeb-Edu)", indomain))
    print(row(f"Out-of-domain (diverse)", ood))
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")
    print(row("Combined", combined))
    print(f"\n  Mask rate: {args.mask_rate*100:.0f}%  seed={args.seed}  n_indomain={len(INDOMAIN_TEXTS)}  n_ood={len(OOD_TEXTS)}")

    if not args.no_demo:
        demo(model, device, SEQ, args.mask_rate)


if __name__ == "__main__":
    main()
