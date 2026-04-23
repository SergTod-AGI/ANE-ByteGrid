"""
ANE-ByteGrid-44M — training loop.

Strategy
--------
- Forward + backward on MPS (Apple GPU) using the exact PyTorch mirror of the
  ANE model graph in training/model_pt.py
- AdamW optimizer with cosine LR schedule + linear warmup
- After every `--save-every` steps: serialize weights to .bin blobs so the
  ANE runtime can validate them with a forward pass
- Dataset: FineWeb-Edu sample-10BT (streaming, no full download needed)

Usage
-----
    python training/train_pt.py                   # start from scratch
    python training/train_pt.py --resume          # resume from weights/
    python training/train_pt.py --steps 100000 --lr 3e-4 --batch 32

Environment is auto-detected: MPS > CUDA > CPU.
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Make imports work whether run from repo root or training/
sys.path.insert(0, str(Path(__file__).parent))

from model_pt import ByteGridModel, make_model, SEQ, INPUT_CHANNELS

# Suppress noisy HuggingFace HTTP retry warnings (harmless prefetch errors on exit)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
from weights_bridge import save_weights_to_blobs, load_weights_from_blobs
from data import ByteWindowStream


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ANE-ByteGrid-44M training")
    p.add_argument("--steps",       type=int,   default=500_000,
                   help="Total training steps (default: 500000)")
    p.add_argument("--batch",       type=int,   default=16,
                   help="Batch size — number of 256-byte windows per step (default: 16)")
    p.add_argument("--lr",          type=float, default=3e-4,
                   help="Peak learning rate (default: 3e-4)")
    p.add_argument("--warmup",      type=int,   default=2000,
                   help="Linear warmup steps (default: 2000)")
    p.add_argument("--save-every",  type=int,   default=1000,
                   help="Save weights to blob format every N steps (default: 1000)")
    p.add_argument("--log-every",   type=int,   default=50,
                   help="Print loss every N steps (default: 50)")
    p.add_argument("--weight-decay",type=float, default=0.1,
                   help="AdamW weight decay (default: 0.1)")
    p.add_argument("--grad-clip",   type=float, default=1.0,
                   help="Gradient norm clip (default: 1.0)")
    p.add_argument("--seed",        type=int,   default=42,
                   help="Dataset shuffle seed (default: 42)")
    p.add_argument("--weights-dir", type=str,   default="weights",
                   help="Directory for .bin weight blobs (default: weights/)")
    p.add_argument("--ckpt-dir",    type=str,   default="weights_probe",
                   help="Directory for PyTorch .pt checkpoints (default: weights_probe/)")
    p.add_argument("--resume",      action="store_true",
                   help="Resume from latest checkpoint in --ckpt-dir")
    p.add_argument("--dtype",       type=str,   default="float32",
                   choices=["float32", "bfloat16"],
                   help="Training dtype (default: float32)")
    return p.parse_args()


# ── LR schedule ──────────────────────────────────────────────────────────

def lr_lambda(step: int, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay to 10% of peak LR."""
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return 0.1 + 0.9 * cosine   # decays to 10%


# ── batch builder ─────────────────────────────────────────────────────────

MASK_RATE = 0.15   # fraction of bytes to mask per window (BERT default)


def make_mask(seq_len: int, mask_rate: float, device: torch.device) -> torch.Tensor:
    """Return a BoolTensor [seq_len] with ~mask_rate positions True."""
    return torch.rand(seq_len, device=device) < mask_rate


def build_batch(
    windows: list[torch.Tensor],
    model: ByteGridModel,
    device: torch.device,
    dtype: torch.dtype,
    masks: list[torch.Tensor],
) -> torch.Tensor:
    """
    Encode a list of byte windows into a batched input tensor with masking.
    Uses the vectorized batch encoder for efficiency.
    Returns: [B, INPUT_CHANNELS, 1, SEQ]
    """
    windows_t = torch.stack(windows, dim=0)   # [B, S]  long
    masks_t   = torch.stack(masks,   dim=0)   # [B, S]  bool
    return model.encode_bytes_batch(windows_t, masked_positions=masks_t).to(dtype=dtype)


def forward_batch(
    model: ByteGridModel,
    batch: torch.Tensor,
    windows: list[torch.Tensor],
    masks: list[torch.Tensor],
) -> torch.Tensor:
    """
    Run a single batched forward pass and return mean masked-byte cross-entropy.
    batch:   [B, 320, 1, 256]
    windows: list of B LongTensors [256] — original (unmasked) bytes
    masks:   list of B BoolTensors [256] — True at masked positions
    """
    logits    = model(batch)                                   # [B, 256, 1, 256]
    windows_t = torch.stack(windows, dim=0)                    # [B, 256]
    masks_t   = torch.stack(masks,   dim=0)                    # [B, 256]
    return model.compute_loss(logits, windows_t, masks_t)


# ── checkpoint helpers ────────────────────────────────────────────────────

def save_checkpoint(model: ByteGridModel, optimizer, step: int, loss: float,
                    ckpt_dir: str) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_{step:08d}.pt")
    torch.save({
        "step":            step,
        "loss":            loss,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)
    # keep only the 3 most recent checkpoints to save disk space
    ckpts = sorted(Path(ckpt_dir).glob("ckpt_*.pt"))
    for old in ckpts[:-3]:
        old.unlink(missing_ok=True)


def load_checkpoint(model: ByteGridModel, optimizer, ckpt_dir: str,
                    device: torch.device) -> int:
    ckpts = sorted(Path(ckpt_dir).glob("ckpt_*.pt"))
    if not ckpts:
        return 0
    path = ckpts[-1]
    print(f"Resuming from {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    return state["step"]


# ── main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    # build model
    model = make_model(device)
    if dtype != torch.float32:
        model = model.to(dtype=dtype)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.1f}M")

    # optimizer — exclude scalars (alphas, logit_scale) from weight decay
    decay_params     = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay_params  = [p for n, p in model.named_parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    # resume
    start_step = 0
    if args.resume:
        if Path(args.weights_dir).exists():
            print(f"Loading .bin blobs from {args.weights_dir}/")
            load_weights_from_blobs(model, args.weights_dir)
        start_step = load_checkpoint(model, optimizer, args.ckpt_dir, device)

    # build scheduler after loading checkpoint so last_epoch is set correctly
    # last_epoch=start_step-1 because LambdaLR calls step() once in __init__,
    # advancing last_epoch by 1 and setting lr = base_lr * lambda(start_step)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: lr_lambda(s, args.warmup, args.steps),
        last_epoch=start_step - 1,
    )

    # dataset
    print(f"Streaming FineWeb-Edu (seed={args.seed})")
    stream = ByteWindowStream(split="train", seed=args.seed, device=device)
    window_iter = iter(stream)

    # training loop
    model.train()
    step      = start_step
    loss_acc  = 0.0
    t0        = time.time()

    print(f"Starting training: steps={args.steps} batch={args.batch} "
          f"lr={args.lr} warmup={args.warmup} dtype={dtype}")

    buffer: list[torch.Tensor] = []

    while step < args.steps:
        # fill batch buffer
        while len(buffer) < args.batch:
            try:
                buffer.append(next(window_iter))
            except StopIteration:
                # re-create stream with incremented seed for next epoch
                args.seed += 1
                stream = ByteWindowStream(split="train", seed=args.seed, device=device)
                window_iter = iter(stream)

        windows = buffer[:args.batch]
        buffer  = buffer[args.batch:]

        # forward + backward
        optimizer.zero_grad()
        masks = [make_mask(SEQ, MASK_RATE, device) for _ in windows]
        batch = build_batch(windows, model, device, dtype, masks)
        windows_t = torch.stack(windows, dim=0)
        masks_t   = torch.stack(masks,   dim=0)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type != "cpu")):
            logits = model(batch)
            loss   = model.compute_loss(logits, windows_t, masks_t)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()

        loss_acc += loss.item()
        step     += 1

        # logging
        if step % args.log_every == 0:
            avg_loss = loss_acc / args.log_every
            ppl      = math.exp(min(avg_loss, 20))
            elapsed  = time.time() - t0
            steps_s  = args.log_every / elapsed
            lr_now   = scheduler.get_last_lr()[0]
            print(
                f"step={step:>8d}  loss={avg_loss:.4f}  ppl={ppl:.1f}  "
                f"lr={lr_now:.2e}  {steps_s:.1f} steps/s"
            )
            loss_acc = 0.0
            t0       = time.time()

        # save
        if step % args.save_every == 0:
            print(f"[step {step}] saving weights...")
            save_weights_to_blobs(model, args.weights_dir)
            save_checkpoint(model, optimizer, step, loss.item(), args.ckpt_dir)
            print(f"[step {step}] saved.")

    # stop dataset iterator before final save to avoid background-thread errors
    del window_iter
    del stream

    # final save
    print("Training complete. Saving final weights...")
    save_weights_to_blobs(model, args.weights_dir)
    save_checkpoint(model, optimizer, step, loss.item(), args.ckpt_dir)
    print("Done.")


if __name__ == "__main__":
    main()
