#!/usr/bin/env python3
"""
Generate loss curves from training checkpoints and eval results.
"""
import torch
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Paths
CKPT_DIR = Path(__file__).parent.parent / "weights_probe"
PAPER_DIR = Path(__file__).parent.parent / "paper"

# Known eval results (from eval_pt.py runs at each checkpoint)
eval_results = {
    114000: {"train_loss": 0.406, "in_domain_loss": 4.62, "ood_loss": None},
    235000: {"train_loss": 0.367, "in_domain_loss": 4.28, "ood_loss": None},
    270000: {"train_loss": 0.355, "in_domain_loss": 4.27, "ood_loss": None},
    380000: {"train_loss": 0.322, "in_domain_loss": 4.50, "ood_loss": None},
    430000: {"train_loss": 0.204, "in_domain_loss": 4.38, "ood_loss": None},
    500000: {"train_loss": 0.251, "in_domain_loss": 4.49, "ood_loss": None},
}

# Extract train_loss from checkpoints
checkpoints = {}
for ckpt_file in sorted(CKPT_DIR.glob("ckpt_*.pt")):
    try:
        ckpt = torch.load(ckpt_file, map_location="cpu")
        step = ckpt["step"]
        loss = ckpt["loss"]
        checkpoints[step] = loss
        print(f"Loaded {ckpt_file.name}: step={step}, loss={loss:.4f}")
    except Exception as e:
        print(f"Failed to load {ckpt_file.name}: {e}")

# Combine with eval data
all_steps = sorted(set(checkpoints.keys()) | set(eval_results.keys()))
train_losses = [eval_results.get(s, {}).get("train_loss") or checkpoints.get(s) for s in all_steps]
in_domain_losses = [eval_results.get(s, {}).get("in_domain_loss") for s in all_steps]

# Print summary
print("\n=== Loss Trajectory ===")
for step, loss in zip(all_steps, train_losses):
    print(f"Step {step:>7d}: train_loss={loss:.4f}")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training loss (all steps)
all_steps_scaled = [s / 1000 for s in all_steps]
ax1.plot(all_steps_scaled, train_losses, "o-", linewidth=2, markersize=6, label="Training loss")
ax1.set_xlabel("Training steps (×1000)", fontsize=11)
ax1.set_ylabel("Cross-entropy loss", fontsize=11)
ax1.set_title("Training Loss Trajectory", fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 510)
ax1.legend(fontsize=10)

# Add value labels on points
for x, y in zip(all_steps_scaled, train_losses):
    if y is not None:
        ax1.text(x, y + 0.02, f"{y:.3f}", ha="center", fontsize=8)

# Plot 2: Eval loss (in-domain only)
eval_steps_scaled = [s / 1000 for s in all_steps if in_domain_losses[all_steps.index(s)] is not None]
eval_losses = [l for l in in_domain_losses if l is not None]
ax2.plot(eval_steps_scaled, eval_losses, "s-", linewidth=2, markersize=7, color="orange", label="In-domain eval loss")
ax2.set_xlabel("Training steps (×1000)", fontsize=11)
ax2.set_ylabel("Eval loss (held-out)", fontsize=11)
ax2.set_title("Held-Out Eval Loss (In-Domain)", fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 510)
ax2.legend(fontsize=10)

# Add value labels
for x, y in zip(eval_steps_scaled, eval_losses):
    ax2.text(x, y + 0.08, f"{y:.2f}", ha="center", fontsize=8)

plt.tight_layout()
output_path = PAPER_DIR / "fig_loss_curves.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {output_path}")

# Also save as PDF for paper
output_pdf = PAPER_DIR / "fig_loss_curves.pdf"
plt.savefig(output_pdf, bbox_inches="tight")
print(f"Saved: {output_pdf}")

plt.close()
