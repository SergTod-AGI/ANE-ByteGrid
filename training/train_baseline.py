#!/usr/bin/env python3
"""
Train baseline transformer on masked byte prediction (same task as ByteGrid).
Quick smoke test: 20k steps to see convergence trajectory.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import argparse
import time
from pathlib import Path
from training.baseline_transformer_pt import BaselineTransformer
from training.model_pt import make_model as make_bytegrid, SEQ
from training.data import ByteWindowStream

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000, help="Training steps")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--save-dir", type=str, default="weights_probe_baseline", help="Checkpoint dir")
    parser.add_argument("--log-every", type=int, default=100, help="Log every N steps")
    args = parser.parse_args()
    
    Path(args.save_dir).mkdir(exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create baseline transformer
    baseline = BaselineTransformer().to(device)
    baseline.train()
    print(f"Baseline transformer created: {sum(p.numel() for p in baseline.parameters()):,} params")
    
    # Optimizer
    optimizer = optim.AdamW(baseline.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    
    # Data stream
    stream = ByteWindowStream(split="train", seed=42, device=device)
    window_iter = iter(stream)
    buffer = []
    
    # Training loop
    step = 0
    loss_acc = 0.0
    t0 = time.time()
    
    while step < args.steps:
        # Fill buffer
        while len(buffer) < args.batch:
            try:
                buffer.append(next(window_iter))
            except StopIteration:
                stream = ByteWindowStream(split="train", seed=42, device=device)
                window_iter = iter(stream)
        
        windows = buffer[:args.batch]
        buffer = buffer[args.batch:]
        
        # Build batch
        byte_seqs = torch.stack([w for w in windows], dim=0)  # [B, 256]
        
        # Create masks (15% mask rate)
        masks = torch.zeros(args.batch, SEQ, dtype=torch.bool, device=device)
        for i in range(args.batch):
            n_mask = max(1, int(SEQ * 0.15))
            mask_indices = torch.randperm(SEQ, device=device)[:n_mask]
            masks[i, mask_indices] = True
        
        # Encode (same input encoding as ByteGrid)
        bytegrid_tmp = make_bytegrid(device)
        input_batch = bytegrid_tmp.encode_bytes_batch(byte_seqs, masks)  # [B, 320, 1, 256]
        
        # Transpose for transformer [B, 256, 320]
        input_batch = input_batch.squeeze(2).permute(0, 2, 1)
        
        # Forward
        optimizer.zero_grad()
        logits = baseline(input_batch)  # [B, 256, 256]
        
        # Compute loss on masked positions
        loss = 0.0
        for i in range(args.batch):
            mask_pos = masks[i].nonzero(as_tuple=True)[0]
            if len(mask_pos) > 0:
                logits_masked = logits[i, mask_pos, :]  # [n_mask, 256]
                targets_masked = byte_seqs[i, mask_pos]   # [n_mask]
                loss += nn.functional.cross_entropy(logits_masked, targets_masked)
        loss = loss / args.batch
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.0)
        optimizer.step()
        
        loss_acc += loss.item()
        step += 1
        
        if step % args.log_every == 0:
            avg_loss = loss_acc / args.log_every
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t0
            steps_s = args.log_every / elapsed
            print(f"step={step:>7d}  loss={avg_loss:.4f}  ppl={ppl:.1f}  {steps_s:.1f} steps/s")
            loss_acc = 0.0
            t0 = time.time()
    
    print(f"\nTraining complete at {step} steps.")


if __name__ == "__main__":
    main()
