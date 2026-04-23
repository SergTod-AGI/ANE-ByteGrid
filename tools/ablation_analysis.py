#!/usr/bin/env python3
"""
Mixer ablation analysis via weight zeroing.
Evaluates the full trained model with local/global mixers disabled to measure component importance.
"""

import torch
import argparse
from pathlib import Path
from training.model_pt import make_model, SEQ
from training.data import ByteWindowStream
import math

def evaluate_with_ablation(model, eval_split="validation", device="mps", ablation_type="none"):
    """
    Run evaluation with optional component ablation.
    ablation_type: "none", "no_local", "no_global", "no_mixers" (both disabled)
    """
    model.eval()
    
    # Disable ablated components by zeroing weights
    if ablation_type in ["no_local", "no_mixers"]:
        for block in model.blocks:
            if hasattr(block, 'local_mixer'):
                block.local_mixer.conv.weight.data.zero_()
    
    if ablation_type in ["no_global", "no_mixers"]:
        for block in model.blocks:
            if hasattr(block, 'global_mixer'):
                block.global_mixer.conv.weight.data.zero_()
    
    # Standard eval texts (from eval_pt.py)
    INDOMAIN_TEXTS = [
        "Scientific discovery relies on careful observation and systematic hypothesis testing.",
        "The Industrial Revolution transformed manufacturing through mechanization and steam power.",
        "Molecular biology studies the structure and function of genes at the cellular level.",
        "Calculus enables computation of areas, volumes, and rates of change in continuous systems.",
        "The Byzantine Empire preserved Greek and Roman knowledge for over a thousand years.",
        "Paleontology reconstructs ancient life through analysis of fossilized remains.",
        "The human genome contains approximately three billion base pairs of DNA.",
        "Educational institutions across the world promote literacy and critical thinking.",
    ]
    
    OOD_TEXTS = [
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        '{"name": "Alice", "age": 30, "city": "Seattle"}',
        "# This is a markdown header\n## Secondary header\n- bullet point 1",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbbbbbbb cccccccccccccccccccc",
        "!@#$%^&*() ~`+-=[]{}|;:,.<>?/\\\"'",
        "123456 999888 555444 222111 777666 333444 888999",
    ]
    
    def eval_on_texts(texts, mask_rate=0.15):
        losses = []
        with torch.no_grad():
            for text in texts:
                # Encode text to bytes
                byte_seq = torch.tensor([ord(c) % 256 for c in text[:256]], dtype=torch.long, device=device)
                while len(byte_seq) < SEQ:
                    byte_seq = torch.cat([byte_seq, torch.zeros(1, dtype=torch.long, device=device)])
                byte_seq = byte_seq[:SEQ]
                
                # Create mask
                mask = torch.zeros(SEQ, dtype=torch.bool, device=device)
                n_mask = max(1, int(SEQ * mask_rate))
                mask_indices = torch.randperm(SEQ, device=device)[:n_mask]
                mask[mask_indices] = True
                
                # Encode
                batch_input = model.encode_bytes_batch(
                    byte_seq.unsqueeze(0), 
                    mask.unsqueeze(0)
                )
                
                # Forward + loss
                logits = model(batch_input)
                loss = model.compute_loss(logits, byte_seq.unsqueeze(0), mask.unsqueeze(0))
                losses.append(loss.item())
        
        return sum(losses) / len(losses) if losses else float('inf')
    
    in_domain_loss = eval_on_texts(INDOMAIN_TEXTS)
    ood_loss = eval_on_texts(OOD_TEXTS)
    combined_loss = (in_domain_loss * len(INDOMAIN_TEXTS) + ood_loss * len(OOD_TEXTS)) / (len(INDOMAIN_TEXTS) + len(OOD_TEXTS))
    
    return in_domain_loss, ood_loss, combined_loss


def main():
    parser = argparse.ArgumentParser(description="Mixer ablation evaluation")
    parser.add_argument("--checkpoint", type=str, default="weights_probe/ckpt_00500000.pt",
                       help="Checkpoint to load")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps/cpu)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = make_model(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    print("\n=== Mixer Ablation Analysis ===\n")
    print(f"{'Configuration':<20} {'In-Domain':<12} {'OOD':<12} {'Combined':<12}")
    print("-" * 56)
    
    # Evaluate full model
    in_d, ood, comb = evaluate_with_ablation(model, device=device, ablation_type="none")
    print(f"{'Full (baseline)':<20} {in_d:<12.3f} {ood:<12.3f} {comb:<12.3f}")
    
    baseline_loss = comb
    
    # Ablate local mixer
    in_d, ood, comb = evaluate_with_ablation(model, device=device, ablation_type="no_local")
    degradation = ((comb - baseline_loss) / baseline_loss) * 100
    print(f"{'No local mixer':<20} {in_d:<12.3f} {ood:<12.3f} {comb:<12.3f}  ({degradation:+.1f}%)")
    
    # Ablate global mixer
    in_d, ood, comb = evaluate_with_ablation(model, device=device, ablation_type="no_global")
    degradation = ((comb - baseline_loss) / baseline_loss) * 100
    print(f"{'No global mixer':<20} {in_d:<12.3f} {ood:<12.3f} {comb:<12.3f}  ({degradation:+.1f}%)")
    
    # Ablate both mixers
    in_d, ood, comb = evaluate_with_ablation(model, device=device, ablation_type="no_mixers")
    degradation = ((comb - baseline_loss) / baseline_loss) * 100
    print(f"{'No mixers (GLU only)':<20} {in_d:<12.3f} {ood:<12.3f} {comb:<12.3f}  ({degradation:+.1f}%)")
    
    print("\nNote: Higher loss indicates more important component.")


if __name__ == "__main__":
    main()
