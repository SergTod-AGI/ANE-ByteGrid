"""
Transformer baseline for comparison with ANE-ByteGrid.

Matches ByteGrid-44M's:
  - Input format: same [1, 320, 1, 256] channel-first encoding
  - Parameter count: ~44M
  - Task: masked byte prediction on 256-byte windows

Architecture: 14-layer BERT-style encoder, D=512, H=8, FFN=2048
  Per layer: MHA (4×512²) + FFN (2×512×2048) + LN×2 ≈ 3.15M × 14 ≈ 44.1M

Key difference from ByteGrid:
  - Uses softmax attention (GPU-native, ANE fallback)
  - Standard [B, S, D] sequence-first layout (transposed for CoreML)
  - All standard PyTorch ops (no conv1x1 constraint)

This model is used to benchmark throughput via CoreML export:
  python tools/export_baseline.py        # builds baseline.mlpackage
  python tools/benchmark_baseline.py     # measures ms/pass on ANE/GPU
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── constants matching ByteGrid config ────────────────────────────────────
SEQ            = 256   # sequence length in bytes
INPUT_CHANNELS = 320   # same stem input as ByteGrid
HIDDEN         = 512   # model dimension
VOCAB          = 256   # byte vocabulary

# ── baseline hyperparameters (matched by parameter count) ─────────────────
NUM_LAYERS = 14
NUM_HEADS  = 8
FFN_DIM    = 2048
HEAD_DIM   = HIDDEN // NUM_HEADS   # 64


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention (bidirectional, no causal mask)."""

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.k = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.v = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.o = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.scale = math.sqrt(HEAD_DIM)

    def forward(self, x):
        # x: [B, S, D]
        B, S, D = x.shape
        q = self.q(x).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)  # [B,H,S,d]
        k = self.k(x).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v(x).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale   # [B,H,S,S]
        attn = F.softmax(attn, dim=-1)
        out  = torch.matmul(attn, v)                                # [B,H,S,d]
        out  = out.transpose(1, 2).reshape(B, S, D)
        return self.o(out)


class FFN(nn.Module):
    """Position-wise feed-forward with GELU."""

    def __init__(self):
        super().__init__()
        self.up   = nn.Linear(HIDDEN, FFN_DIM, bias=False)
        self.down = nn.Linear(FFN_DIM, HIDDEN, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(HIDDEN)
        self.attn = MultiHeadSelfAttention()
        self.ln2  = nn.LayerNorm(HIDDEN)
        self.ffn  = FFN()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class BaselineTransformer(nn.Module):
    """
    44M-parameter transformer encoder that accepts the same 320-channel
    channel-first input as ByteGrid-44M.

    Input:  [B, 320, 1, 256]   (same as ANE ByteGrid)
    Output: [B, 256, 256]      (logits over byte vocab at each position)
    """

    def __init__(self):
        super().__init__()
        # Stem: flatten channel-first input → project to hidden
        # Input [B,320,1,256] → squeeze H dim → [B,320,256] → transpose → [B,256,320]
        # then linear 320→512 per position
        self.stem = nn.Linear(INPUT_CHANNELS, HIDDEN, bias=False)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYERS)])
        self.head_norm = nn.LayerNorm(HIDDEN)
        self.head = nn.Linear(HIDDEN, VOCAB, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x_ane):
        """
        Args:
            x_ane: [B, 320, 1, 256] — same tensor format as ByteGrid stem input
        Returns:
            logits: [B, 256, 256] — [batch, seq_pos, byte_vocab]
        """
        # [B,320,1,256] → [B,320,256] (squeeze H=1 dim)
        x = x_ane.squeeze(2)           # [B, 320, 256]
        x = x.permute(0, 2, 1)         # [B, 256, 320] = [B, S, input_channels]
        x = self.stem(x)               # [B, 256, 512] = [B, S, D]

        for block in self.blocks:
            x = block(x)

        x = self.head_norm(x)
        logits = self.head(x)          # [B, 256, 256]
        return logits

    def encode_bytes(self, byte_seq, device, masked_positions=None):
        """
        Build the same [1, 320, 1, 256] input tensor as ByteGrid.encode_bytes().
        This ensures identical input representations for a fair comparison.

        Args:
            byte_seq: LongTensor[S=256] of raw byte values 0..255
            device: torch device
            masked_positions: optional BoolTensor[S] — mask positions (zeroes byte/class)

        Returns:
            Tensor [1, 320, 1, 256] fp16
        """
        import numpy as np

        S = SEQ
        # Byte one-hot  [256, S]
        byte_onehot = torch.zeros(256, S, device=device, dtype=torch.float16)
        if masked_positions is not None:
            unmasked = ~masked_positions
            byte_onehot[byte_seq[unmasked].long(), unmasked.nonzero(as_tuple=True)[0]] = 1.0
        else:
            byte_onehot[byte_seq.long(), torch.arange(S, device=device)] = 1.0

        # Byte class features  [16, S]
        byte_class = torch.zeros(16, S, device=device, dtype=torch.float16)
        for i, b in enumerate(byte_seq.tolist()):
            if masked_positions is not None and masked_positions[i]:
                continue
            ch = chr(b)
            if ch.isalpha():
                c = 0 if ch.isupper() else 1
            elif ch.isdigit():
                c = 2
            elif ch in ' \t':
                c = 3
            elif ch in '\n\r':
                c = 4
            elif b < 0x80:
                c = 5
            elif b < 0xC0:
                c = 6
            elif b < 0xE0:
                c = 7
            elif b < 0xF0:
                c = 8
            else:
                c = 9
            byte_class[c, i] = 1.0

        # Position features  [32, S]  (binary Fourier)
        pos_feat = torch.zeros(32, S, device=device, dtype=torch.float16)
        for p in range(S):
            for k in range(16):
                pos_feat[2*k,   p] = math.sin(p / (10000 ** (2*k / 32)))
                pos_feat[2*k+1, p] = math.cos(p / (10000 ** (2*k / 32)))

        # Control features  [16, S]
        ctrl = torch.zeros(16, S, device=device, dtype=torch.float16)
        ctrl[9, 0] = 1.0   # BOS at position 0
        if masked_positions is not None:
            ctrl[10, masked_positions] = 1.0   # MASK indicator

        # Concatenate → [320, 1, S] → [1, 320, 1, S]
        x = torch.cat([byte_onehot, byte_class, pos_feat, ctrl], dim=0)  # [320, S]
        x = x.unsqueeze(1).unsqueeze(0)   # [1, 320, 1, 256]
        return x

    def compute_loss(self, logits, byte_seq, masked_positions):
        """
        Cross-entropy over masked positions only.
        Mirrors ByteGridModel.compute_loss() exactly.

        Args:
            logits: [1, 256, 256]  — [1, S, vocab]
            byte_seq: LongTensor[S]
            masked_positions: BoolTensor[S]

        Returns:
            scalar loss
        """
        logits_2d  = logits[0]                          # [S, vocab]
        targets    = byte_seq.long()                    # [S]
        masked_log = logits_2d[masked_positions]        # [M, vocab]
        masked_tgt = targets[masked_positions]          # [M]
        if masked_log.shape[0] == 0:
            return torch.tensor(0.0, device=logits.device)
        return F.cross_entropy(masked_log.float(), masked_tgt)


def make_model(device="cpu"):
    model = BaselineTransformer().to(device)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import sys

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model = make_model(device)
    n = count_parameters(model)
    print(f"Parameters: {n:,}  ({n/1e6:.1f}M)")

    # Smoke test
    B = 2
    x = torch.randn(B, INPUT_CHANNELS, 1, SEQ, device=device, dtype=torch.float16)
    model.eval()
    with torch.no_grad():
        out = model(x.float())  # forward pass in fp32 for stability
    print(f"Output shape: {out.shape}")   # [2, 256, 256]

    # Loss smoke test
    byte_seq = torch.randint(0, 256, (SEQ,), device=device)
    masks    = torch.zeros(SEQ, dtype=torch.bool, device=device)
    masks[torch.randperm(SEQ, device=device)[:38]] = True  # 15%
    inp  = model.encode_bytes(byte_seq, device, masked_positions=masks)
    logits = model(inp.float())
    loss = model.compute_loss(logits, byte_seq, masks)
    print(f"Smoke loss: {loss.item():.4f}  (uniform={math.log(256):.3f})")
    print("Baseline transformer smoke test passed.")
