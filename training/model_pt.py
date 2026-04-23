"""
ANE-ByteGrid-44M — exact PyTorch mirror of the MIL graph.

All shapes match the ANE kernel exactly so gradients are valid for the same
weights used during ANE inference.  The design follows ANE_ARCHITECTURE.md.

Shape convention: [N=1, C, 1, S=256]  (channel-first, height=1)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── constants from config.h ────────────────────────────────────────────────
SEQ            = 256
INPUT_CHANNELS = 320
BYTE_CHANNELS  = 256
CLASS_CHANNELS = 16
POS_CHANNELS   = 32
CTRL_CHANNELS  = 16
HIDDEN         = 512
LAYERS         = 24
BLOCK          = 16   # local block size
GROUPS         = 16   # number of blocks = SEQ // BLOCK
PACKED         = HIDDEN * BLOCK   # 8192
GLU            = 1024
VOCAB          = 256


# ── helpers ────────────────────────────────────────────────────────────────

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    x: [N, C, 1, S]
    w: [1, C, 1, 1]
    Mirrors the MIL RMSNorm: sum over channel axis, multiply by learnable scale.
    Note: the MIL implementation divides by 512 (1/512 = 0.001953125) then adds eps
    before taking rsqrt — this is a mean-of-squares RMSNorm over the channel axis.
    """
    # sum of squares over channel dim, keep dims
    ss = (x * x).sum(dim=1, keepdim=True)          # [N,1,1,S]
    mean_sq = ss * (1.0 / HIDDEN)                   # divide by 512
    rrms = torch.rsqrt(mean_sq + eps)               # [N,1,1,S]
    return x * rrms * w                             # broadcast [N,C,1,S]


# ── sub-layers ─────────────────────────────────────────────────────────────

class LocalMixer(nn.Module):
    """
    Mix tokens within each 16-byte chunk via grouped conv1x1.
    Mirrors ane_bg_gen_local_mixer_mil().
    """
    def __init__(self):
        super().__init__()
        # weight: [PACKED=8192, BLOCK=16, 1, 1], groups=HIDDEN=512
        # each of the 512 groups mixes 16 positions inside one chunk
        self.conv = nn.Conv2d(
            in_channels=PACKED,
            out_channels=PACKED,
            kernel_size=(1, 1),
            groups=HIDDEN,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 512, 1, 256]
        B = x.shape[0]
        x4 = x.view(B, HIDDEN, BLOCK, GROUPS)                # [B,512,16,16]
        xt = x4.permute(0, 1, 3, 2)                          # [B,512,16,16] chunk<->pos swap
        xp = xt.reshape(B, PACKED, 1, GROUPS)                # [B,8192,1,16]
        ym = self.conv(xp)                                    # [B,8192,1,16]
        y4 = ym.view(B, HIDDEN, BLOCK, GROUPS)               # [B,512,16,16]
        yi = y4.permute(0, 1, 3, 2)                          # [B,512,16,16]
        return yi.reshape(B, HIDDEN, 1, SEQ)                  # [B,512,1,256]


class GlobalMixer(nn.Module):
    """
    Mix same intra-chunk positions across all 16 chunks via grouped conv1x1.
    Mirrors ane_bg_gen_global_mixer_mil().
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=PACKED,
            out_channels=PACKED,
            kernel_size=(1, 1),
            groups=HIDDEN,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 512, 1, 256]
        B = x.shape[0]
        x4 = x.view(B, HIDDEN, BLOCK, GROUPS)                # [B,512,16,16]
        # no transpose here — spatial axis is pos_in_chunk (GROUPS)
        xp = x4.reshape(B, PACKED, 1, BLOCK)                 # [B,8192,1,16]
        ym = self.conv(xp)                                    # [B,8192,1,16]
        y4 = ym.view(B, HIDDEN, BLOCK, GROUPS)               # [B,512,16,16]
        return y4.reshape(B, HIDDEN, 1, SEQ)                  # [B,512,1,256]


class ChannelGLU(nn.Module):
    """
    Channel mixer: SiLU-gated MLP via conv1x1.
    Mirrors ane_bg_gen_channel_glu_mil().
    """
    def __init__(self):
        super().__init__()
        self.wv = nn.Conv2d(HIDDEN, GLU, kernel_size=1, bias=False)
        self.wg = nn.Conv2d(HIDDEN, GLU, kernel_size=1, bias=False)
        self.wo = nn.Conv2d(GLU, HIDDEN, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.wv(x)                       # [1,1024,1,256]
        h3 = self.wg(x)                       # [1,1024,1,256]
        silu = h1 * torch.sigmoid(h1)         # SiLU on value branch
        gate = silu * h3                      # element-wise gate
        return self.wo(gate)                  # [1,512,1,256]


class ByteGridBlock(nn.Module):
    """
    One full block: local mixer + global mixer + channel GLU, each with
    RMSNorm pre-norm and a per-sublayer learnable alpha scalar residual.
    Mirrors ane_bg_gen_block_mil().
    """
    def __init__(self):
        super().__init__()
        # RMSNorm scale vectors [1, HIDDEN, 1, 1]
        self.rms_local  = nn.Parameter(torch.ones(1, HIDDEN, 1, 1))
        self.rms_global = nn.Parameter(torch.ones(1, HIDDEN, 1, 1))
        self.rms_ffn    = nn.Parameter(torch.ones(1, HIDDEN, 1, 1))

        # learnable residual alpha scalars (initialized at model construction)
        self.alpha_local  = nn.Parameter(torch.tensor(0.0))
        self.alpha_global = nn.Parameter(torch.tensor(0.0))
        self.alpha_mlp    = nn.Parameter(torch.tensor(0.0))

        self.local_mixer  = LocalMixer()
        self.global_mixer = GlobalMixer()
        self.channel_glu  = ChannelGLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sublayer A: local token mixing
        u  = rms_norm(x, self.rms_local)
        x  = x + self.alpha_local  * self.local_mixer(u)

        # sublayer B: global token mixing
        v  = rms_norm(x, self.rms_global)
        x  = x + self.alpha_global * self.global_mixer(v)

        # sublayer C: channel GLU
        w  = rms_norm(x, self.rms_ffn)
        x  = x + self.alpha_mlp   * self.channel_glu(w)

        return x


class ByteGridModel(nn.Module):
    """
    Full ANE-ByteGrid-44M model in PyTorch.

    Input:  raw byte sequence as a LongTensor of shape [S] (values 0-255)
    Output: logits [1, VOCAB=256, 1, S=256]

    Training target: masked byte prediction (BERT-style).
    The architecture is bidirectional — all positions are visible simultaneously.
    Next-byte prediction would let the model trivially copy b[t+1] from position
    t+1 in the input, so masked prediction is the correct objective.
    """

    def __init__(self):
        super().__init__()
        # stem: conv1x1 320 -> 512
        self.stem = nn.Conv2d(INPUT_CHANNELS, HIDDEN, kernel_size=1, bias=False)

        # 24 core blocks
        self.blocks = nn.ModuleList([ByteGridBlock() for _ in range(LAYERS)])

        # output head
        self.head_rms = nn.Parameter(torch.ones(1, HIDDEN, 1, 1))
        self.head_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.head = nn.Conv2d(HIDDEN, VOCAB, kernel_size=1, bias=False)

    def encode_bytes(
        self,
        byte_seq: torch.Tensor,
        device: torch.device,
        masked_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Build the [1, 320, 1, 256] input tensor from a raw byte sequence.

        byte_seq:         LongTensor [S] with values 0-255
        masked_positions: BoolTensor [S] — True at positions to mask.
                          Masked positions have their byte one-hot and class
                          channels zeroed out, and ctrl channel 10 set to 1.0
                          (the MASK token indicator).
        """
        S = byte_seq.shape[0]
        buf = torch.zeros(1, INPUT_CHANNELS, 1, S, dtype=torch.float16, device=device)
        t = torch.arange(S, device=device)

        # one-hot byte channels [0:256] — zero for masked positions
        visible = torch.ones(S, dtype=torch.bool, device=device)
        if masked_positions is not None:
            visible = ~masked_positions
        vis_idx = t[visible]
        buf[0, byte_seq[visible], 0, vis_idx] = 1.0

        # byte-class channels [256:272] — zero for masked positions
        class_map = _byte_class_table(device)
        cls = class_map[byte_seq]
        buf[0, BYTE_CHANNELS + cls[visible], 0, vis_idx] = 1.0

        # position channels [272:304] — fixed Fourier binary features (always present)
        pos_offset = BYTE_CHANNELS + CLASS_CHANNELS
        for i in range(POS_CHANNELS // 2):
            period = 1 << (i + 1)
            bit = ((t // period) & 1).bool()
            buf[0, pos_offset + 2 * i,     0, :] = torch.where(bit,  torch.ones(S, device=device),  -torch.ones(S, device=device))
            buf[0, pos_offset + 2 * i + 1, 0, :] = torch.where(bit, -torch.ones(S, device=device),   torch.ones(S, device=device))

        # control channels [304:320]
        ctrl_offset = pos_offset + POS_CHANNELS
        buf[0, ctrl_offset + 0, 0, 0] = 1.0                      # BOS
        buf[0, ctrl_offset + 1, 0, S - 1] = 1.0                  # EOS
        block_starts = (t % BLOCK == 0)
        buf[0, ctrl_offset + 2, 0, block_starts] = 1.0           # block boundary
        first_half = ((t % 32) < 16)
        buf[0, ctrl_offset + 3, 0, first_half] = 1.0             # first-half marker
        # per-byte property channels (4-9) — zero for masked positions
        byte_vals = byte_seq.to(torch.long)
        odd   = (byte_vals & 1).bool()
        digit = (byte_vals >= 48) & (byte_vals <= 57)
        lower = (byte_vals >= 97) & (byte_vals <= 122)
        upper = (byte_vals >= 65) & (byte_vals <= 90)
        space = (byte_vals == 32) | (byte_vals == 10) | (byte_vals == 9)
        sep   = (byte_vals == 124) | (byte_vals == 45) | (byte_vals == 95)
        for ch_idx, prop in enumerate([odd, digit, lower, upper, space, sep], start=4):
            active = prop & visible
            buf[0, ctrl_offset + ch_idx, 0, active] = 1.0
        # ctrl channel 10: MASK indicator
        if masked_positions is not None:
            buf[0, ctrl_offset + 10, 0, masked_positions] = 1.0

        return buf

    def encode_bytes_batch(
        self,
        byte_seqs: torch.Tensor,
        masked_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Vectorized batch encoder. Builds [B, 320, 1, S] in one shot.

        byte_seqs:        LongTensor [B, S] with values 0-255
        masked_positions: BoolTensor [B, S] — True at masked positions

        Returns: float16 tensor [B, 320, 1, S]
        """
        B, S = byte_seqs.shape
        device = byte_seqs.device
        buf = torch.zeros(B, INPUT_CHANNELS, 1, S, dtype=torch.float16, device=device)

        t = torch.arange(S, device=device)  # [S]

        # --- byte one-hot channels [0:256] ---
        # visible[b, s] = True when not masked
        if masked_positions is not None:
            visible = ~masked_positions   # [B, S]
        else:
            visible = torch.ones(B, S, dtype=torch.bool, device=device)

        # batch_idx, seq_idx where visible
        b_vis, s_vis = visible.nonzero(as_tuple=True)          # [M], [M]
        byte_vis = byte_seqs[b_vis, s_vis]                     # [M]
        buf[b_vis, byte_vis, 0, s_vis] = 1.0

        # --- byte-class channels [256:272] ---
        class_map = _byte_class_table(device)                  # [256] → [0,15]
        cls = class_map[byte_seqs]                             # [B, S]
        cls_vis = cls[b_vis, s_vis]                            # [M]
        buf[b_vis, BYTE_CHANNELS + cls_vis, 0, s_vis] = 1.0

        # --- position channels [272:304] (same for all samples) ---
        pos_offset = BYTE_CHANNELS + CLASS_CHANNELS
        for i in range(POS_CHANNELS // 2):
            period = 1 << (i + 1)
            bit = ((t // period) & 1).bool()                   # [S]
            pos_val = torch.where(bit,
                                  torch.ones(S, device=device, dtype=torch.float16),
                                  -torch.ones(S, device=device, dtype=torch.float16))
            buf[:, pos_offset + 2 * i,     0, :] = pos_val.unsqueeze(0)
            buf[:, pos_offset + 2 * i + 1, 0, :] = -pos_val.unsqueeze(0)

        # --- control channels [304:320] ---
        ctrl_offset = pos_offset + POS_CHANNELS
        buf[:, ctrl_offset + 0, 0, 0]       = 1.0             # BOS
        buf[:, ctrl_offset + 1, 0, S - 1]   = 1.0             # EOS
        block_starts = (t % BLOCK == 0)
        buf[:, ctrl_offset + 2, 0, block_starts] = 1.0        # block boundary
        first_half = ((t % 32) < 16)
        buf[:, ctrl_offset + 3, 0, first_half]   = 1.0        # first-half marker

        # per-byte property channels (4-9) — only at visible positions
        byte_vals = byte_seqs                                  # [B, S] long
        props = [
            (byte_vals & 1).bool(),                                                  # odd
            (byte_vals >= 48) & (byte_vals <= 57),                                   # digit
            (byte_vals >= 97) & (byte_vals <= 122),                                  # lower
            (byte_vals >= 65) & (byte_vals <= 90),                                   # upper
            (byte_vals == 32) | (byte_vals == 10) | (byte_vals == 9),               # space
            (byte_vals == 124) | (byte_vals == 45) | (byte_vals == 95),             # sep
        ]
        for ch_idx, prop in enumerate(props, start=4):
            active = prop & visible                            # [B, S]
            b_a, s_a = active.nonzero(as_tuple=True)
            buf[b_a, ctrl_offset + ch_idx, 0, s_a] = 1.0

        # MASK indicator channel 10
        if masked_positions is not None:
            b_m, s_m = masked_positions.nonzero(as_tuple=True)
            buf[b_m, ctrl_offset + 10, 0, s_m] = 1.0

        return buf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 320, 1, 256] fp16 or fp32 input tensor
        returns: logits [B, 256, 1, 256]
        """
        # cast input to the model's parameter dtype
        x = x.to(dtype=next(self.parameters()).dtype)
        h = self.stem(x)                          # [B,512,1,256]

        for block in self.blocks:
            h = block(h)

        h = rms_norm(h, self.head_rms)
        logits = self.head(h) * self.head_logit_scale
        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        byte_seqs: torch.Tensor,
        masked_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Masked byte prediction cross-entropy loss — supports both batched and
        single-sample inputs.

        Batched (preferred):
          logits:           [B, 256, 1, 256]
          byte_seqs:        LongTensor [B, S]   — original (unmasked) byte sequences
          masked_positions: BoolTensor [B, S]   — True at masked positions

        Single-sample (legacy):
          logits:           [1, 256, 1, 256]
          byte_seqs:        LongTensor [S]
          masked_positions: BoolTensor [S]

        Loss is computed only over masked positions.
        """
        if byte_seqs.dim() == 1:
            # single-sample path (kept for backward compat)
            pred   = logits[0, :, 0, :].T          # [S, vocab]
            pred_m = pred[masked_positions]         # [n_masked, vocab]
            tgt    = byte_seqs[masked_positions].long()
            return F.cross_entropy(pred_m, tgt)

        # batched path: logits [B, vocab, 1, S], seqs [B, S], masks [B, S]
        # flatten all masked positions across the batch
        pred_flat = logits[:, :, 0, :].permute(0, 2, 1)  # [B, S, vocab]
        tgt_flat  = byte_seqs.long()                       # [B, S]
        pred_m    = pred_flat[masked_positions]            # [total_masked, vocab]
        tgt_m     = tgt_flat[masked_positions]             # [total_masked]
        return F.cross_entropy(pred_m, tgt_m)


# ── byte class lookup (mirrors train_byte_class in train.m) ───────────────

_byte_class_cache: dict = {}

def _byte_class_table(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key in _byte_class_cache:
        return _byte_class_cache[key]
    table = torch.zeros(256, dtype=torch.long)
    for b in range(256):
        c = b
        if 97 <= c <= 122:       table[b] = 0
        elif 65 <= c <= 90:      table[b] = 1
        elif 48 <= c <= 57:      table[b] = 2
        elif c == 32:            table[b] = 3
        elif c == 10:            table[b] = 4
        elif c == 9:             table[b] = 5
        elif c in (45, 95):      table[b] = 6
        elif c in (124, 47, 92): table[b] = 7
        elif c in (46, 44, 59, 58): table[b] = 8
        elif c in (40, 41, 91, 93, 123, 125): table[b] = 9
        elif c in (34, 39):      table[b] = 10
        elif c < 32:             table[b] = 11
        elif c < 128:            table[b] = 12
        elif (c & 0xe0) == 0xc0: table[b] = 13
        elif (c & 0xc0) == 0x80: table[b] = 14
        else:                    table[b] = 15
    table = table.to(device)
    _byte_class_cache[key] = table
    return table


# ── weight initializer matching ane_bg_fill_xavier_uniform ────────────────

def _init_xavier_like(weight: torch.Tensor, seed: int) -> None:
    """
    Replicate the deterministic Xavier uniform used in model.m so that
    a freshly constructed Python model starts from the same distribution.
    (For reproducibility; actual training will diverge from the ANE blob anyway.)
    """
    rows, cols = weight.numel() // weight.shape[0], weight.shape[0]
    limit = math.sqrt(6.0 / (rows + cols))
    nn.init.uniform_(weight, -limit, limit)


def make_model(device: torch.device) -> ByteGridModel:
    """Construct and initialize the model on the given device."""
    model = ByteGridModel().to(device=device, dtype=torch.float32)

    # Initialize alphas to match the Objective-C defaults
    # alpha_base = 1/sqrt(24) ≈ 0.2041
    alpha_base = 1.0 / math.sqrt(LAYERS)
    alpha_local_mul  = 1.15
    alpha_global_mul = 1.40
    alpha_mlp_mul    = 0.05

    for layer_idx, block in enumerate(model.blocks):
        depth = (layer_idx + 1) / LAYERS           # depth_power=0.0 → factor=1.0
        block.alpha_local.data.fill_(alpha_base * alpha_local_mul  * depth)
        block.alpha_global.data.fill_(alpha_base * alpha_global_mul * depth)
        block.alpha_mlp.data.fill_(alpha_base * alpha_mlp_mul    * depth)

    # head logit scale
    model.head_logit_scale.data.fill_(0.97)

    return model
