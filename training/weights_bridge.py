"""
ANE-ByteGrid-44M — weight bridge between PyTorch and ANE blob format.

The ANE runtime reads `.bin` blobs with a small custom header:
  bytes [0..3]  = version byte at offset 0 (0x01)
  bytes [4..7]  = type byte at offset 4 (0x02 = fp16)
  bytes [64..67]= magic 0xDEADBEEF
  bytes [68]    = data type (0x01 = fp16)
  bytes [72..75]= payload byte count (uint32 LE)
  bytes [80..83]= data offset = 128 (uint32 LE)
  bytes [128..] = raw fp16 data

This matches ane_bg_write_weight_blob() in model.m exactly.

Usage:
    save_weights_to_blobs(model, weight_dir)   # after each optimizer step
    load_weights_from_blobs(model, weight_dir) # to resume training
"""

import struct
import os
import math
from pathlib import Path

import torch
import numpy as np

HIDDEN  = 512
LAYERS  = 24
GLU     = 1024
VOCAB   = 256
INPUT_CHANNELS = 320
PACKED  = HIDDEN * 16   # 8192
BLOCK   = 16


# ── blob I/O ──────────────────────────────────────────────────────────────

def _write_blob(path: str, data_fp16: np.ndarray) -> None:
    """Write a weight blob in the format expected by ane_bg_write_weight_blob()."""
    payload = data_fp16.astype(np.float16).flatten()
    payload_bytes = payload.tobytes()
    header = bytearray(128)
    header[0] = 0x01                                          # version
    header[4] = 0x02                                          # type tag
    struct.pack_into('<I', header, 64, 0xDEADBEEF)            # magic
    header[68] = 0x01                                         # fp16 data type
    struct.pack_into('<I', header, 72, len(payload_bytes))    # payload byte count
    struct.pack_into('<I', header, 80, 128)                   # data offset
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        f.write(header)
        f.write(payload_bytes)


def _read_blob(path: str) -> np.ndarray:
    """Read fp16 data from an ANE weight blob."""
    with open(path, 'rb') as f:
        header = f.read(128)
    payload_bytes = struct.unpack_from('<I', header, 72)[0]
    data_offset   = struct.unpack_from('<I', header, 80)[0]
    with open(path, 'rb') as f:
        f.seek(data_offset)
        raw = f.read(payload_bytes)
    return np.frombuffer(raw, dtype=np.float16).copy()


def _to_fp16(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy().astype(np.float16)


def _from_fp16(arr: np.ndarray, shape: tuple, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).reshape(shape).to(device)


# ── save ──────────────────────────────────────────────────────────────────

def save_weights_to_blobs(model, weight_dir: str) -> None:
    """
    Serialize all trainable parameters from a ByteGridModel instance into
    the .bin blobs that the ANE runtime reads.
    """
    d = weight_dir

    # stem: [HIDDEN=512, INPUT_CHANNELS=320, 1, 1] -> flat [512*320]
    _write_blob(f'{d}/stem.bin', _to_fp16(model.stem.weight.squeeze()))

    for i, block in enumerate(model.blocks):
        prefix = f'{d}/block_{i:02d}'

        # local mixer conv: [PACKED=8192, 1, 1, 1] weight (groups=HIDDEN)
        # PyTorch stores grouped conv weight as [out_ch, in_ch/groups, kH, kW]
        # = [8192, 1, 1, 1] but the ANE blob expects [8192, 16, 1, 1] layout.
        # The PyTorch weight matrix for a grouped conv with groups=512:
        #   out_channels = 8192, in_channels = 8192, groups = 512
        #   weight shape: [8192, 8192//512=16, 1, 1] = [8192, 16, 1, 1]  ✓
        _write_blob(f'{prefix}_local.bin',  _to_fp16(block.local_mixer.conv.weight))
        _write_blob(f'{prefix}_global.bin', _to_fp16(block.global_mixer.conv.weight))

        # GLU projections
        _write_blob(f'{prefix}_wv.bin', _to_fp16(block.channel_glu.wv.weight))
        _write_blob(f'{prefix}_wg.bin', _to_fp16(block.channel_glu.wg.weight))
        _write_blob(f'{prefix}_wo.bin', _to_fp16(block.channel_glu.wo.weight))

        # RMSNorm scale vectors [1, HIDDEN, 1, 1] -> [HIDDEN]
        _write_blob(f'{prefix}_rms_local.bin',  _to_fp16(block.rms_local.squeeze()))
        _write_blob(f'{prefix}_rms_global.bin', _to_fp16(block.rms_global.squeeze()))
        _write_blob(f'{prefix}_rms_ffn.bin',    _to_fp16(block.rms_ffn.squeeze()))

        # alpha scalars [scalar fp16]
        _write_blob(f'{prefix}_alpha_local.bin',  _to_fp16(block.alpha_local.unsqueeze(0)))
        _write_blob(f'{prefix}_alpha_global.bin', _to_fp16(block.alpha_global.unsqueeze(0)))
        _write_blob(f'{prefix}_alpha_mlp.bin',    _to_fp16(block.alpha_mlp.unsqueeze(0)))

    # head
    _write_blob(f'{d}/head_rms.bin',         _to_fp16(model.head_rms.squeeze()))
    _write_blob(f'{d}/head.bin',             _to_fp16(model.head.weight))
    _write_blob(f'{d}/head_logit_scale.bin', _to_fp16(model.head_logit_scale.unsqueeze(0)))


# ── load ──────────────────────────────────────────────────────────────────

def load_weights_from_blobs(model, weight_dir: str) -> None:
    """
    Load weights from .bin blobs back into a ByteGridModel.
    Use this to resume training from a checkpoint or to verify round-trip.
    """
    d = weight_dir
    dev = next(model.parameters()).device

    stem_data = _from_fp16(_read_blob(f'{d}/stem.bin'), (HIDDEN, INPUT_CHANNELS, 1, 1), dev)
    model.stem.weight.data.copy_(stem_data)

    for i, block in enumerate(model.blocks):
        prefix = f'{d}/block_{i:02d}'

        lw = _from_fp16(_read_blob(f'{prefix}_local.bin'),  (HIDDEN * BLOCK, BLOCK, 1, 1), dev)
        gw = _from_fp16(_read_blob(f'{prefix}_global.bin'), (HIDDEN * BLOCK, BLOCK, 1, 1), dev)
        block.local_mixer.conv.weight.data.copy_(lw)
        block.global_mixer.conv.weight.data.copy_(gw)

        block.channel_glu.wv.weight.data.copy_(_from_fp16(_read_blob(f'{prefix}_wv.bin'), (GLU, HIDDEN, 1, 1), dev))
        block.channel_glu.wg.weight.data.copy_(_from_fp16(_read_blob(f'{prefix}_wg.bin'), (GLU, HIDDEN, 1, 1), dev))
        block.channel_glu.wo.weight.data.copy_(_from_fp16(_read_blob(f'{prefix}_wo.bin'), (HIDDEN, GLU, 1, 1), dev))

        block.rms_local.data.copy_( _from_fp16(_read_blob(f'{prefix}_rms_local.bin'),  (1, HIDDEN, 1, 1), dev))
        block.rms_global.data.copy_(_from_fp16(_read_blob(f'{prefix}_rms_global.bin'), (1, HIDDEN, 1, 1), dev))
        block.rms_ffn.data.copy_(   _from_fp16(_read_blob(f'{prefix}_rms_ffn.bin'),    (1, HIDDEN, 1, 1), dev))

        block.alpha_local.data.copy_( _from_fp16(_read_blob(f'{prefix}_alpha_local.bin'),  (), dev))
        block.alpha_global.data.copy_(_from_fp16(_read_blob(f'{prefix}_alpha_global.bin'), (), dev))
        block.alpha_mlp.data.copy_(   _from_fp16(_read_blob(f'{prefix}_alpha_mlp.bin'),    (), dev))

    model.head_rms.data.copy_(_from_fp16(_read_blob(f'{d}/head_rms.bin'), (1, HIDDEN, 1, 1), dev))
    model.head.weight.data.copy_(_from_fp16(_read_blob(f'{d}/head.bin'), (VOCAB, HIDDEN, 1, 1), dev))
    model.head_logit_scale.data.copy_(_from_fp16(_read_blob(f'{d}/head_logit_scale.bin'), (), dev))
