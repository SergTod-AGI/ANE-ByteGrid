"""
ANE-ByteGrid-44M — streaming dataset for training.

Dataset: FineWeb-Edu (sample-10BT subset, ~10B tokens of high-quality web text).
Chosen because:
  - high-quality filtered Common Crawl text (educational content filter)
  - 10B bytes ≈ 39M training windows of 256 bytes — enough for Stage A+B
  - streaming-ready via HuggingFace datasets, no full download needed
  - diverse UTF-8 text: articles, books, educational pages

Usage:
    stream = ByteWindowStream(split='train', seed=42)
    for window in stream:            # window: LongTensor [256]
        ...
"""

import random
from typing import Iterator

import torch
from datasets import load_dataset

SEQ = 256


class ByteWindowStream:
    """
    Streams 256-byte windows from FineWeb-Edu.

    Text is encoded as UTF-8, chunked into fixed SEQ=256 windows.
    Short documents are accumulated in a buffer to avoid waste.
    Windows that are shorter than SEQ are right-padded with 0x00.
    """

    DATASET_NAME   = "HuggingFaceFW/fineweb-edu"
    DATASET_CONFIG = "sample-10BT"

    def __init__(
        self,
        split: str = "train",
        seed: int = 42,
        buffer_bytes: int = 1 << 20,   # 1 MB byte buffer before yielding windows
        device: torch.device | None = None,
    ):
        self.split        = split
        self.seed         = seed
        self.buffer_bytes = buffer_bytes
        self.device       = device or torch.device("cpu")

    def __iter__(self) -> Iterator[torch.Tensor]:
        ds = load_dataset(
            self.DATASET_NAME,
            name=self.DATASET_CONFIG,
            split=self.split,
            streaming=True,
            trust_remote_code=False,
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=10_000)

        buf = bytearray()
        for example in ds:
            text: str = example.get("text", "")
            if not text:
                continue
            try:
                raw = text.encode("utf-8", errors="replace")
            except Exception:
                continue
            buf.extend(raw)

            # yield full SEQ-byte windows from the buffer
            while len(buf) >= SEQ:
                window_bytes = bytes(buf[:SEQ])
                del buf[:SEQ]
                arr = torch.tensor(
                    list(window_bytes), dtype=torch.long, device=self.device
                )
                yield arr

        # yield final padded window if non-empty
        if buf:
            window_bytes = bytes(buf) + b"\x00" * (SEQ - len(buf))
            arr = torch.tensor(
                list(window_bytes[:SEQ]), dtype=torch.long, device=self.device
            )
            yield arr


def make_dataloader(
    split: str = "train",
    seed: int = 42,
    device: torch.device | None = None,
) -> ByteWindowStream:
    """Return a ByteWindowStream ready to iterate."""
    return ByteWindowStream(split=split, seed=seed, device=device)
