"""Memory-mapped bulletformat reader with fully vectorized numpy decoding.

Independent of convert_fen.py's per-record Python codec on purpose: two implementations of
the same format cross-check each other (convert_fen.py --check validates records this loader
then consumes).
"""

import numpy as np
import torch

FEATURES = 768

BF_DTYPE = np.dtype([
    ("occ", "<u8"),
    ("pcs", "u1", 16),
    ("score", "<i2"),
    ("result", "u1"),
    ("ksq", "u1"),
    ("opp_ksq", "u1"),
    ("pad", "u1", 3),
])
assert BF_DTYPE.itemsize == 32


def open_bf(path):
    return np.memmap(path, dtype=BF_DTYPE, mode="r")


def decode_batch(recs):
    """Structured slice (B,) -> (stm_idx, opp_idx (B,32) int64 padded with 768,
    score (B,) f32 stm-relative cp, result (B,) f32 in {0, 0.5, 1} stm-relative)."""
    B = len(recs)

    # Occupied squares, ascending per record: unpack the u64 to (B, 64) bits.
    # (ascontiguousarray: structured-array fields are strided views; .view needs dense bytes)
    occ = np.ascontiguousarray(recs["occ"])
    bits = np.unpackbits(occ.view(np.uint8).reshape(B, 8), axis=1, bitorder="little")
    rows, sqs = np.nonzero(bits)  # row-major -> ascending square order within each record
    counts = bits.sum(axis=1).astype(np.int64)

    # Nibble k of record r describes its k-th occupied square (low nibble first).
    nibs = np.empty((B, 32), np.uint8)
    pcs = recs["pcs"]
    nibs[:, 0::2] = pcs & 0xF
    nibs[:, 1::2] = pcs >> 4
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
    ordinal = np.arange(len(rows)) - np.repeat(starts, counts)
    nib = nibs[rows, ordinal]

    typ = (nib & 7).astype(np.int64)
    col = (nib >> 3).astype(np.int64)
    typ[typ >= 6] = 3  # unmoved-rook marker used by some public writers -> plain rook
    sq = sqs.astype(np.int64)

    stm_feat = 384 * col + 64 * typ + sq  # records are stm-normalized: "white" = side to move
    opp_feat = 384 * (1 - col) + 64 * typ + (sq ^ 56)

    stm_idx = np.full((B, 32), FEATURES, np.int64)
    opp_idx = np.full((B, 32), FEATURES, np.int64)
    stm_idx[rows, ordinal] = stm_feat
    opp_idx[rows, ordinal] = opp_feat

    score = recs["score"].astype(np.float32)
    result = recs["result"].astype(np.float32) / 2.0
    return stm_idx, opp_idx, score, result


class Batches:
    """Sequential batches over a (pre-shuffled) .bf file, as torch tensors on `device`.
    `start`/`stop` bound the record range (used for the train/validation split)."""

    def __init__(self, path, batch_size, device, start=0, stop=None):
        self.arr = open_bf(path)
        self.batch_size = batch_size
        self.device = device
        self.start = start
        self.stop = len(self.arr) if stop is None else stop

    def __len__(self):
        return (self.stop - self.start) // self.batch_size

    def __iter__(self):
        for lo in range(self.start, self.stop - self.batch_size + 1, self.batch_size):
            stm, opp, score, result = decode_batch(self.arr[lo:lo + self.batch_size])
            yield (torch.from_numpy(stm).to(self.device),
                   torch.from_numpy(opp).to(self.device),
                   torch.from_numpy(score).to(self.device),
                   torch.from_numpy(result).to(self.device))
