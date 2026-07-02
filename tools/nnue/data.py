"""Memory-mapped bulletformat reader with fully vectorized numpy decoding.

Produces king-bucketed, horizontally-mirrored feature indices (kings included) matching
src/nnue.cpp's feature_index exactly, plus the material output bucket per position.

Independent of convert_fen.py's per-record Python codec on purpose: two implementations of
the same format cross-check each other.
"""

import numpy as np
import torch

KING_BUCKETS = 8
FEATURES = KING_BUCKETS * 768  # 6144

# Must match nnue::KING_BUCKET in src/nnue.h (indexed by the mirrored, persp-oriented square).
KING_BUCKET = np.array([
    0, 1, 2, 3, 3, 2, 1, 0,
    4, 4, 5, 5, 5, 5, 4, 4,
    6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
], dtype=np.int64)

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


def king_ctx(oriented_ksq):
    """(bucket, mirror-xor-mask) per record from a perspective-oriented king square array."""
    mir = ((oriented_ksq & 7) >= 4).astype(np.int64) * 7  # 0 or 7, xor-mask form
    bucket = KING_BUCKET[oriented_ksq ^ mir]
    return bucket, mir


def decode_batch(recs):
    """Structured slice (B,) -> (stm_idx, opp_idx (B,32) int64 padded with FEATURES,
    score (B,) f32 stm-relative cp, result (B,) f32 in {0,0.5,1}, obkt (B,) int64)."""
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

    # King contexts. Records are stm-normalized: "white" = side to move; ksq is the stm king
    # (stm-persp-oriented) and opp_ksq is the opponent king already ^56'd (opp-persp-oriented).
    b_s, m_s = king_ctx(recs["ksq"].astype(np.int64))
    b_o, m_o = king_ctx(recs["opp_ksq"].astype(np.int64))
    b_s, m_s = b_s[rows], m_s[rows]  # broadcast per piece
    b_o, m_o = b_o[rows], m_o[rows]

    stm_feat = 768 * b_s + 64 * (6 * col + typ) + (sq ^ m_s)
    opp_feat = 768 * b_o + 64 * (6 * (1 - col) + typ) + ((sq ^ 56) ^ m_o)

    stm_idx = np.full((B, 32), FEATURES, np.int64)
    opp_idx = np.full((B, 32), FEATURES, np.int64)
    stm_idx[rows, ordinal] = stm_feat
    opp_idx[rows, ordinal] = opp_feat

    score = recs["score"].astype(np.float32)
    result = recs["result"].astype(np.float32) / 2.0
    obkt = (counts - 2) // 4
    return stm_idx, opp_idx, score, result, obkt


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
            stm, opp, score, result, obkt = decode_batch(self.arr[lo:lo + self.batch_size])
            yield (torch.from_numpy(stm).to(self.device),
                   torch.from_numpy(opp).to(self.device),
                   torch.from_numpy(score).to(self.device),
                   torch.from_numpy(result).to(self.device),
                   torch.from_numpy(obkt).to(self.device))
