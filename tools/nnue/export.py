#!/usr/bin/env python3
"""Quantize a trained checkpoint (train.py output) into an askaig .nnue file (v2).

Usage:
  python3 export.py checkpoint.pt out.nnue        # quantize a trained model
  python3 export.py --random out.nnue [--seed N]  # random plumbing-test net

File layout (little-endian, must match src/nnue.h):
  32-byte header: magic "AKNN", u32 version=2, u32 features=6144, u32 hl, u32 buckets=8,
                  u16 qa=255, u16 qb=64, u16 scale=400, u8 activation=1 (SCReLU), u8 pad[5]
  int16 ft_w[6144][HL] (feature-major), int16 ft_b[HL],
  int16 out_w[8][2*HL], int32 out_b[8]

Quantization: ft int16 = round(f*QA); out_w int16 = round(f*QB); out_b int32 = round(f*QA*QB).
The trainer clips floats to +-1.98 so |out_w_q| <= 127 (needed: QA*|w| must fit int16 for the
engine's mullo+madd SCReLU) and |ft_w_q| <= 505. Both bounds are hard-asserted here.
"""

import argparse
import struct
import sys

import numpy as np

KING_BUCKETS = 8
FEATURES = KING_BUCKETS * 768  # 6144
HL = 512
OUT_BUCKETS = 8
QA, QB, SCALE = 255, 64, 400
CLIP = 1.98  # must match model.py; 1.98*QB = 126.7 -> |out_w_q| <= 127, QA*127 < 32768


def write_net(path, ft_w, ft_b, out_w, out_b):
    """ft_w (FEATURES, HL), ft_b (HL,), out_w (OUT_BUCKETS, 2*HL), out_b (OUT_BUCKETS,)."""
    assert ft_w.shape == (FEATURES, HL) and ft_b.shape == (HL,)
    assert out_w.shape == (OUT_BUCKETS, 2 * HL) and out_b.shape == (OUT_BUCKETS,)

    ft_w_q = np.round(ft_w * QA).astype(np.int64)
    ft_b_q = np.round(ft_b * QA).astype(np.int64)
    out_w_q = np.round(out_w * QB).astype(np.int64)
    out_b_q = np.round(out_b.astype(np.float64) * QA * QB).astype(np.int64)

    # Load-bearing overflow guarantees (see src/nnue.cpp dot_reduce / acc int16 headroom).
    assert np.abs(out_w_q).max() <= 127, f"|out_w_q| max {np.abs(out_w_q).max()} > 127 - clip violated"
    assert np.abs(ft_w_q).max() <= round(CLIP * QA) + 1, "ft weight clip violated"
    assert np.abs(ft_b_q).max() < 32768 and np.abs(ft_w_q).max() < 32768
    assert np.abs(out_b_q).max() < 2**31

    header = struct.pack("<4sIIIIHHHB5x", b"AKNN", 2, FEATURES, HL, OUT_BUCKETS, QA, QB, SCALE, 1)
    assert len(header) == 32
    with open(path, "wb") as f:
        f.write(header)
        f.write(ft_w_q.astype("<i2").tobytes())
        f.write(ft_b_q.astype("<i2").tobytes())
        f.write(out_w_q.astype("<i2").tobytes())
        f.write(out_b_q.astype("<i4").tobytes())
    size = 32 + 2 * (FEATURES * HL + HL + OUT_BUCKETS * 2 * HL) + 4 * OUT_BUCKETS
    print(f"wrote {path} ({size} bytes)")


def random_net(seed):
    """Small random weights: a nonsense but well-formed net for plumbing tests."""
    rng = np.random.default_rng(seed)
    ft_w = rng.uniform(-0.05, 0.05, (FEATURES, HL)).astype(np.float32)
    ft_b = rng.uniform(0.0, 0.1, HL).astype(np.float32)
    out_w = rng.uniform(-0.5, 0.5, (OUT_BUCKETS, 2 * HL)).astype(np.float32)
    out_b = np.zeros(OUT_BUCKETS, np.float32)
    return ft_w, ft_b, out_w, out_b


def from_checkpoint(path):
    import torch

    sd = torch.load(path, map_location="cpu")
    if "model" in sd:
        sd = sd["model"]
    # model.py: ft = nn.Embedding(FEATURES+1, HL) (row f = feature f, last row = zero padding),
    #           ft_bias = (HL,), out = nn.Linear(2*HL, OUT_BUCKETS)
    ft_w = sd["ft.weight"].numpy()[:FEATURES]  # (FEATURES, HL), already feature-major
    # The padding row is dropped here, so it MUST be zero — MPS's embedding backward ignores
    # padding_idx and lets it drift unless model.clip() re-pins it every step.
    assert np.abs(sd["ft.weight"].numpy()[FEATURES]).max() == 0, \
        "padding row nonzero (MPS padding_idx bug) - retrain with the current model.clip()"
    ft_b = sd["ft_bias"].numpy()
    out_w = sd["out.weight"].numpy()  # (OUT_BUCKETS, 2*HL)
    out_b = sd["out.bias"].numpy()  # (OUT_BUCKETS,)
    assert np.abs(ft_w).max() <= CLIP + 1e-6 and np.abs(out_w).max() <= CLIP + 1e-6, "checkpoint not clipped"
    return ft_w.astype(np.float32), ft_b.astype(np.float32), out_w.astype(np.float32), out_b.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", help="checkpoint .pt (omit with --random)")
    ap.add_argument("out", help="output .nnue path")
    ap.add_argument("--random", action="store_true", help="emit a random test net")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.random:
        parts = random_net(args.seed)
    else:
        if not args.src:
            ap.error("checkpoint path required without --random")
        parts = from_checkpoint(args.src)
    write_net(args.out, *parts)


if __name__ == "__main__":
    sys.exit(main())
