#!/usr/bin/env python3
"""Convert a bullet checkpoint (tools/nnue/bullet) into an askaig .nnue (AKNN v2).

  python3 bullet_to_nnue.py <checkpoint_dir>|<raw.bin> out.nnue        # from raw f32 (default)
  python3 bullet_to_nnue.py --quantised <checkpoint_dir>|<quantised.bin> out.nnue

Default (raw.bin): reads bullet's raw f32 weights, reshapes them into the engine's
(ft_w, ft_b, out_w, out_b) and hands them to export.write_net — so QA/QB quantization and the
int16-overflow asserts are identical to the PyTorch path (one source of truth). No feature or
bucket permutation is needed: ChessBucketsMirrored + MaterialCount::<8> reproduce the engine's
`idx = 768*bucket + 384*enemy + 64*pt + oriented(sq)` and `(popcount(occ)-2)/4` exactly. The
only reshape is transposing l1w from bullet's input-major [1024][8] to out_w [8][1024].

--quantised: bullet already emitted the AKNN body (same layout) in quantised.bin; this just
slices it and prepends the 32-byte header. Handy as a cross-check against the raw path.
"""

import argparse
import os
import struct
import sys

import numpy as np

from export import FEATURES, HL, OUT_BUCKETS, QA, QB, SCALE, write_net

L0W = FEATURES * HL  # 6144 * 512 feature-major
L0B = HL  # 512
L1W = OUT_BUCKETS * 2 * HL  # 8 * 1024
L1B = OUT_BUCKETS  # 8


def from_raw(path):
    """bullet raw.bin = f32 [l0w, l0b, l1w, l1b], native column-major, no transforms."""
    raw = np.fromfile(path, dtype="<f4")
    want = L0W + L0B + L1W + L1B
    assert raw.size == want, f"{path}: {raw.size} f32 values, expected {want} (architecture mismatch?)"
    o = 0
    l0w = raw[o:o + L0W]; o += L0W
    l0b = raw[o:o + L0B]; o += L0B
    l1w = raw[o:o + L1W]; o += L1W
    l1b = raw[o:o + L1B]
    ft_w = l0w.reshape(FEATURES, HL).astype(np.float32)  # [feature][hidden] — already AKNN order
    ft_b = l0b.astype(np.float32)  # [hidden]
    # l1w is bucket-fast [1024][8] (flat j*8+k); transpose to out_w[bucket][input] = [8][1024].
    out_w = l1w.reshape(2 * HL, OUT_BUCKETS).T.copy().astype(np.float32)
    out_b = l1b.astype(np.float32)  # [8]
    return ft_w, ft_b, out_w, out_b


def from_quantised(path, out):
    """quantised.bin = AKNN body (i16 ft_w, i16 ft_b, i16 out_w, i32 out_b) + b"bullet" pad."""
    body = 2 * (FEATURES * HL + HL + OUT_BUCKETS * 2 * HL) + 4 * OUT_BUCKETS
    data = open(path, "rb").read()
    assert len(data) >= body, f"{path}: {len(data)} bytes < {body} expected"
    header = struct.pack("<4sIIIIHHHB5x", b"AKNN", 2, FEATURES, HL, OUT_BUCKETS, QA, QB, SCALE, 1)
    assert len(header) == 32
    with open(out, "wb") as f:
        f.write(header)
        f.write(data[:body])
    print(f"wrote {out} ({32 + body} bytes) from {path}")


def resolve(path, name):
    return path if os.path.isfile(path) else os.path.join(path, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="bullet checkpoint dir (or a direct raw.bin/quantised.bin path)")
    ap.add_argument("out", help="output .nnue path")
    ap.add_argument("--quantised", action="store_true", help="convert from quantised.bin instead of raw.bin")
    args = ap.parse_args()

    if args.quantised:
        from_quantised(resolve(args.src, "quantised.bin"), args.out)
    else:
        write_net(args.out, *from_raw(resolve(args.src, "raw.bin")))


if __name__ == "__main__":
    sys.exit(main())
