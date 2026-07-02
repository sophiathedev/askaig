#!/usr/bin/env python3
"""Parity check: engine C++ eval  ==  exact integer simulation of the .nnue  ~=  float model.

Three-way comparison over N FENs:
  1. engine vs int-sim: must match EXACTLY (both are the same integer math; any difference is
     an engine or export bug, including the C-style truncating divisions).
  2. float model (checkpoint) vs int-sim: reports the quantization error; gated loosely
     (--max-cp, default mean <= 8). With SCReLU at QB=64 the output-layer rounding alone is
     ~5-10 cp mean (|w_q| <= 127 is the int16 mullo ceiling, so QB cannot go higher) — this
     is the standard, play-irrelevant noise every QB=64 engine accepts, not a bug.

Usage:
  python3 parity.py ../../build/askaig net.nnue book.txt --n 1000 [--ckpt smoke.pt]
  (book.txt = FEN-per-line file; extra fields after the 4 FEN fields are ignored)
"""

import argparse
import random
import struct
import subprocess
import sys

import numpy as np

from convert_fen import PIECE_NIBBLE, parse_board

FEATURES, HL = 768, 256
QA, QB, SCALE = 255, 64, 400


def load_net(path):
    with open(path, "rb") as f:
        raw = f.read()
    magic, ver, feats, hl, buckets, qa, qb, scale, act = struct.unpack("<4sIIIIHHHB", raw[:27])
    assert magic == b"AKNN" and ver == 1 and feats == FEATURES and hl == HL
    assert qa == QA and qb == QB and scale == SCALE and act == 1
    off = 32
    ft_w = np.frombuffer(raw, "<i2", FEATURES * HL, off).reshape(FEATURES, HL).astype(np.int64)
    off += 2 * FEATURES * HL
    ft_b = np.frombuffer(raw, "<i2", HL, off).astype(np.int64)
    off += 2 * HL
    out_w = np.frombuffer(raw, "<i2", 2 * HL, off).astype(np.int64)
    off += 4 * HL
    out_b = struct.unpack_from("<i", raw, off)[0]
    return ft_w, ft_b, out_w, out_b


def features_of(fen):
    """FEN -> (white-persp indices, black-persp indices, white_to_move)."""
    fields = fen.split()
    pieces = parse_board(fields[0])
    w, b = [], []
    for sq, p in pieces.items():
        nib = PIECE_NIBBLE[p]
        col, typ = nib >> 3, nib & 7
        w.append(384 * col + 64 * typ + sq)
        b.append(384 * (col ^ 1) + 64 * typ + (sq ^ 56))
    return w, b, fields[1] == "w"


def tdiv(a, b):
    """C-style truncating integer division (Python // floors; they differ for negatives)."""
    q = abs(a) // b
    return q if a >= 0 else -q


def int_eval(net, fen):
    ft_w, ft_b, out_w, out_b = net
    w_idx, b_idx, wtm = features_of(fen)
    acc_w = ft_b + ft_w[w_idx].sum(axis=0)
    acc_b = ft_b + ft_w[b_idx].sum(axis=0)
    stm, opp = (acc_w, acc_b) if wtm else (acc_b, acc_w)
    v_stm = np.clip(stm, 0, QA)
    v_opp = np.clip(opp, 0, QA)
    s = int((v_stm * v_stm * out_w[:HL]).sum() + (v_opp * v_opp * out_w[HL:]).sum())
    assert abs(s) < 2**31, "int32 overflow - the export-time guarantee is violated"
    return tdiv((tdiv(s, QA) + out_b) * SCALE, QA * QB)


def engine_evals(binary, net, fens):
    import os

    lines = [f"setoption name EvalFile value {os.path.abspath(net)}"]  # not the embedded default!
    for fen in fens:
        lines += [f"position fen {fen}", "eval"]
    lines.append("quit")
    out = subprocess.run([binary], input="\n".join(lines) + "\n", capture_output=True, text=True).stdout
    assert "EvalFile loaded" in out, "engine did not accept the EvalFile - parity would test the embedded net"
    vals = [int(l.split()[1]) for l in out.splitlines() if l.startswith("eval ")]
    assert len(vals) == len(fens), f"engine returned {len(vals)} evals for {len(fens)} positions"
    return vals


def float_evals(ckpt, fens, device="cpu"):
    import torch

    from model import Net

    sd = torch.load(ckpt, map_location="cpu")
    model = Net(sd.get("hl", HL))
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    outs = []
    with torch.no_grad():
        for fen in fens:
            w, b, wtm = features_of(fen)
            stm, opp = (w, b) if wtm else (b, w)
            pad = lambda xs: torch.tensor([xs + [FEATURES] * (32 - len(xs))], dtype=torch.long)
            outs.append(model(pad(stm), pad(opp)).item() * SCALE)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("net")
    ap.add_argument("book", help="FEN-per-line file (extra trailing fields ignored)")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--ckpt", help="also compare against the float checkpoint")
    ap.add_argument("--max-cp", type=float, default=8.0, help="max allowed MEAN |float - int| cp")
    args = ap.parse_args()

    rng = random.Random(42)
    with open(args.book) as f:
        pool = [" ".join(l.split()[:4]) + " 0 1" for l in f if l.strip()]
    fens = rng.sample(pool, min(args.n, len(pool)))

    net = load_net(args.net)
    sim = [int_eval(net, f) for f in fens]
    eng = engine_evals(args.binary, args.net, fens)
    mismatches = [(f, a, b) for f, a, b in zip(fens, eng, sim) if a != b]
    if mismatches:
        for f, a, b in mismatches[:10]:
            print(f"ENGINE {a} != SIM {b}  {f}")
        print(f"FAIL: {len(mismatches)}/{len(fens)} engine-vs-simulation mismatches")
        return 1
    print(f"engine == int-simulation on {len(fens)}/{len(fens)} positions (exact)")

    if args.ckpt:
        flt = float_evals(args.ckpt, fens)
        err = np.abs(np.array(flt) - np.array(sim, dtype=float))
        print(f"float vs quantized: mean {err.mean():.2f} cp, max {err.max():.2f} cp")
        if err.mean() > args.max_cp:
            print(f"FAIL: mean quantization error {err.mean():.2f} > {args.max_cp} cp")
            return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
