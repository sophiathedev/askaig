#!/usr/bin/env python3
"""
PyTorch NNUE trainer for askaig (HalfKA + horizontal-mirror, single king bucket, L1=1024, SCReLU).

Runs on CUDA (Colab), Apple MPS (M-series), or CPU — device auto-detected. Reads the engine's datagen
text (`FEN | score_cp_stm | wdl_stm`), trains, and exports a `.nnue` file whose bytes match the engine's
`struct Network` exactly (see src/nnue/net_format.md). Everything here MUST stay in sync with
src/nnue/network.h and src/nnue/features.h — the `--check` mode replicates the engine's INTEGER forward
in Python so you can confirm `askaig`'s eval == this for the same net & FEN (the contract gate).

Usage:
  python train.py --data all.txt --out askaig-v1.nnue --epochs 30
  python train.py --check askaig-v1.nnue "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"

Local M3:  device auto = mps.   Colab: select a GPU runtime, device auto = cuda.
"""
import argparse
import struct
import sys
import numpy as np

# --- contract constants (MUST match src/nnue/network.h) -----------------------------------------
INPUT  = 768          # 12 piece-planes x 64
L1     = 1024         # hidden width per perspective
QA     = 255          # FT / activation scale
QB     = 64           # output-weight scale
SCALE  = 400          # output -> centipawn
MAX_FEATS = 32        # at most 32 pieces on the board

PIECE_FROM_CHAR = {c: i for i, c in enumerate("PNBRQK")}  # type 0..5


# --- feature extraction (MUST match src/nnue/features.h feature_index) ---------------------------
def parse_fen(fen):
    """Returns (pieces, white_king_sq, black_king_sq, stm) where pieces = list of (color, type, sq).
    Squares are a1=0..h8=63 (engine convention)."""
    board, side = fen.split()[0], fen.split()[1]
    pieces, wk, bk = [], None, None
    sq = 56  # a8
    for ch in board:
        if ch == '/':
            sq -= 16
        elif ch.isdigit():
            sq += int(ch)
        else:
            color = 0 if ch.isupper() else 1            # WHITE=0, BLACK=1
            ptype = PIECE_FROM_CHAR[ch.upper()]
            pieces.append((color, ptype, sq))
            if ptype == 5:                               # KING
                if color == 0: wk = sq
                else:          bk = sq
            sq += 1
    return pieces, wk, bk, (0 if side == 'w' else 1)


def features_for(perspective, pieces, wk, bk):
    """HalfKA + mirror feature indices for one perspective (0=WHITE,1=BLACK). Mirrors features.h."""
    king = wk if perspective == 0 else bk
    rel_king = king if perspective == 0 else (king ^ 56)
    mirror = (rel_king & 7) >= 4
    out = []
    for color, ptype, s in pieces:
        rel_sq = s if perspective == 0 else (s ^ 56)
        if mirror:
            rel_sq ^= 7
        rel_color = 0 if color == perspective else 1
        plane = rel_color * 6 + ptype
        out.append(plane * 64 + rel_sq)
    return out


def encode_line(line):
    """`FEN | score | wdl` -> (us_feats, them_feats, score, wdl) with us = side to move."""
    fen, score, wdl = (p.strip() for p in line.split('|'))
    pieces, wk, bk, stm = parse_fen(fen)
    wf = features_for(0, pieces, wk, bk)
    bf = features_for(1, pieces, wk, bk)
    us, them = (wf, bf) if stm == 0 else (bf, wf)
    return us, them, float(score), float(wdl)


# --- data loading (with an .npz cache so re-runs skip FEN parsing) ------------------------------
def load_data(path, cache):
    if cache and __import__('os').path.exists(cache):
        print(f"loading cache {cache}")
        z = np.load(cache)
        return z['us'], z['them'], z['score'], z['wdl']

    print(f"parsing {path} ...")
    us_rows, them_rows, scores, wdls = [], [], [], []
    pad = INPUT  # padding index -> the zeroed pad row of the embedding
    with open(path) as f:
        for n, line in enumerate(f):
            if '|' not in line:
                continue
            us, them, score, wdl = encode_line(line)
            u = (us + [pad] * MAX_FEATS)[:MAX_FEATS]
            t = (them + [pad] * MAX_FEATS)[:MAX_FEATS]
            us_rows.append(u); them_rows.append(t); scores.append(score); wdls.append(wdl)
            if (n + 1) % 500000 == 0:
                print(f"  {n+1} positions")
    us = np.asarray(us_rows, dtype=np.int16)
    them = np.asarray(them_rows, dtype=np.int16)
    score = np.asarray(scores, dtype=np.float32)
    wdl = np.asarray(wdls, dtype=np.float32)
    print(f"parsed {len(score)} positions")
    if cache:
        np.savez(cache, us=us, them=them, score=score, wdl=wdl)
        print(f"cached -> {cache}")
    return us, them, score, wdl


# --- model --------------------------------------------------------------------------------------
def build_model(torch):
    import torch.nn as nn

    class NNUE(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(INPUT + 1, L1, padding_idx=INPUT)  # +1 zeroed pad row
            self.ft_bias = nn.Parameter(torch.zeros(L1))
            self.out = nn.Linear(2 * L1, 1)
            nn.init.normal_(self.emb.weight, std=0.01)
            with torch.no_grad():
                self.emb.weight[INPUT].zero_()

        def forward(self, us, them):
            au = self.emb(us).sum(1) + self.ft_bias       # [B, L1]
            at = self.emb(them).sum(1) + self.ft_bias
            x = torch.cat([torch.clamp(au, 0, 1) ** 2, torch.clamp(at, 0, 1) ** 2], 1)
            return self.out(x).squeeze(1)                 # out_f ; cp = SCALE * out_f

    return NNUE()


# --- export to .nnue (struct Network byte layout) ----------------------------------------------
def export_nnue(model, path):
    import torch
    ftw = model.emb.weight.detach().cpu()[:INPUT]          # [INPUT, L1] row=feature (feature-major)
    ftb = model.ft_bias.detach().cpu()
    ow  = model.out.weight.detach().cpu()[0]               # [2*L1] = [us L1][them L1]
    ob  = model.out.bias.detach().cpu()[0]

    ftw_q = torch.round(ftw * QA).to(torch.int32)
    ftb_q = torch.round(ftb * QA).to(torch.int32)
    ow_q  = torch.round(ow * QB).to(torch.int32)
    ob_q  = int(round(float(ob) * QA * QB))

    def clamp16(t, name):
        lo, hi = int(t.min()), int(t.max())
        if lo < -32768 or hi > 32767:
            print(f"WARNING: {name} out of int16 range [{lo},{hi}] — net may be mis-scaled", file=sys.stderr)
        return t.clamp(-32768, 32767).to(torch.int16)

    ftw_q = clamp16(ftw_q, "ft_weights")
    ftb_q = clamp16(ftb_q, "ft_bias")
    ow_q  = clamp16(ow_q,  "out_weights")

    with open(path, 'wb') as f:
        f.write(ftw_q.numpy().astype('<i2').tobytes())     # C-order = ft_weights[feat*L1 + i]
        f.write(ftb_q.numpy().astype('<i2').tobytes())
        f.write(ow_q.numpy().astype('<i2').tobytes())
        f.write(struct.pack('<i', ob_q))
    nbytes = INPUT * L1 * 2 + L1 * 2 + 2 * L1 * 2 + 4
    print(f"exported {path} ({nbytes} bytes; max|ftw|={int(ftw_q.abs().max())})")


# --- integer forward in Python (EXACT replica of engine nnue.cpp) -> the contract gate ----------
def _tdiv(a, b):
    """C-style integer division: truncate toward zero (Python // floors toward -inf)."""
    a, b = int(a), int(b)
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def check_int(net_path, fen):
    raw = open(net_path, 'rb').read()
    off = 0
    ftw = np.frombuffer(raw, '<i2', INPUT * L1, off).reshape(INPUT, L1); off += INPUT * L1 * 2
    ftb = np.frombuffer(raw, '<i2', L1, off);                            off += L1 * 2
    ow  = np.frombuffer(raw, '<i2', 2 * L1, off);                        off += 2 * L1 * 2
    ob  = struct.unpack_from('<i', raw, off)[0]

    pieces, wk, bk, stm = parse_fen(fen)
    accs = []
    for P in (0, 1):
        a = ftb.astype(np.int64).copy()
        for f in features_for(P, pieces, wk, bk):
            a += ftw[f]
        accs.append(a)
    us, them = (accs[stm], accs[1 - stm])
    wu, wt = ow[:L1], ow[L1:]
    xs = np.clip(us, 0, QA); xt = np.clip(them, 0, QA)
    s = int(np.sum(xs.astype(np.int64) ** 2 * wu) + np.sum(xt.astype(np.int64) ** 2 * wt))
    out = _tdiv(s, QA) + ob
    cp = _tdiv(out * SCALE, QA * QB)
    print(f"int eval (engine-replica): {cp} cp  [compare to: ./askaig 'position fen {fen}' then 'eval']")
    return cp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data')
    ap.add_argument('--out', default='askaig-v1.nnue')
    ap.add_argument('--cache', default=None, help='.npz parsed-data cache (default: <data>.npz)')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=16384)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wdl', type=float, default=0.5, help='lambda: score vs game-result blend')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--check', nargs=2, metavar=('NET', 'FEN'), help='int-eval a FEN with a .nnue and exit')
    args = ap.parse_args()

    if args.check:
        check_int(args.check[0], args.check[1])
        return

    import torch
    dev = args.device
    if dev == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"device: {dev}")

    cache = args.cache or (args.data + '.npz')
    us_np, them_np, score_np, wdl_np = load_data(args.data, cache)
    N = len(score_np)

    us_t   = torch.from_numpy(us_np.astype(np.int64))
    them_t = torch.from_numpy(them_np.astype(np.int64))
    # training target = lambda*sigmoid(score/SCALE) + (1-lambda)*wdl  (win-prob space)
    target = args.wdl * torch.sigmoid(torch.from_numpy(score_np) / SCALE) + (1 - args.wdl) * torch.from_numpy(wdl_np)

    model = build_model(torch).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for ep in range(args.epochs):
        perm = torch.randperm(N)
        total = 0.0
        for i in range(0, N, args.batch):
            idx = perm[i:i + args.batch]
            u = us_t[idx].to(dev); t = them_t[idx].to(dev); y = target[idx].to(dev)
            pred = torch.sigmoid(model(u, t))
            loss = ((pred - y) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        sched.step()
        print(f"epoch {ep+1}/{args.epochs}  loss {total/N:.6f}  lr {sched.get_last_lr()[0]:.2e}")
        export_nnue(model, args.out)   # checkpoint each epoch (also the final net)

    print(f"done -> {args.out}")


if __name__ == '__main__':
    main()
