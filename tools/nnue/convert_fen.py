#!/usr/bin/env python3
"""Convert FEN-text training books to bulletformat ChessBoard records (.bf, 32 bytes each).

Record layout (little-endian, jw1912/bulletformat "ChessBoard"):
  u64 occ         occupied-squares bitboard (a1=bit 0 ... h8=bit 63)
  u8  pcs[16]     one nibble per occupied square in ascending square order, low nibble first;
                  nibble = (color << 3) | type, type P=0..K=5, color 0 = side to move
  i16 score       centipawns, side-to-move relative
  u8  result      0 = stm loss, 1 = draw, 2 = stm win
  u8  ksq         stm king square
  u8  opp_ksq     opponent king square ^ 56
  u8  pad[3]
Records are stm-normalized: when black is to move the board is vertically mirrored, colors
swapped, score negated and result inverted — "white" in a record always means the side to move.

Input formats (--format):
  lichess : "<board> <turn> <castle> <ep> <p>"  p = white-POV win prob (sigmoid K=0.005 of SF cp)
            -> score = logit(p)/K clamped to +-1000 (matching tools/lichess_label.py's mate clamp),
               result derived from p thresholds (only used when training with lambda > 0)
  jsonl   : the raw Lichess eval DB (lichess_db_eval.jsonl.zst, decompressed on stdin or a file):
            one JSON object per line -> deepest eval's principal cp, exact (mates clamped +-1000)
  result  : "<6-field FEN> <r>"  r = white-POV game result 1.0 / 0.5 / 0.0 -> score = 0 (!)
  epd     : '<board> <turn> <castle> <ep> c9 "1-0";'  (Zurichess quiet-labeled style) -> score = 0 (!)
(!) result-only inputs carry no teacher score; train those with lambda = 1.

Usage:
  python3 convert_fen.py --format lichess tools/work/lichess.book tools/work/lichess.bf
  zstd -dc lichess_db_eval.jsonl.zst | python3 convert_fen.py --format jsonl - big.bf
  python3 convert_fen.py --check tools/work/lichess.bf 100   # decode/roundtrip-verify N records
"""

import argparse
import math
import random
import struct
import sys

RECORD = struct.Struct("<Q16shBBB3x")
assert RECORD.size == 32
K = 0.005
CP_CLAMP = 1000  # lichess_label.py clamps mates to +-1000 cp; keep parity

PIECE_NIBBLE = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                "p": 8, "n": 9, "b": 10, "r": 11, "q": 12, "k": 13}
NIBBLE_PIECE = {v: k for k, v in PIECE_NIBBLE.items()}


def parse_board(board_field):
    """FEN board field -> {square: piece char}, a1=0..h8=63."""
    out = {}
    rank, file = 7, 0
    for ch in board_field:
        if ch == "/":
            rank, file = rank - 1, 0
        elif ch.isdigit():
            file += int(ch)
        else:
            out[rank * 8 + file] = ch
            file += 1
    return out


def encode(pieces, white_to_move, cp_white, result_white2):
    """pieces: {sq: char}; cp_white: white-POV cp; result_white2: white-POV result in {0,1,2}."""
    if white_to_move:
        score, result = cp_white, result_white2
    else:  # stm normalization: mirror vertically, swap colors, flip score/result
        pieces = {sq ^ 56: p.swapcase() for sq, p in pieces.items()}
        score, result = -cp_white, 2 - result_white2

    occ, ksq, opp_ksq = 0, 0, 0
    pcs = bytearray(16)
    for i, sq in enumerate(sorted(pieces)):
        p = pieces[sq]
        occ |= 1 << sq
        pcs[i // 2] |= PIECE_NIBBLE[p] << (4 * (i % 2))
        if p == "K":
            ksq = sq
        elif p == "k":
            opp_ksq = sq ^ 56
    score = max(-32000, min(32000, int(round(score))))
    return RECORD.pack(occ, bytes(pcs), score, result, ksq, opp_ksq)


def decode(rec):
    """32-byte record -> (pieces dict, score, result, ksq, opp_ksq). Validates invariants."""
    occ, pcs, score, result, ksq, opp_ksq = RECORD.unpack(rec)
    pieces, i = {}, 0
    for sq in range(64):
        if occ >> sq & 1:
            nb = (pcs[i // 2] >> (4 * (i % 2))) & 0xF
            typ = nb & 7
            if typ >= 6:  # some public writers mark unmoved rooks with type 6 -> plain rook
                nb = (nb & 8) | 3
            pieces[sq] = NIBBLE_PIECE[nb]
            i += 1
    kings = [sq for sq, p in pieces.items() if p == "K"]
    okings = [sq for sq, p in pieces.items() if p == "k"]
    assert len(kings) == 1 and len(okings) == 1, "record must have exactly one king per side"
    assert kings[0] == ksq and okings[0] == (opp_ksq ^ 56), "king squares disagree with pcs"
    assert result in (0, 1, 2)
    return pieces, score, result, ksq, opp_ksq


def pieces_to_board_field(pieces):
    rows = []
    for rank in range(7, -1, -1):
        row, empty = "", 0
        for file in range(8):
            p = pieces.get(rank * 8 + file)
            if p is None:
                empty += 1
            else:
                row += (str(empty) if empty else "") + p
                empty = 0
        rows.append(row + (str(empty) if empty else ""))
    return "/".join(rows)


def parse_jsonl(line):
    """One Lichess eval-DB JSON line -> (board_field, white_to_move, cp_white, result2) or None."""
    import json

    o = json.loads(line)
    fen, evals = o.get("fen"), o.get("evals")
    if not fen or not evals:
        return None
    pv = max(evals, key=lambda e: e.get("depth", 0)).get("pvs", [{}])[0]
    if "cp" in pv:
        cp = pv["cp"]
    elif "mate" in pv:
        cp = CP_CLAMP if pv["mate"] > 0 else -CP_CLAMP
    else:
        return None
    cp = max(-CP_CLAMP, min(CP_CLAMP, cp))  # white-POV on lichess
    t = fen.split()
    res2 = 2 if cp > 160 else (0 if cp < -160 else 1)  # ~sigmoid(0.005*160) = 0.69
    return t[0], t[1] == "w", cp, res2


def convert(args):
    n_in = n_out = 0
    fin = sys.stdin if args.src == "-" else open(args.src)
    with open(args.out, "wb") as fout:
        for line in fin:
            n_in += 1
            t = line.split()
            if not t:
                continue
            try:
                if args.format == "jsonl":
                    parsed = parse_jsonl(line)
                    if parsed is None:
                        continue
                    board, wtm, cp, res2 = parsed
                elif args.format == "lichess":
                    board, wtm, label = t[0], t[1] == "w", float(t[4])
                    p = min(max(label, 1e-6), 1 - 1e-6)
                    cp = max(-CP_CLAMP, min(CP_CLAMP, math.log(p / (1 - p)) / K))
                    res2 = 2 if label > 2 / 3 else (0 if label < 1 / 3 else 1)
                elif args.format == "result":
                    board, wtm, cp = t[0], t[1] == "w", 0.0
                    res2 = round(2 * float(t[6]))
                else:  # epd: ... c9 "1-0";
                    board, wtm, cp = t[0], t[1] == "w", 0.0
                    r = line.split('"')[1]
                    res2 = 2 if r == "1-0" else (0 if r == "0-1" else 1)
                fout.write(encode(parse_board(board), wtm, cp, res2))
                n_out += 1
                if n_out % 5_000_000 == 0:
                    print(f"... {n_out} records", file=sys.stderr)
            except (IndexError, ValueError, KeyError) as e:
                print(f"skip line {n_in}: {e}", file=sys.stderr)
    if fin is not sys.stdin:
        fin.close()
    print(f"{args.src}: {n_in} lines -> {n_out} records ({n_out * 32} bytes) -> {args.out}")


def check(args):
    """Decode N random records; validate invariants + encode/decode roundtrip; print a few FENs."""
    with open(args.src, "rb") as f:
        f.seek(0, 2)
        total = f.tell() // 32
        assert f.tell() % 32 == 0, "file size not a multiple of 32"
        rng = random.Random(42)
        idxs = sorted(rng.sample(range(total), min(int(args.out), total)))
        shown = 0
        for i in idxs:
            f.seek(i * 32)
            rec = f.read(32)
            pieces, score, result, ksq, opp_ksq = decode(rec)
            # roundtrip: records are stm-normalized, so re-encoding as white-to-move must be identity
            # (skip when the record contains type-6 rook markers, which decode() canonicalizes)
            re_rec = encode(pieces, True, score, result)
            assert re_rec == rec or any((rec[8 + j // 2] >> (4 * (j % 2))) & 7 >= 6 for j in range(64)), \
                f"roundtrip mismatch at record {i}"
            if shown < 10:
                print(f"#{i}: {pieces_to_board_field(pieces)} w - - | score(stm) {score} result(stm) {result}")
                shown += 1
        print(f"checked {len(idxs)}/{total} records: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out", help="output .bf (or N for --check)")
    ap.add_argument("--format", choices=["lichess", "result", "epd"], default="lichess")
    ap.add_argument("--check", action="store_true", help="decode-verify N random records of src")
    args = ap.parse_args()
    check(args) if args.check else convert(args)


if __name__ == "__main__":
    main()
