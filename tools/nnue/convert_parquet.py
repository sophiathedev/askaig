#!/usr/bin/env python3
"""Convert the official Lichess evaluations dataset on Hugging Face to bulletformat shards.

Dataset: Lichess/chess-position-evaluations — 944M rows (fen, line, depth, knodes, cp, mate),
one row per (position, eval); a position appears once per stored depth, ADJACENT rows (the
parquet is flattened from the per-position JSONL), so a consecutive-fen dedup keeping the
deepest row recovers "one best eval per position" exactly.

Per shard: download (HF CDN, fast) -> read only fen/cp/mate/depth columns -> dedup -> encode
(stm-normalized bulletformat via convert_fen.encode) -> <outdir>/data_XXXX.bf (tmp+rename, so
finished shards are skipped on resume) -> delete the parquet.

Usage: python3 convert_parquet.py <outdir> [--shards N] [--dl DIR]
Then:  cat <outdir>/*.bf > full.bf && shuffle_bin.py full.bf full-shuf.bf
"""

import argparse
import os
import sys
import time

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

from convert_fen import CP_CLAMP, encode, parse_board

REPO = "Lichess/chess-position-evaluations"


def row_cp(cp, mate):
    """White-POV cp from a (cp, mate) pair; None if the row carries neither."""
    if cp is None:
        if mate is None:
            return None
        return CP_CLAMP if mate > 0 else -CP_CLAMP
    return max(-CP_CLAMP, min(CP_CLAMP, cp))


def convert_shard(parquet_path, out_path):
    pf = pq.ParquetFile(parquet_path)
    n_rows = n_recs = n_skip = 0
    pending = None  # (fen, depth, cp) — deepest row of the current consecutive-fen run

    with open(out_path, "wb") as fout:

        def emit(fen, cp):
            nonlocal n_recs, n_skip
            try:  # the eval DB contains a few corrupt FENs (33+ pieces) — skip, don't die
                t = fen.split()
                res2 = 2 if cp > 160 else (0 if cp < -160 else 1)
                rec = encode(parse_board(t[0]), t[1] == "w", cp, res2)
            except (IndexError, ValueError, KeyError):
                n_skip += 1
                return
            fout.write(rec)
            n_recs += 1

        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["fen", "cp", "mate", "depth"])
            for fen, cp, mate, depth in zip(
                tbl.column("fen").to_pylist(), tbl.column("cp").to_pylist(),
                tbl.column("mate").to_pylist(), tbl.column("depth").to_pylist()):
                n_rows += 1
                c = row_cp(cp, mate)
                if c is None:
                    continue
                if pending is not None and pending[0] == fen:
                    if depth > pending[1]:
                        pending = (fen, depth, c)
                else:
                    if pending is not None:
                        emit(pending[0], pending[2])
                    pending = (fen, depth, c)
        if pending is not None:
            emit(pending[0], pending[2])
    if n_skip:
        print(f"  ({n_skip} corrupt rows skipped)", flush=True)
    return n_rows, n_recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--shards", type=int, default=0, help="convert only the first N shards (0 = all)")
    ap.add_argument("--dl", default=None, help="parquet download dir (default <outdir>/dl)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dl = args.dl or os.path.join(args.outdir, "dl")
    os.makedirs(dl, exist_ok=True)

    shards = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
    if args.shards:
        shards = shards[: args.shards]

    total = 0
    for i, shard in enumerate(shards):
        out = os.path.join(args.outdir, os.path.basename(shard).replace(".parquet", ".bf"))
        if os.path.exists(out):
            total += os.path.getsize(out) // 32
            print(f"[{i + 1}/{len(shards)}] {shard}: already done, skipping", flush=True)
            continue
        t0 = time.time()
        path = hf_hub_download(REPO, shard, repo_type="dataset", local_dir=dl)
        t1 = time.time()
        n_rows, n_recs = convert_shard(path, out + ".tmp")
        os.rename(out + ".tmp", out)
        os.remove(path)
        total += n_recs
        print(f"[{i + 1}/{len(shards)}] {shard}: {n_rows} rows -> {n_recs} records "
              f"(dl {t1 - t0:.0f}s, convert {time.time() - t1:.0f}s, total {total / 1e6:.1f}M)", flush=True)
    print(f"DONE: {total} records in {args.outdir}")


if __name__ == "__main__":
    sys.exit(main())
