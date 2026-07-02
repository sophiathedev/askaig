#!/usr/bin/env python3
"""Uniform external-memory shuffle of .bf files (32-byte records), for datasets >> RAM.

Two passes: (1) stream the input in blocks and scatter each record to a random bucket file;
(2) load each bucket (sized to fit RAM), shuffle it in memory, append to the output. Bucket
assignment is uniform per record, so the composition is a uniform shuffle of the whole file.

Pre-shuffling matters: source datasets are typically ordered (by game, by source shard), and
the trainer reads sequentially — an unshuffled file trains on correlated batches.

Usage: python3 shuffle_bin.py in.bf out.bf [--bucket-mb 1024] [--seed 42] [--tmp DIR]
"""

import argparse
import os
import tempfile

import numpy as np

REC = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--bucket-mb", type=int, default=1024, help="max bucket size (= pass-2 RAM use)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tmp", default=None, help="bucket directory (default: alongside the output)")
    args = ap.parse_args()

    size = os.path.getsize(args.src)
    assert size % REC == 0, "input size not a multiple of the record size"
    total = size // REC
    nbuckets = max(1, (size + args.bucket_mb * (1 << 20) - 1) // (args.bucket_mb * (1 << 20)) * 2)
    rng = np.random.default_rng(args.seed)
    print(f"{args.src}: {total} records ({size >> 20} MiB) -> {nbuckets} buckets")

    tmpdir = tempfile.mkdtemp(dir=args.tmp or os.path.dirname(os.path.abspath(args.out)))
    buckets = [open(os.path.join(tmpdir, f"b{i}.tmp"), "wb") for i in range(nbuckets)]
    try:
        # Pass 1: scatter, streaming in ~256 MiB blocks.
        block = (256 << 20) // REC
        with open(args.src, "rb") as f:
            while True:
                buf = f.read(block * REC)
                if not buf:
                    break
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(-1, REC)
                assign = rng.integers(0, nbuckets, len(arr))
                for b in range(nbuckets):
                    sel = arr[assign == b]
                    if len(sel):
                        buckets[b].write(sel.tobytes())
        for b in buckets:
            b.close()

        # Pass 2: shuffle each bucket in RAM, append to the output.
        written = 0
        with open(args.out, "wb") as out:
            for i in range(nbuckets):
                path = os.path.join(tmpdir, f"b{i}.tmp")
                arr = np.fromfile(path, dtype=np.uint8).reshape(-1, REC)
                rng.shuffle(arr, axis=0)
                out.write(arr.tobytes())
                written += len(arr)
                os.remove(path)
        assert written == total, f"record count changed: {written} != {total}"
        print(f"wrote {args.out}: {written} records, uniformly shuffled")
    finally:
        for b in buckets:
            if not b.closed:
                b.close()
        for name in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, name))
        os.rmdir(tmpdir)


if __name__ == "__main__":
    main()
