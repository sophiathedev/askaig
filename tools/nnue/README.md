# askaig NNUE training pipeline

Everything needed to produce `networks/default.nnue` — the engine's embedded net (see
`src/nnue.h` for the exact spec):

    (768 x 8 king buckets -> 512)x2 -> 1 of 8 material output buckets, SCReLU  (net v2)

Features are (own-king bucket, piece, square) with kings included and horizontal mirroring
when the perspective's king sits on files e-h; the output head is picked by material count.
Labels: Stockfish evals (the Lichess evaluation DB provides the deepest available analysis
per position, typically depth 18-40+), distilled via sigmoid(cp/400).

## Files

| file             | role |
|------------------|------|
| `convert_fen.py` | FEN books / Lichess eval JSONL → bulletformat `.bf` (32 B/record, stm-normalized); `--check` decode-verifies |
| `shuffle_bin.py` | uniform external-memory shuffle (mandatory before training) |
| `data.py`        | memory-mapped `.bf` reader, vectorized numpy decode → torch batches |
| `model.py`       | the float model, mirrors `src/nnue.cpp` exactly; per-step weight clip ±1.98 |
| `train.py`       | MPS/CPU training loop (Adam, cosine LR, sigmoid-MSE loss, λ WDL blend) |
| `export.py`      | checkpoint → quantized `.nnue` (QA=255 / QB=64 / SCALE=400) with overflow hard-asserts; `--random` emits a plumbing-test net |
| `parity.py`      | engine == exact-int-simulation (must be 0-diff) + float-vs-quantized error report |
| `bullet/`        | **Rust trainer on [bullet](https://github.com/jw1912/bullet)** — Metal on Apple Silicon, the fast path (replaces `train.py`) |
| `bullet_to_nnue.py` | bullet checkpoint `raw.bin` → quantized `.nnue` via `export.write_net` (no permutation; same QA/QB overflow asserts) |

## Recipe

```sh
# 1. data: either convert a local FEN book...
python3 convert_fen.py --format lichess ../work/lichess.book ../work/data.bf
# ...or stream the full Lichess eval DB (394M positions, ~21 GB download):
curl -sL https://database.lichess.org/lichess_db_eval.jsonl.zst | zstd -dc \
  | python3 convert_fen.py --format jsonl - ../work/data.bf
# (bulletformat datasets from other engines need no conversion at all)

# 2. shuffle (uniform, external memory)
python3 shuffle_bin.py ../work/data.bf ../work/data-shuf.bf

# 3. sanity-check the records
python3 convert_fen.py --check ../work/data-shuf.bf 100

# 4. train (MPS on Apple Silicon; --lam 0 for teacher-only data, 0.2-0.4 with real results)
python3 train.py ../work/data-shuf.bf ../work/net.pt --epochs 30 --lam 0.0

# 5. export quantized + verify
python3 export.py ../work/net.pt ../work/net.nnue
python3 parity.py ../../build/askaig ../work/net.nnue ../work/lichess.book --n 1000 --ckpt ../work/net.pt

# 6. ship it
cp ../work/net.nnue ../../networks/default.nnue   # CMake re-embeds on the next build
```

## Faster training: bullet (Rust + Metal)

`train.py` on MPS is slow — Metal handles the sparse feature transformer poorly. `bullet/` is a
Rust trainer on [bullet](https://github.com/jw1912/bullet) running the **same** architecture on the
Apple GPU via bullet's `metal` backend (much faster, identical net). It reproduces `src/nnue.h`
exactly: `ChessBucketsMirrored` + `MaterialCount::<8>` match the engine's feature index and output
bucket bit-for-bit, so no permutation is needed and the data prep is unchanged.

```sh
# 0. one-time: Xcode Command-Line Tools + rustc >= 1.87 (the metal feature needs no CUDA/HIP toolkit)

# 1-3. same convert -> shuffle -> check as above, into ../work/data-shuf.bf

# 4. train on the Metal GPU (config via env; ASKAIG_DATA is the only required one)
cd bullet
ASKAIG_DATA=../../work/data-shuf.bf ASKAIG_SB=160 ASKAIG_LAMBDA=0.0 cargo run --release
# -> checkpoints/askaig-<sb>/{raw.bin,quantised.bin}, saved every save_rate superbatches (+ the last)

# 5. convert the newest checkpoint -> quantized .nnue, then verify (0-diff required)
python3 ../bullet_to_nnue.py checkpoints/askaig-160 ../../work/net.nnue
python3 ../parity.py ../../../build/askaig ../../work/net.nnue ../../work/lichess.book --n 1000

# 6. ship it (from repo root, as above)
cp ../../work/net.nnue ../../../networks/default.nnue
```

Env knobs: `ASKAIG_DATA` (comma-separated `.bf`), `ASKAIG_SB` (superbatches, default 160),
`ASKAIG_BPS` (batches/superbatch, default 6104 ≈ 100M pos), `ASKAIG_BATCH` (16384), `ASKAIG_LAMBDA`
(WDL blend, 0 for teacher-eval data, 0.2–0.4 with real results), `ASKAIG_LR` (0.001),
`ASKAIG_THREADS` (loader decode, 4). The bullet rev is pinned in `bullet/Cargo.toml`.

Engine-side gates (run after any net or kernel change):

```
selftest nnue 1000 80    # incremental accumulator == full refresh, bit-exact
bench evalnps            # eval throughput + cross-build checksum
eval                     # startpos sanity
```

## Sharp edges (all enforced by asserts, listed for context)

- **±1.98 weight clip is load-bearing**: `QA*|w_q| = 255*127 < 32768` is what lets the engine
  compute SCReLU as `mullo(int16) + madd`. `export.py` refuses unclipped checkpoints.
- **MPS `padding_idx` bug**: the embedding backward on MPS updates the padding row despite
  `padding_idx=768`; `model.clip()` re-zeroes it every step and `export.py` asserts it's zero.
- **Records are stm-normalized** ("white" = side to move); `decode()`/`decode_batch` map piece
  nibble type ≥ 6 (unmoved-rook marker some public writers use) to a plain rook.
- Integer division in the final scaling truncates toward zero (C semantics); `parity.py`'s
  simulator replicates that (`tdiv`) — Python's `//` floors and WILL mismatch on negatives.
