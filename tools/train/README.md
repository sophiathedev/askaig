# askaig NNUE trainer (PyTorch)

Trains the v1 net (HalfKA + horizontal-mirror, single king bucket, **L1=1024, SCReLU**) and exports a
`.nnue` file whose bytes match the engine's `struct Network` exactly (see
[`src/nnue/net_format.md`](../../src/nnue/net_format.md)).

Runs on **CUDA** (Colab GPU), **Apple MPS** (M-series), or **CPU** — auto-detected. Designed for the
"datagen on your Mac → train on a Colab GPU → run the net locally" workflow, but it trains anywhere.

The constants at the top of `train.py` (`INPUT/L1/QA/QB/SCALE`) **must stay equal to**
`src/nnue/network.h`. They are already in sync; the contract check below proves it.

## 0. Contract check (do this once — it's already verified, but re-run after any change)

```bash
# dump the engine's current net to a .nnue, then int-eval the SAME net in Python:
./build/askaig nnuedump /tmp/net.nnue
./build/askaig nnuetest | grep ok                                   # engine evals
python3 tools/train/train.py --check /tmp/net.nnue "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - -"
```
The Python `int eval` must equal the engine's eval for the same FEN. If it does, the feature indexing +
byte layout + integer forward all agree — nets you export will load correctly.

## 1. Generate data (locally on the Mac — CPU is fine)

NNUE needs LOTS of data — aim for **tens of millions** of positions, not a million (a small set
overfits badly). Run parallel shards with **distinct seeds** (`datagen <out> <games> <depth> <seed>
[nodes]`):

```bash
# 8 shards, distinct seeds. `nodes`>0 (5th-ish arg) searches a fixed node count per move — uniform and
# faster than fixed depth; 5000 nodes is a good datagen budget. (Use depth instead by dropping the last
# arg, e.g. depth 8.)
for i in $(seq 1 8); do ./build/askaig datagen data$i.txt 300000 0 $i 5000 & done; wait
cat data*.txt > all.txt
gzip -k all.txt        # for upload to Colab/Drive
```

(~80 quiet positions per game, so 8×300k games ≈ 190M positions. Generate as much as your machine/time
allow — more data is the single biggest lever for a stronger net.)

## 2. Train

Local (M3 GPU via MPS, or CPU):
```bash
pip install torch numpy
python3 tools/train/train.py --data all.txt --out askaig-v1.nnue --epochs 30
```

Google Colab (free NVIDIA GPU):
1. Runtime → Change runtime type → **GPU**.
2. Upload `all.txt.gz` to Drive; mount Drive; `gunzip`.
3. In a cell:
   ```python
   !pip -q install torch numpy
   !python train.py --data /content/all.txt --out /content/drive/MyDrive/askaig-v1.nnue --epochs 30
   ```
   (copy `train.py` into the Colab session, e.g. clone the repo or upload the file).

Key flags:
- `--epochs` (default 30 — but with **early-stop** on, it stops when validation loss plateaus),
- `--val` (held-out fraction, default 0.05) — **`val` loss is the honest signal**: when it stops falling
  while `train` keeps dropping, the net is **overfitting**. The exported net is the **best-val-loss**
  epoch, not the last (most-overfit) one,
- `--patience` (early-stop after N epochs without val improvement, default 6; 0 = off),
- `--wd` (weight decay / L2, default 1e-7 — raise it to fight overfit on small data),
- `--l1` (hidden width, default 1024 — **must equal `src/nnue/network.h` L1**; rebuild the engine if you
  change it. Smaller L1 (e.g. 256) overfits less when data is limited),
- `--lr` (1e-3, cosine-decayed), `--wdl` (lambda, 0.5 = blend score and game-result equally),
  `--batch` (16384), `--device` (auto / cuda / mps / cpu).

First run parses the FENs into `<data>.npz`; later runs reuse that cache.

> **If the net plays far worse than the hand-crafted eval**, it's almost always **too little data +
> overfitting**, not a pipeline bug (the contract check above is exact). Watch the gap between `train`
> and `val` loss: a large gap = overfit → get **more data**, raise `--wd`, lower `--epochs`/`--l1`.

## 3. Use the net

```bash
# verify the trained net loads + the contract still holds:
python3 tools/train/train.py --check askaig-v1.nnue "<some FEN>"
# then (after M6 wires EvalFile): setoption name EvalFile value askaig-v1.nnue
```

## Iterate (the real quality lever — "regen")

Once a net plays well, generate NEW data labelled by the **NNUE** eval (stronger than the HCE labels of
the first pass) and retrain — `data → net → better data → better net`. That loop, not cranking datagen
depth, is where label quality compounds.

## Notes
- Data loading is pure-Python FEN parsing (slower than a Rust trainer); the `.npz` cache makes re-runs
  fast. For tens of millions of positions keep an eye on RAM (each position is 2×32 int16 + 2 float32).
- If `export` warns "out of int16 range", the net is mis-scaled — lower the learning rate or check QA.
- This is a from-scratch MIT trainer (no Stockfish/bullet net), so the engine stays MIT and original.
