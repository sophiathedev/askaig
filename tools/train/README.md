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

```bash
# parallel shards, distinct seeds; ~80 quiet positions per game (see datagen docs)
for i in $(seq 1 6); do ./build/askaig datagen data$i.txt 200000 8 $i & done; wait
cat data*.txt > all.txt
gzip -k all.txt        # for upload to Colab/Drive
```

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

Key flags: `--epochs` (20–40), `--batch` (16384), `--lr` (1e-3, cosine-decayed), `--wdl` (lambda, 0.5 =
blend search-score and game-result equally), `--device` (auto / cuda / mps / cpu). The net is exported
**every epoch** (so a disconnect keeps the latest). First run parses the FENs into `<data>.npz`; later
runs reuse that cache.

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
