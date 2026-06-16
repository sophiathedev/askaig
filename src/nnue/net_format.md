# NNUE net format contract (engine ↔ trainer)

The engine's inference (`nnue.cpp` + `network.h` + `features.h`) and the trainer (`tools/train/`, a
PyTorch trainer) must agree **byte-for-byte** on this, or evaluation is garbage. This file is the single
source of truth; the reference check (engine inference == trainer's integer inference on the same net &
FENs) enforces it. The net file uses the **`.nnue`** extension (engine `EvalFile` loads `.nnue`; the
trainer exports `.nnue`).

## Architecture (v1)

- Feature set: **HalfKA + horizontal-mirror, single king bucket** (`NUM_BUCKETS = 1`).
  - Per perspective P, feature = `(king_bucket, plane, square)`:
    - square oriented to P's view (`^56` for BLACK), then `^7` mirror when P's king is on files e-h;
    - `plane = (color_of(pc) == P ? 0 : 1) * 6 + type_of(pc)` (us/them × {P,N,B,R,Q,K});
    - `king_bucket = 0` (v1).
  - `INPUT_DIM = NUM_BUCKETS * 12 * 64 = 768` per perspective.
- Feature transformer: `768 → L1` per perspective, **L1 = 1024**.
- Activation: **SCReLU** — `a = clamp(acc, 0, QA)^2`.
- Output head: `[us L1][them L1] → 1` (side-to-move perspective first).

## Quantisation

- `QA = 255` (FT weights + activation), `QB = 64` (output weights), `SCALE = 400` (output → cp).
- FT weights + bias + accumulator: `int16`. Output weights: `int16`. Output bias: `int32`.
- Inference descale: `cp = (Σ a·w / QA + out_bias) * SCALE / (QA * QB)`.

## On-disk layout (`struct Network`, packed in this order, little-endian)

1. `int16 ft_weights[INPUT_DIM * L1]` — **feature-major**: column for feature `f` at `[f*L1 .. f*L1+L1)`.
2. `int16 ft_bias[L1]`
3. `int16 out_weights[2 * L1]` — `[us L1][them L1]`.
4. `int32 out_bias`

The engine `load()` reads exactly `sizeof(Network)` bytes in this order (little-endian, no padding:
`768*1024 + 1024 + 2048` int16 = 1,579,008 bytes, then one int32 = 1,579,012 total). The PyTorch
exporter (`tools/train/train.py --export`) writes exactly these bytes. If any of L1 / QA / QB / SCALE /
activation / layout changes, change BOTH `network.h` and `tools/train/train.py` and re-run the reference
check.

## Float ↔ integer mapping (trainer ↔ engine)

The trainer's float forward and the engine's integer forward must denote the same eval:
`cp = SCALE * (Σ screlu(acc_f) · out_w_f + out_b_f)`, where `acc_f ∈ [0,1]`, `screlu(x)=clamp(x,0,1)^2`.
Quantisation on export: `ft_w_int = round(ft_w_f * QA)`, `ft_bias_int = round(ft_bias_f * QA)`,
`out_w_int = round(out_w_f * QB)`, `out_bias_int = round(out_b_f * QA * QB)`. With these, the engine's
integer pipeline (`cp = (Σ x²·ow_int / QA + ob_int) * SCALE / (QA*QB)`) reproduces `SCALE * out_f` up to
rounding.

> v1 status: the engine currently runs a deterministic PLACEHOLDER net (see `nnue::init`) so the
> inference path is testable before a real net exists. Mirror-symmetry holds for any weights, so the
> placeholder already validates the feature/perspective/mirror code.
