<h1 align="center">Askaig 🏔️</h1>
<p align="center">A UCI chess engine in modern C++26 — bitboards, NNUE eval, lazy-SMP search.</p>

> [!IMPORTANT]
> This product was maintained by [@sophiathedev](https://github.com/sophiathedev) and having LLM's contributions.

**Askaig** is a **UCI chess engine** in **C++26** — magic bitboards for move generation, an
**NNUE** evaluation, and a heavily-pruned alpha-beta search parallelised with **lazy SMP**. Move
generation is ported from [nkarve/surge](https://github.com/nkarve/surge) (MIT); everything else
(NNUE inference, search, TT, UCI, SIMD) is built on top of it.

---

## Elo Ratings

| Version      | STC <sub>8.0+0.08s  | LTC <sub>60.0+0.60s | VLTC <sub>2m24s+1.12s |
| ------------ | ------------------- | ------------------- | --------------------- |
| **20260704** | **3005<sub>(+610)** | **3193<sub>(+532)** | **3245<sub>(+536)**   |
| 20260628     | 2395<sub>(-1)       | 2661<sub>(+23)      | 2709<sub>(-25)        |
| 20260616     | 2396<sub>(+new)     | 2638<sub>(+new)     | 2734<sub>(+new)       |
|              | cElo <sub>(∆ prev)  | cElo <sub>(∆ prev)  | cElo <sub>(∆ prev)    |

---

## Features

- **UCI protocol** — works with any UCI GUI.
- **C++26**, magic bitboards, 16-bit moves, Zobrist hashing.
- **SIMD** — AVX2, ARM NEON, scalar fallback, auto-selected.
- **NNUE evaluation**, quantized int16/int32, incremental accumulator, embedded net
  (`setoption name EvalFile` to override). Training pipeline in
  [`tools/nnue/`](tools/nnue/README.md).
- **Lazy SMP** parallel search sharing a lockless transposition table.

## File Structure

```text
askaig/
├── cmake/                  # CMake auxiliary scripts
│   └── embed_file.cmake    # Embeds the neural network weights file directly into the binary
├── networks/               # Neural network weights
│   └── default.nnue        # Default evaluation network (quantized int16/int32)
├── src/                    # Chess engine source code (C++26)
│   ├── main.cpp            # Entry point
│   ├── uci.h/.cpp          # UCI (Universal Chess Interface) protocol implementation
│   ├── smp.h/.cpp          # Lazy SMP parallel search helper threads manager
│   ├── search.h/.cpp       # Principal alpha-beta/Negamax search and pruning heuristics
│   ├── movepick.h          # Move generation wrapper & move ordering state machine
│   ├── position.h/.cpp     # Board representation, move generator (magic bitboards), and move play
│   ├── see.h/.cpp          # Static Exchange Evaluation (SEE)
│   ├── nnue.h/.cpp         # NNUE evaluation network inference (HalfKP, SCReLU accumulator)
│   ├── simd.h              # SIMD (AVX2, ARM NEON, scalar fallback) optimization for NNUE
│   ├── tt.h/.cpp           # Multi-threaded lockless Transposition Table
│   ├── history.h           # Move history, capture history, and continuation history tables
│   ├── tables.h/.cpp       # Precomputed attack bitboard lookup tables
│   ├── types.h/.cpp        # Basic type definitions (Square, Move, Piece, Color, Value, etc.)
│   └── datagen.h/.cpp      # Self-play data generation for NNUE training
└── tools/                  # Match playing, tuning, and training tools
    ├── nnue/               # NNUE training pipeline (PyTorch + Rust/Metal bullet runner)
    ├── spsa.py             # SPSA (Simultaneous Perturbation Stochastic Approximation) parameters tuner
    ├── sprt.sh             # SPRT (Sequential Probability Ratio Test) match runner against baseline
    ├── build_pgo.sh        # Build script for Profile-Guided Optimization (PGO)
    └── datagen.sh          # Orchestrator script for multi-threaded self-play training data generation
```

## Search

Fail-soft negamax (alpha-beta), iterative deepening, parallelised by lazy SMP:

- **Move ordering**: TT move → captures (MVV-LVA + capture history) → killers → quiets
  (butterfly + continuation history) → losing captures.
- **Transposition table**: 3-entry clusters, generation aging, depth-preferred replacement.
  Resizable via `setoption name Hash`.
- **Quiescence**: stand-pat cutoff, SEE + delta pruning.
- **PVS** with aspiration windows.
- **Pruning**: reverse futility, razoring, null-move, ProbCut, late-move, futility, history, SEE.
- **Reductions**: log-formula LMR with confidence-scaled re-search depth.
- **Extensions**: check extensions, singular extensions (multicut, double/triple/negative).
- **Correction history**: five tables nudging static eval toward what search has found.
- **Draw detection**: repetition + fifty-move rule; `Contempt` biases against early draws.
- **Time management**: soft/hard budgets scaled by node concentration, move stability, eval
  trend; 200 ms floor always reserved.

Each helper thread runs its own full search on a private copy of the position — only the TT and
history tables are shared. Main thread manages time, reports `bestmove`; `nodes`/`nps` summed
across threads. `go perft` uses its own independent worker pool.

## Building

Requires CMake + a C++26 compiler (Clang/GCC). Release + SIMD by default:

```bash
cmake -S . -B build && cmake --build build
./build/askaig
```

Presets: `release-simd`, `debug`. Knobs: `-DCMAKE_BUILD_TYPE=Debug`, `-DSIMD=OFF`,
`-DARCH=AVX2|ARM_NEON`.

## Usage

```bash
printf 'uci\nposition startpos moves e2e4 e7e5\ngo depth 12\nquit\n' | ./build/askaig
```

Commands: `uci`, `isready`, `ucinewgame`, `position [startpos|fen <fen>] [moves ...]`,
`go depth <n> / movetime <ms> / wtime <ms> btime <ms> [winc/binc/movestogo] / infinite / perft
<depth>`, `setoption name Hash|Threads|Contempt|EvalFile value <x>`, `d`, `eval`, `stop`, `quit`.

Debug: `bench [depth]`, `bench evalnps`, `selftest [nnue|perft|see|draw|search|contempt|stop|
all]`. Runs in CI on every push/PR under ASan/UBSan/TSan + an assertions-enabled Debug build.

`./build/askaig bench` node count is a bit-reproducible search signature: functional patches
change it, pure refactors/UCI changes don't.

## Correctness (perft)

```bash
printf 'position startpos\ngo perft 6\nquit\n' | ./build/askaig | grep "Nodes searched"
# Nodes searched: 119060324
```

Holds across all build configs. `go perft` parallelises across `Threads` and is memoised by an
exact, collision-free state hash, kept per-thread.

## Credits

- Move generation ported from **[nkarve/surge](https://github.com/nkarve/surge)** (MIT License).
- NNUE training data from the [Lichess evaluation database](https://database.lichess.org/#evals).
- Search/evaluation techniques follow the [Chess Programming Wiki](https://www.chessprogramming.org/).
