# Askaig

**Askaig** is a strong, single-binary **UCI chess engine** written in modern **C++26**. It uses a
**bitboard** board representation with **magic bitboards** (hyperbola quintessence) for sliding
pieces, and plays through a multi-threaded, heavily-pruned alpha-beta search with a tapered
hand-crafted evaluation.

The move generator is **ported from [nkarve/surge](https://github.com/nkarve/surge)** (MIT) — a fast,
correct, magic-bitboard legal move generator — and the engine (search, evaluation, transposition
table, UCI front-end, SIMD primitives) is built on top of it.

---

## Features

- **UCI protocol** — drops straight into a UCI loop, works with any UCI GUI (CuteChess, Arena, BanksiaGUI, …).
- **C++26**, header-heavy, template-per-`Color` design (the side is a compile-time parameter).
- **Magic bitboards**, a 16-bit move representation, make/unmake with an undo stack, and **Zobrist hashing** (with a side-to-move term).
- **SIMD** bit primitives with three back-ends — **AVX2** (POPCNT/TZCNT/AVX2), **ARM NEON** (vcnt/rbit), and a portable **`<bit>`** fallback — auto-selected by architecture.
- **Multi-threaded** search (Lazy SMP) sharing a lockless transposition table; reported `nps` scales with thread count.

## Search

A multi-threaded **(Lazy SMP) iterative-deepening PVS** (negamax + alpha-beta) search:

- **Aspiration windows** around the previous iteration's score.
- **Principal Variation Search** (full window on the first move, null-window scout on the rest).
- **Transposition table** — lockless, power-of-two, depth-preferred replacement, ply-adjusted mate scores (default 2 GiB, resizable via `setoption Hash`).
- **Move ordering** — TT move → MVV-LVA captures → killer moves → history heuristic.
- **Quiescence** search at the horizon (capture resolution) with **delta pruning**.
- **Pruning** — null-move pruning (with a zugzwang guard), reverse futility / static null move, futility pruning, late move pruning (LMP), **late move reductions (LMR)**, and mate-distance pruning.
- **Extensions** — check, mate-threat, one-reply, recapture, passed-pawn, and **singular** extensions.
- The exact **principal variation** is collected via a triangular PV table and reported on each `info` line.

Together these cut the searched node count by **~95%+** on quiet positions versus plain alpha-beta, letting the engine reach much greater depth in the same time. With `Threads 1` the search is fully deterministic.

## Evaluation

A **tapered** (middlegame ↔ endgame) evaluation, interpolated by game phase from the remaining material:

- **Material + piece-square tables** — maintained **incrementally** in the make/unmake primitives (two MG/EG accumulators); the king table tapers from "hide on the back rank" to "march to the centre".
- **Pawn structure** — doubled / isolated penalties, a **tapered passed-pawn** bonus, centered pawns, and knight **outposts** (a knight on an enemy "hole" defended by a pawn).
- **King safety** — missing pawn-shield and open/semi-open files beside the king, **scaled by the opponent's attacking material** (fades out in the endgame).
- **Mobility** — per safe square each piece attacks, **tapered** (sliders are worth more on the open endgame board).
- **Pin penalties** for pieces pinned to their own king.
- **Bishop pair** (tapered), **rook on an open / semi-open file**, and **rook on the 7th rank** (tapered).
- A thread-local **evaluation cache** keyed by the Zobrist hash.

## Building

Requires CMake and a C++26 compiler (Clang or GCC). Builds default to **Release** (`-O3`) with **SIMD on** (architecture auto-detected: arm64 → NEON, x86 → AVX2):

```bash
cmake -S . -B build && cmake --build build
./build/askaig
```

Presets are also provided:

```bash
cmake --preset release-simd && cmake --build --preset release-simd   # -O3 + SIMD
cmake --preset debug        && cmake --build --preset debug          # -O0, no SIMD
```

Useful knobs: `-DCMAKE_BUILD_TYPE=Debug`, `-DSIMD=OFF` (portable `<bit>` fallback), `-DARCH=AVX2|ARM_NEON` (force the architecture).

## Usage

`askaig` speaks UCI on stdin/stdout:

```bash
printf 'uci\nposition startpos moves e2e4 e7e5\ngo depth 12\nquit\n' | ./build/askaig
```

Supported commands: `uci`, `isready`, `ucinewgame`, `position [startpos | fen <fen>] [moves ...]`,
`go [depth <n>]`, `go perft <depth>`, `setoption name Hash value <MB>`, `setoption name Threads value <n>`,
`d` / `display`, `stop`, `quit`.

## Correctness (perft)

Move generation is verified by **perft**. From the start position, `go perft 6` must report exactly:

```bash
printf 'position startpos\ngo perft 6\nquit\n' | ./build/askaig | grep "Nodes searched"
# Nodes searched: 119060324
```

This number must hold across all build configurations (scalar, NEON, AVX2). `go perft` is **parallelised** across the `Threads` option (root moves are split over worker threads), so it scales close to linearly with core count while the divide output and total stay identical to the single-threaded run.

## Credits

- Move generation ported from **[nkarve/surge](https://github.com/nkarve/surge)** (MIT License).
- Piece-square tables are Tomasz Michniewski's "Simplified Evaluation Function".
- Many search/evaluation techniques follow the [Chess Programming Wiki](https://www.chessprogramming.org/).
