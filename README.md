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
- **Move ordering** — TT move → MVV-LVA captures → killer moves → **signed history** (gravity bonus on quiet cut-offs, malus for the quiets refuted before them).
- **Quiescence** search at the horizon (capture resolution) with **delta pruning**.
- **Draw detection** — repetition and the fifty-move rule are scored as draws in the search (via a per-ply hash + halfmove clock), so the engine claims draws when worse and never repeats away a won position.
- **Pruning** — null-move pruning (with a zugzwang guard), reverse futility / static null move, futility pruning, late move pruning (LMP), **log-table late move reductions (LMR)** adjusted by the signed history score, **SEE pruning of losing captures** in quiescence, and mate-distance pruning.
- **Adaptive time management** — clock-based searches scale their per-move budget by best-move stability: more time when the best move keeps changing or the score drops, less when it has settled (banking time for later moves), with a hard cap as a safety net. Also supports `movetime` and pondering.
- **Extensions** — check, mate-threat, one-reply, passed-pawn, and **singular** extensions.
- The exact **principal variation** is collected via a triangular PV table and reported on each `info` line.

Together these cut the searched node count by **~95%+** on quiet positions versus plain alpha-beta, letting the engine reach much greater depth in the same time. With `Threads 1` the search is fully deterministic.

## Evaluation

A **tapered** (middlegame ↔ endgame) evaluation, interpolated by game phase from the remaining material:

- **Material + piece-square tables** — [**PeSTO's Evaluation Function**](https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function) (Ronald Friederich's tuned tables), with separate middlegame/endgame material values and tables for every piece, maintained **incrementally** in the make/unmake primitives (two MG/EG accumulators) and interpolated by game phase.
- **Pawn structure** — doubled / isolated penalties, a **tapered passed-pawn** bonus (reduced when **blockaded**, plus an endgame **king-distance race** term), centered pawns, and knight **outposts** (a knight on an enemy "hole" defended by a pawn).
- **King safety** — missing pawn-shield and open/semi-open files beside the king, **scaled by the opponent's attacking material** (fades out in the endgame), plus a **king-zone attack** term (weighted attacks into the ring around the enemy king, scaled by the number of attackers).
- **Mobility** — per safe square each piece attacks, **tapered** (sliders are worth more on the open endgame board); mobility, threats and king-zone attacks share a **single pass** over the piece attack sets.
- **Threats** — pieces attacked by pawns / minors / rooks (by victim type) and **hanging** (attacked, undefended) pieces.
- **Pin penalties** for pieces pinned to their own king.
- **Bishop pair** (tapered), **rook on an open / semi-open file**, and **rook on the 7th rank** (tapered).
- A **tempo** bonus for the side to move.

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
`go [depth <n>]`, `go movetime <ms>`, `go wtime <ms> btime <ms> [winc <ms>] [binc <ms>] [movestogo <n>]`
(clock-based time management), `go ponder` + `ponderhit` (think on the opponent's time), `go infinite`
(search until `stop`), `go perft <depth>`, `setoption name Hash value <MB>`, `setoption name Threads value <n>`,
`d` / `display`, `eval` (static evaluation of the current position), `stop`, `quit`.

## Correctness (perft)

Move generation is verified by **perft**. From the start position, `go perft 6` must report exactly:

```bash
printf 'position startpos\ngo perft 6\nquit\n' | ./build/askaig | grep "Nodes searched"
# Nodes searched: 119060324
```

This number must hold across all build configurations (scalar, NEON, AVX2). `go perft` is **parallelised** across the `Threads` option (root moves are split over worker threads), so it scales close to linearly with core count while the divide output and total stay identical to the single-threaded run. It is also **memoised by an exact perft hash**: each entry stores the full move-generation state (placement + side + castling + en-passant) and is matched bit-for-bit, so — unlike a plain Zobrist-keyed perft hash — there is **no chance of an inaccurate count** from key collisions. The table is per-thread (never shared, never stale).

## Credits

- Move generation ported from **[nkarve/surge](https://github.com/nkarve/surge)** (MIT License).
- Piece-square tables are [PeSTO's Evaluation Function](https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function) (Ronald Friederich / rofChade).
- Many search/evaluation techniques follow the [Chess Programming Wiki](https://www.chessprogramming.org/).
