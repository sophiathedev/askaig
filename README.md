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

A multi-threaded **(Lazy SMP) iterative-deepening PVS** (negamax + alpha-beta, **fail-soft**) search:

- **Aspiration windows** around the previous iteration's score.
- **Principal Variation Search** (full window on the first move, null-window scout on the rest).
- **Transposition table** — lockless, **4-entry bucket** per 64-byte cache line, **generation aging** (eviction by `depth − 8×age`), depth-preferred replacement, carries the **static eval** per entry (reused by shallow pruning nodes), ply-adjusted mate scores. Allocated with `mmap`/`munmap` so resize returns memory to the OS immediately; anonymous pages are lazily faulted (`ucinewgame` is O(1)). Default 2 GiB, resizable via `setoption Hash`.
- **Move ordering** — TT move → MVV-LVA captures refined by a **capture history** (`[attacker][to][victim type]`, gravity-updated alongside quiet history) → killer moves → **signed butterfly history** + **continuation history** at 1/2/4/6 plies back (countermove, follow-up, and two longer-range plan tables, keyed by `(piece, to)×(piece, to)`, persistent across searches in per-thread heuristic slots).
- **Quiescence** search at the horizon with **check evasions** (when in check: no stand-pat, all evasions searched including quiets, mate detected if no legal move), **delta pruning**, and **SEE pruning** of losing captures.
- **SEE pruning in the main search** — losing quiet moves (`> 65×depth` cp) and losing captures (`> 20×depth²` cp) are skipped at `depth ≤ 9` on non-PV nodes; evaluated with a from-scratch static exchange algorithm including x-ray attackers and king recaptures.
- **ProbCut** (`depth ≥ 5`) — captures pre-screened by quiescence at `beta + 150`; confirmed by a `depth − 4` search; stored as a lower-bound TT entry.
- **Correction history** — several per-side running estimates of how far the static eval has historically sat from the search result, keyed by the **pawn skeleton**, the **non-pawn material**, the **minor pieces**, the **major pieces**, and the **previous move** (continuation); their sum nudges the static eval used for pruning and the quiescence stand-pat toward the truth (the raw, uncorrected eval is what's stored in the TT). Updated by the same gravity formula as the other history tables and persistent across moves.
- **Draw detection** — repetition and the fifty-move rule scored as draws in the search (per-ply hash + halfmove clock); the engine claims draws when worse and never repeats away a won position.
- **Pruning** — null-move (adaptive R, zugzwang guard), reverse futility / static null move, **razoring** (drop to a verifying quiescence search when the eval is far below alpha), futility pruning, late-move pruning (LMP), **history pruning** (skip quiets with strongly negative history at shallow depth), **log-table LMR** (`1.00 + ln(d)·ln(i)/1.50`) adjusted by the signed butterfly history and the **improving** flag, and mate-distance pruning.
- **Adaptive time management** — clock-based searches scale their per-move soft budget by best-move stability (`scale = 1.4 − 0.08×stable`, `+0.3` on score drops ≥ 30 cp, clamped `[0.5, 1.8]`): more time when the best move keeps changing or the score drops, less when it has settled. Hard deadline polled every 2048 nodes to bound any runaway. Also supports `movetime` and **pondering** (`go ponder` + `ponderhit`).
- **Extensions** — check, mate-threat, one-reply, passed-pawn (pawn pushed to the relative 7th), and **singular** (reduced multi-search shows the TT move is the only good one).
- **Heuristics persistence** — butterfly, continuation, and capture history tables live in persistent per-thread slots (`g_heur`), carrying over across moves; cleared only by `ucinewgame`. History bonuses and maluses are applied to quiet cut-off moves and the refuted quiets searched before them.
- The exact **principal variation** is collected via a triangular PV table and reported on each `info` line.

Together these cut the searched node count by **~95%+** on quiet positions versus plain alpha-beta, letting the engine reach much greater depth in the same time. With `Threads 1` the search is fully deterministic.

## Evaluation

The engine uses a **hand-crafted evaluation (HCE)**: PeSTO's tapered material + piece-square tables (maintained incrementally in the make/unmake primitives), plus mobility, threats, king safety, pawn structure, passed pawns and a fifty-move damping. It is also what `bench`/`tune` operate on.

The hand-crafted evaluation is a **tapered** (middlegame ↔ endgame) evaluation, interpolated by game phase from the remaining material. All tunable constants were **Texel-tuned** (coordinate descent on self-play positions with L2 regularisation and per-parameter hard bounds):

- **Material + piece-square tables** — [**PeSTO's Evaluation Function**](https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function) (Ronald Friederich's tuned tables), with separate MG/EG material values and tables for every piece. Maintained **incrementally** in the make/unmake primitives (two running MG/EG accumulators) and interpolated by game phase.
- **Pawn structure** — doubled / isolated penalties, a **tapered passed-pawn** bonus (reduced when **blockaded**, plus an endgame **king-distance race** term), centered pawns, and knight **outposts** (a knight on an enemy "hole" defended by a pawn). Evaluated via a **per-thread pawn cache** (exact `(wp, bp)` match, ~8k entries) so most positions pay O(1) for the pawn terms.
- **King safety** — missing pawn-shield and open/semi-open files beside the king, **scaled by the opponent's attacking material** (fades out in the endgame), plus a **king-zone attack** term (weighted attacks into the ring around the enemy king, scaled by the number of distinct attackers, quartered in the endgame).
- **Mobility** — per safe square each piece attacks, **tapered** (sliders are worth more on the open endgame board); mobility, threats, and king-zone attacks share a **single pass** over the piece attack maps so the costly magic-bitboard lookups are done exactly once per piece.
- **Threats** — pieces attacked by pawns / minors / rooks (by victim type) and **hanging** (attacked, undefended) pieces.
- **Pin penalties** for pieces pinned to their own king.
- **Bishop pair** (tapered), **rook on an open / semi-open file**, and **rook on the 7th rank** (tapered).
- A **tempo** bonus for the side to move.
- **Fifty-move damping** — the static eval is damped toward zero as the halfmove clock approaches 100 (so the engine doesn't over-value a winning position it can no longer convert).

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
(search until `stop`), `go perft <depth> [noncache]`, `setoption name Hash value <MB>`, `setoption name Threads value <n>`,
`d` / `display`, `eval` (static evaluation of the current position), `stop`, `quit`.

### Benchmark

```bash
./build/askaig bench          # 12 fixed positions, Threads=1, 16 MB TT — prints per-position nodes
./build/askaig bench 10       # same at depth 10
```

The total node count is a **bit-reproducible search signature**: functional patches (changed
move-ordering, pruning, extensions) change it; pure refactors and UCI changes don't. Use it to
classify changes before running full SPRT.

### Texel tuning

```bash
./build/askaig tune <book> [threads] [lambda]
```

`<book>` is a `<FEN> <result>` file (one line per position, result in `{1.0, 0.5, 0.0}` from
White's POV). The tuner fits the sigmoid scale K by ternary search, then runs coordinate descent
over all 64 eval constants, minimising `mse + λ × L2_penalty` (L2 pulls toward compiled-in
defaults; hard bounds prevent collinear features from trading off into absurd pairs). Progress and
the final paste-ready dump are printed to stdout; results are also written to `<book>.tuned`.

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
