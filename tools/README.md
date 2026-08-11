# SPRT testing harness

The **only** reliable way to know whether an engine change makes Askaig *stronger* is to play many
games against the previous version and run a statistical test. Node counts, perft, and tactical
spot-checks confirm **correctness**, never **Elo**. Use this harness for every change intended to
gain strength (evaluation tweaks, search tuning, time management, …).

## One-time setup

```bash
bash tools/setup.sh
```

This builds the [`fastchess`](https://github.com/Disservin/fastchess) match-runner from source (with
`clang++`) and downloads the default opening book (`UHO_Lichess_4852_v1.epd`). Everything it produces lives
under `tools/` and is git-ignored; only the scripts here are committed.

Requirements: `git`, `clang++`, `curl`, `unzip` (all present on a standard macOS dev setup).

## Running a test

```bash
# edit the eval/search, leave the change uncommitted, then:
tools/sprt.sh                 # candidate = working tree   vs   baseline = HEAD

# if the change is already committed:
tools/sprt.sh HEAD~1          # candidate = working tree   vs   baseline = HEAD~1

# name a run, stop it at any time, then continue with the frozen binaries and stats:
RUN_ID=tt-cutoff tools/sprt.sh HEAD~1
tools/sprt.sh --resume tt-cutoff
```

The script builds both engines (each committed ref in its own throwaway `git worktree`) and plays
them with `fastchess` until the SPRT concludes. It prints a live Elo estimate and ends with either
`H1 was accepted` (the change is an improvement) or `H0 was accepted` (it is not — revert it).

Each experiment lives under `tools/work/sprt-runs/<run-id>/`. The directory contains frozen PGO
binaries, SHA-256 checksums, an autosaved `fastchess.json`, its PGN, and the console log. Resume
loads the saved schedule, seed, pentanomial statistics, and LLR state instead of starting over.

To import a match created by the old shared `config.json`/`tools/work/games.pgn` layout:

```bash
RUN_ID=tt-cutoff-legacy tools/sprt.sh --adopt config.json HEAD~1
tools/sprt.sh --resume tt-cutoff-legacy
```

Import keeps the existing fastchess statistics and extracts the matching tail of the shared PGN.
If an old engine binary no longer exists, the supplied base ref is rebuilt with PGO before the run
is frozen.

### What the result means

- The default test is **`elo0=0 elo1=10`** with `alpha=beta=0.1`. Small candidates can use
  `ELO1=5 ALPHA=0.05 BETA=0.05` for a slower, stricter confirmation.
- A clear regression accepts **H0** quickly. A tiny/zero change can run a long time (many thousands of
  games) without concluding — that itself tells you the change is Elo-neutral.

## Tunables (environment variables)

| Var | Default | Meaning |
|-----|---------|---------|
| `TC` | `5+0.05` | time control, `seconds+increment` |
| `HASH` | `16` | TT size (MB) per engine |
| `CONCURRENCY` | P-cores−1 | parallel games |
| `ELO0`/`ELO1` | `0`/`10` | SPRT hypothesis bounds |
| `ALPHA`/`BETA` | `0.1`/`0.1` | SPRT error rates |
| `ROUNDS` | `5000` | max opening-pairs (each played with both colors) |
| `BOOK` | `UHO_Lichess_4852_v1.epd` | opening book path |
| `ADJUDICATE` | `1` | `0` disables draw/resign adjudication (games run to mate/50-move/repetition, ~2x wall-clock) |
| `RUN_ID` | timestamp + commit | Stable name used by `--resume` |

Example — a stricter, higher-quality (slower) test:

```bash
TC=20+0.2 ELO0=0 ELO1=3 tools/sprt.sh HEAD~1
```

## Notes

- **Most games end by adjudication — that is intended.** Draws need BOTH engines reporting
  |score| ≤ 10cp for 8 straight moves after move 40; resigns need BOTH engines agreeing
  (`twosided=true`) one side is down ≥ 700cp for 4 straight moves. These are the fishtest/OpenBench
  thresholds: they only cut games whose outcome is no longer in doubt, roughly halving wall-clock.
  If a verdict looks suspicious, re-run the test with `ADJUDICATE=0` as a cross-check.
- Engines run at **`Threads=1`** so each game is reproducible given the opening; variety comes from
  the book (a book is therefore mandatory). Use a longer book / more rounds for more games.
- Games and logs are isolated under `tools/work/sprt-runs/<run-id>/`.
- For an absolute strength number (vs. a reference engine of known Elo), point one `-engine` at that
  engine instead of a second Askaig build and drop `-sprt` in favour of `-rounds`.
