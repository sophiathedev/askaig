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
`clang++`) and downloads a balanced opening book (`UHO_4060_v3.epd`). Everything it produces lives
under `tools/` and is git-ignored; only the scripts here are committed.

Requirements: `git`, `clang++`, `curl`, `unzip` (all present on a standard macOS dev setup).

## Running a test

```bash
# edit the eval/search, leave the change uncommitted, then:
tools/sprt.sh                 # candidate = working tree   vs   baseline = HEAD

# if the change is already committed:
tools/sprt.sh HEAD~1          # candidate = working tree   vs   baseline = HEAD~1

# compare two specific commits/tags/branches:
tools/sprt.sh <base-ref> <candidate-ref>
```

The script builds both engines (each committed ref in its own throwaway `git worktree`) and plays
them with `fastchess` until the SPRT concludes. It prints a live Elo estimate and ends with either
`H1 was accepted` (the change is an improvement) or `H0 was accepted` (it is not — revert it).

### What the result means

- The default test is **`elo0=0 elo1=5`** with `alpha=beta=0.05`: H1 = "the candidate is somewhere
  between 0 and +5 Elo or better". Accepting H1 means the change is, with 95% confidence, **not a
  regression and probably a small gain** — the standard "does this patch help?" test.
- A clear regression accepts **H0** quickly. A tiny/zero change can run a long time (many thousands of
  games) without concluding — that itself tells you the change is Elo-neutral.

## Tunables (environment variables)

| Var | Default | Meaning |
|-----|---------|---------|
| `TC` | `8+0.08` | time control, `seconds+increment` |
| `HASH` | `64` | TT size (MB) per engine |
| `CONCURRENCY` | cores−3 | parallel games |
| `ELO0`/`ELO1` | `0`/`5` | SPRT hypothesis bounds |
| `ROUNDS` | `5000` | max opening-pairs (each played with both colors) |
| `BOOK` | `UHO_4060_v3.epd` | opening book path |

Example — a stricter, higher-quality (slower) test:

```bash
TC=20+0.2 ELO0=0 ELO1=3 tools/sprt.sh HEAD~1
```

## Notes

- Engines run at **`Threads=1`** so each game is reproducible given the opening; variety comes from
  the book (a book is therefore mandatory). Use a longer book / more rounds for more games.
- Games are written to `tools/work/games.pgn` for inspection.
- For an absolute strength number (vs. a reference engine of known Elo), point one `-engine` at that
  engine instead of a second Askaig build and drop `-sprt` in favour of `-rounds`.
