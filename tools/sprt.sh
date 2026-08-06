#!/usr/bin/env bash
#
# SPRT self-play test: decides whether the CANDIDATE build is stronger than the BASELINE, by playing
# them against each other until the Sequential Probability Ratio Test reaches a verdict (H1 = the
# candidate gains in (elo0, elo1); H0 = it does not). This is the ONLY reliable measure of strength —
# node counts and tactical spot-checks confirm *correctness*, not Elo.
#
#   tools/sprt.sh [BASE]
#
#     CANDIDATE  = build-pgo/askaig — a fresh PGO build of the CURRENT source tree.
#     BASE       = the opponent. Two forms, auto-detected:
#                    - a path to an EXECUTABLE (stockfish, another askaig binary, ...) -> played
#                      directly, no build. THIS is how you test against a DIFFERENT engine.
#                    - a git ref (commit/tag/branch, default HEAD) -> built Release into build/ from
#                      a throwaway worktree of that commit.
#
# Typical loop — edit code, leave it uncommitted, test vs the last commit:
#     tools/sprt.sh
# If the change is already committed (tree == HEAD), compare against its parent:
#     tools/sprt.sh HEAD~1
# Play against ANOTHER engine (absolute-strength gauge — usually pair with SPRT off, see ELO bounds):
#     tools/sprt.sh ~/engines/stockfish
#     tools/sprt.sh ~/Downloads/askaig-20260616
#
# Tunables via env: TC (5+0.05), HASH (16), CONCURRENCY (P-cores - 1), ELO0/ELO1 (0/10),
#                   ALPHA/BETA (0.1), ROUNDS (5000), BOOK (UHO_4060_v3.epd), ADJUDICATE (1),
#                   DEPTH (unset).
#
# DEPTH=<n> plays FIXED-DEPTH games (both engines search exactly n plies, no clock). This removes
# every speed effect — nps, cache pressure, time management — and measures ONLY the quality of the
# searched tree (eval + ordering + pruning decisions). The standard diagnostic when a TC test fails:
# if the patch wins at fixed depth but loses on the clock, the heuristic signal is good and the
# implementation is too slow; if it loses at fixed depth too, the signal itself is harmful.
#
# The defaults are the FAST regime for an engine still gaining tens of Elo per change: bounds
# [0, 10] need ~3-4x fewer games than [0, 5] for a true gain >= 10, alpha/beta 0.1 another ~1.4x,
# and 5s games ~1.6x wall-clock. Trade-off: ~10% false accept/reject, and small (+3-5 Elo) patches
# will often fail — re-test those with ELO1=5 ALPHA=0.05 BETA=0.05 TC=8+0.08 when they matter.
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# Locate the fastchess match-runner: $FASTCHESS env wins, else search common locations.
if [ -z "${FASTCHESS:-}" ]; then
  for c in "$HOME/Downloads/fastchess-mac-arm64/fastchess" "$HERE/fastchess-src/fastchess" \
           "$(command -v fastchess 2>/dev/null || true)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then FASTCHESS="$c"; break; fi
  done
fi
BOOK="${BOOK:-$HERE/books/UHO_4060_v3.epd}"
# Opening-book format passed to fastchess (epd|pgn). Default epd (UHO). A balanced PGN book
# (e.g. books/8moves_v3.pgn) reaches endgames far more often — set BOOK_FORMAT=pgn for it. Pair
# with ADJUDICATE=0 when testing endgame eval, or the resign rule ends won games before they
# reach the bare-king phase those terms operate in.
BOOK_FORMAT="${BOOK_FORMAT:-epd}"
[ -n "${FASTCHESS:-}" ] && [ -x "$FASTCHESS" ] || {
  echo "fastchess not found — set FASTCHESS=/path/to/fastchess (or run 'bash tools/setup.sh')"
  exit 1
}
[ -f "$BOOK" ] || { echo "opening book missing — run 'bash tools/setup.sh' to download it"; exit 1; }

TC="${TC:-5+0.05}"
DEPTH="${DEPTH:-}" # set to play fixed-depth games instead of timed ones (see header)
HASH="${HASH:-16}"
CORES="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
# Only "real" cores run a search at full speed; on Apple Silicon the efficiency cores are far slower.
# Oversubscribing them starves fastchess itself — it then misses an engine's reply within its timeout
# and aborts the whole match ("stalled / disconnected"), even though the engine answered fine. So base
# concurrency on the *performance* cores (perflevel0 on Apple Silicon; physical cores elsewhere) and
# leave one free for fastchess + the OS. Override with CONCURRENCY=<n> if you know your machine.
PCORES="$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || sysctl -n hw.physicalcpu 2>/dev/null || echo "$CORES")"
CONCURRENCY="${CONCURRENCY:-$(( PCORES > 2 ? PCORES - 1 : 1 ))}"
ELO0="${ELO0:-0}"
ELO1="${ELO1:-10}"
ALPHA="${ALPHA:-0.1}"
BETA="${BETA:-0.1}"
ROUNDS="${ROUNDS:-5000}"

BASE_REF="${1:-HEAD}"

# Adjudication — ends games whose outcome is no longer in doubt (the fishtest/OpenBench standard;
# with an unbalanced UHO book MOST games end this way, which is the intended time saving, roughly
# halving the wall-clock per game):
#   draw:   after move 34, BOTH engines report |score| <= 20cp for 8 consecutive moves. The window
#           must clear the engine's TEMPO bonus ({18,10} mg/eg): the side to move always reports
#           at least ~+10cp, so a 10cp window NEVER caught a dead-drawn shuffle — measured over
#           ~2900 games, 82% of draws ran to an actual 3-fold/50-move (median draw 112 plies, max
#           520). 20cp (the OpenBench default, like movenumber 34) is above the endgame tempo.
#   resign: BOTH engines agree (twosided=true) one side is down >= 700cp for 4 consecutive moves.
#           twosided matters for eval patches: a one-sided resign lets a candidate with a broken
#           (pessimistic) eval forfeit positions it could still hold, biasing the SPRT against it.
#   maxmoves 200: hard backstop for shuffles the draw window still misses (middlegame tempo 18cp
#           sits just under the 20cp window) — a 400-ply game costs ~12 normal games of wall-clock.
# ADJUDICATE=0 disables all three — games then run to mate / 50-move / repetition; useful as a
# sanity cross-check if an adjudicated result looks suspicious. The tokens contain no spaces, so
# the unquoted expansion below splitting into words is exactly what we want.
ADJ_ARGS=""
if [ "${ADJUDICATE:-1}" != "0" ]; then
  ADJ_ARGS="-draw movenumber=34 movecount=8 score=20 -resign movecount=4 score=700 twosided=true -maxmoves 200"
fi

WORK="$HERE/work"
mkdir -p "$WORK"

# --- Candidate: PGO build of the current source tree ---------------------------------------------
CAND_DIR="$ROOT/build-pgo"
PGO_SCRIPT="$HERE/build_pgo.sh"
[ -f "$PGO_SCRIPT" ] || { echo "PGO helper missing: $PGO_SCRIPT"; exit 1; }
echo ">> building candidate (current tree, PGO) -> build-pgo/ ..."
bash "$PGO_SCRIPT"
CAND_BIN="$CAND_DIR/askaig"
[ -x "$CAND_BIN" ] || { echo "candidate binary missing: $CAND_BIN"; exit 1; }

# --- Baseline: an EXTERNAL engine executable, or BASE_REF (a git ref) built into build/ -----------
# If BASE_REF points at an executable (stockfish, another askaig binary, ...) play it directly — no
# build. Otherwise treat it as a git ref: materialise it in a throwaway worktree and build Release into
# build/ (wiped first so the CMake cache never points at a stale source dir).
if [ -f "$BASE_REF" ] && [ -x "$BASE_REF" ]; then
  BASE_BIN="$(cd "$(dirname "$BASE_REF")" && pwd)/$(basename "$BASE_REF")"   # absolute path
  BASE_NAME="$(basename "$BASE_REF")"
  echo ">> baseline = external engine: $BASE_BIN"
else
  BASE_SRC="$WORK/base-src"
  cleanup() { git -C "$ROOT" worktree remove --force "$BASE_SRC" >/dev/null 2>&1 || true; }
  trap cleanup EXIT
  cleanup
  BASE_NAME="base"
  echo ">> building baseline  ($BASE_REF) -> build/ ..."
  git -C "$ROOT" worktree add --detach -f "$BASE_SRC" "$BASE_REF" >/dev/null
  rm -rf "$ROOT/build"
  cmake -S "$BASE_SRC" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$ROOT/build" -j"$CORES" >/dev/null
  BASE_BIN="$ROOT/build/askaig"
fi
[ -x "$BASE_BIN" ] || { echo "baseline binary missing: $BASE_BIN"; exit 1; }

# Timed games by default; DEPTH=<n> switches to fixed-depth (no clock — see the header).
if [ -n "$DEPTH" ]; then LIMIT="depth=$DEPTH"; else LIMIT="tc=$TC"; fi

# SPRT early-stopping by default; SPRT=0 runs a FIXED ROUNDS match instead — an absolute strength
# gauge with the full game count, which is what you want when BASE is a DIFFERENT engine of known
# rating (an SPRT vs a much stronger engine just accepts H0 after a handful of games).
SPRT_ARGS=""
[ "${SPRT:-1}" != "0" ] && SPRT_ARGS="-sprt elo0=$ELO0 elo1=$ELO1 alpha=$ALPHA beta=$BETA"

if [ -n "$SPRT_ARGS" ]; then
  echo ">> SPRT  cand vs $BASE_NAME   $LIMIT  Hash=$HASH  concurrency=$CONCURRENCY  H1=elo in ($ELO0,$ELO1)  alpha=$ALPHA beta=$BETA"
else
  echo ">> MATCH cand vs $BASE_NAME   $LIMIT  Hash=$HASH  concurrency=$CONCURRENCY  fixed $ROUNDS rounds x2 (no SPRT)"
fi
echo

"$FASTCHESS" \
  -engine cmd="$CAND_BIN" name=cand \
  -engine cmd="$BASE_BIN" name="$BASE_NAME" \
  -each "$LIMIT" option.Hash="$HASH" option.Threads=1 proto=uci \
  -openings file="$BOOK" format="$BOOK_FORMAT" order=random \
  -games 2 -rounds "$ROUNDS" -repeat \
  $SPRT_ARGS \
  $ADJ_ARGS \
  -recover \
  -concurrency "$CONCURRENCY" \
  -ratinginterval 20 \
  -pgnout file="$WORK/games.pgn"
