#!/usr/bin/env bash
#
# SPRT self-play test: decides whether the CANDIDATE build is stronger than the BASELINE, by playing
# them against each other until the Sequential Probability Ratio Test reaches a verdict (H1 = the
# candidate gains in (elo0, elo1); H0 = it does not). This is the ONLY reliable measure of strength —
# node counts and tactical spot-checks confirm *correctness*, not Elo.
#
#   tools/sprt.sh [BASE_REF]
#
#     CANDIDATE  = cmake-build-release/askaig — the CLion "Release + SIMD" preset build of the
#                  CURRENT source tree. The script re-runs that build (configuring the preset first
#                  if the directory is missing) so the binary is fresh and in Release mode.
#     BASELINE   = build/askaig — built (Release) from the git ref BASE_REF, default HEAD.
#                  build/ is wiped and rebuilt from a temporary worktree of that commit each run.
#
# Typical loop — edit code (CLion keeps cmake-build-release current), test vs the last commit:
#     tools/sprt.sh
# If you already committed the change (so the source tree == HEAD), compare against its parent:
#     tools/sprt.sh HEAD~1
#
# Tunables via env: TC (5+0.05), HASH (16), CONCURRENCY (P-cores - 1), ELO0/ELO1 (0/10),
#                   ALPHA/BETA (0.1), ROUNDS (5000), BOOK (UHO_4060_v3.epd), ADJUDICATE (1).
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
[ -n "${FASTCHESS:-}" ] && [ -x "$FASTCHESS" ] || {
  echo "fastchess not found — set FASTCHESS=/path/to/fastchess (or run 'bash tools/setup.sh')"
  exit 1
}
[ -f "$BOOK" ] || { echo "opening book missing — run 'bash tools/setup.sh' to download it"; exit 1; }

TC="${TC:-5+0.05}"
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
#   draw:   after move 40, BOTH engines report |score| <= 10cp for 8 consecutive moves
#   resign: BOTH engines agree (twosided=true) one side is down >= 700cp for 4 consecutive moves.
#           twosided matters for eval patches: a one-sided resign lets a candidate with a broken
#           (pessimistic) eval forfeit positions it could still hold, biasing the SPRT against it.
# ADJUDICATE=0 disables both — games then run to mate / 50-move / repetition (~2x wall-clock);
# useful as a sanity cross-check if an adjudicated result looks suspicious. The tokens contain no
# spaces, so the unquoted expansion below splitting into words is exactly what we want.
ADJ_ARGS=""
if [ "${ADJUDICATE:-1}" != "0" ]; then
  ADJ_ARGS="-draw movenumber=40 movecount=8 score=10 -resign movecount=4 score=700 twosided=true"
fi

WORK="$HERE/work"
mkdir -p "$WORK"

# --- Candidate: the CLion Release+SIMD build of the current source tree --------------------------
# Refresh it so the binary matches the sources (and configure the preset first if it's missing).
CAND_DIR="$ROOT/cmake-build-release"
echo ">> building candidate (current tree) -> cmake-build-release/ ..."
if [ ! -f "$CAND_DIR/CMakeCache.txt" ]; then
  cmake --preset release-simd -S "$ROOT" >/dev/null
fi
cmake --build "$CAND_DIR" -j"$CORES" >/dev/null
CAND_BIN="$CAND_DIR/askaig"
[ -x "$CAND_BIN" ] || { echo "candidate binary missing: $CAND_BIN"; exit 1; }

# --- Baseline: BASE_REF built (Release) into build/ ----------------------------------------------
# The commit is materialised in a temporary worktree; build/ is wiped first so the CMake cache never
# points at a stale source directory.
BASE_SRC="$WORK/base-src"
cleanup() { git -C "$ROOT" worktree remove --force "$BASE_SRC" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup
echo ">> building baseline  ($BASE_REF) -> build/ ..."
git -C "$ROOT" worktree add --detach -f "$BASE_SRC" "$BASE_REF" >/dev/null
rm -rf "$ROOT/build"
cmake -S "$BASE_SRC" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$ROOT/build" -j"$CORES" >/dev/null
BASE_BIN="$ROOT/build/askaig"
[ -x "$BASE_BIN" ] || { echo "baseline binary missing: $BASE_BIN"; exit 1; }

echo ">> SPRT  cand vs base   TC=$TC  Hash=$HASH  concurrency=$CONCURRENCY  H1=elo in ($ELO0,$ELO1)  alpha=$ALPHA beta=$BETA"
echo

"$FASTCHESS" \
  -engine cmd="$CAND_BIN" name=cand \
  -engine cmd="$BASE_BIN" name=base \
  -each tc="$TC" option.Hash="$HASH" option.Threads=1 proto=uci \
  -openings file="$BOOK" format=epd order=random \
  -games 2 -rounds "$ROUNDS" -repeat \
  -sprt elo0="$ELO0" elo1="$ELO1" alpha="$ALPHA" beta="$BETA" \
  $ADJ_ARGS \
  -recover \
  -concurrency "$CONCURRENCY" \
  -ratinginterval 20 \
  -pgnout file="$WORK/games.pgn"
