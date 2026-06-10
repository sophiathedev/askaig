#!/usr/bin/env bash
#
# SPRT self-play test: decides whether the CANDIDATE build is stronger than the BASELINE, by playing
# them against each other until the Sequential Probability Ratio Test reaches a verdict (H1 = the
# candidate gains in (elo0, elo1); H0 = it does not). This is the ONLY reliable measure of strength —
# node counts and tactical spot-checks confirm *correctness*, not Elo.
#
#   tools/sprt.sh [BASE_REF] [TEST_REF]
#
#     BASE_REF   git ref for the baseline engine   (default: HEAD)
#     TEST_REF   git ref for the candidate engine   (default: the current working tree)
#
# Typical loop — edit eval, then test the working tree against the last commit:
#     tools/sprt.sh
# If you already committed the change, compare against its parent:
#     tools/sprt.sh HEAD~1
# Compare two specific commits/tags:
#     tools/sprt.sh v1 v2
#
# Tunables via env: TC (5+0.05), HASH (16), CONCURRENCY (P-cores - 1), ELO0/ELO1 (0/10),
#                   ALPHA/BETA (0.1), ROUNDS (5000), BOOK (UHO_4060_v3.epd).
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
TEST_REF="${2:-}" # empty => current working tree (includes uncommitted edits)

WORK="$HERE/work"
mkdir -p "$WORK"

# Build a git ref into its own worktree+binary; an empty ref builds the current working tree in place.
# Echoes the resulting binary path.
build() { # build <ref|""> <outdir>
  local ref="$1" out="$2" src
  if [ -z "$ref" ]; then
    src="$ROOT" # the live working tree (out-of-source build, so cmake-build-* is untouched)
  else
    src="$out/src"
    [ -d "$src" ] && git -C "$ROOT" worktree remove --force "$src" >/dev/null 2>&1 || true
    git -C "$ROOT" worktree add --detach -f "$src" "$ref" >/dev/null
  fi
  cmake -S "$src" -B "$out/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$out/build" -j"$CORES" >/dev/null
  echo "$out/build/askaig"
}

cleanup() {
  for wt in "$WORK/base/src" "$WORK/cand/src"; do
    [ -d "$wt" ] && git -C "$ROOT" worktree remove --force "$wt" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

echo ">> building baseline  ($BASE_REF) ..."
BASE_BIN="$(build "$BASE_REF" "$WORK/base")"
echo ">> building candidate (${TEST_REF:-working tree}) ..."
CAND_BIN="$(build "$TEST_REF" "$WORK/cand")"

echo ">> SPRT  cand vs base   TC=$TC  Hash=$HASH  concurrency=$CONCURRENCY  H1=elo in ($ELO0,$ELO1)  alpha=$ALPHA beta=$BETA"
echo

"$FASTCHESS" \
  -engine cmd="$CAND_BIN" name=cand \
  -engine cmd="$BASE_BIN" name=base \
  -each tc="$TC" option.Hash="$HASH" option.Threads=1 proto=uci \
  -openings file="$BOOK" format=epd order=random \
  -games 2 -rounds "$ROUNDS" -repeat \
  -sprt elo0="$ELO0" elo1="$ELO1" alpha="$ALPHA" beta="$BETA" \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=700 \
  -recover \
  -concurrency "$CONCURRENCY" \
  -ratinginterval 20 \
  -pgnout file="$WORK/games.pgn"
