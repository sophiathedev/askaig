#!/usr/bin/env bash
#
# Self-play data generation for Texel tuning: the CURRENT build plays itself for a fixed number of
# opening pairs (no SPRT — both sides are the same binary, we only want games with honest labels),
# then tools/extract.py turns the PGN into a labeled position book.
#
#   tools/datagen.sh [ROUNDS]      (default 2000 rounds = 4000 games)
#
# Tunables via env: TC (10+0.1 — longer than SPRT's 5+0.05: better moves, better labels),
#                   CONCURRENCY (P-cores - 1), HASH (16), BOOK (UHO_4060_v3.epd),
#                   OUT (tools/work/datagen.pgn), DATA (tools/work/data.book).
#
# Adjudication stays ON: a twosided 700cp resign or a 20cp draw window produces labels as reliable
# as playing the game out, at half the wall-clock. Time forfeits/aborts produce WRONG labels, but
# extract.py skips those terminations. The PGN appends across runs; extract.py dedupes positions,
# so re-running only grows the book. Regenerate after every strength jump (each merged SPRT round):
# stronger games -> better labels -> better tuning -> stronger engine.
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

if [ -z "${FASTCHESS:-}" ]; then
  for c in "$HOME/Downloads/fastchess-mac-arm64/fastchess" "$HERE/fastchess-src/fastchess" \
           "$(command -v fastchess 2>/dev/null || true)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then FASTCHESS="$c"; break; fi
  done
fi
BOOK="${BOOK:-$HERE/books/UHO_4060_v3.epd}"
[ -n "${FASTCHESS:-}" ] && [ -x "$FASTCHESS" ] || { echo "fastchess not found - run 'bash tools/setup.sh'"; exit 1; }
[ -f "$BOOK" ] || { echo "opening book missing - run 'bash tools/setup.sh'"; exit 1; }

ROUNDS="${1:-2000}"
TC="${TC:-10+0.1}"
HASH="${HASH:-16}"
PCORES="$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || sysctl -n hw.physicalcpu 2>/dev/null || echo 4)"
CONCURRENCY="${CONCURRENCY:-$(( PCORES > 2 ? PCORES - 1 : 1 ))}"
OUT="${OUT:-$HERE/work/datagen.pgn}"
DATA="${DATA:-$HERE/work/data.book}"
mkdir -p "$HERE/work"

echo ">> building the current tree -> cmake-build-release/ ..."
if [ ! -f "$ROOT/cmake-build-release/CMakeCache.txt" ]; then
  cmake --preset release-simd -S "$ROOT" >/dev/null
fi
cmake --build "$ROOT/cmake-build-release" -j >/dev/null
BIN="$ROOT/cmake-build-release/askaig"

echo ">> datagen: $ROUNDS rounds ($((ROUNDS * 2)) games) at TC=$TC, concurrency=$CONCURRENCY -> $OUT"
echo

"$FASTCHESS" \
  -engine cmd="$BIN" name=dataA \
  -engine cmd="$BIN" name=dataB \
  -each tc="$TC" option.Hash="$HASH" option.Threads=1 proto=uci \
  -openings file="$BOOK" format=epd order=random \
  -games 2 -rounds "$ROUNDS" -repeat \
  -draw movenumber=34 movecount=8 score=20 \
  -resign movecount=4 score=700 twosided=true \
  -maxmoves 200 \
  -recover \
  -concurrency "$CONCURRENCY" \
  -ratinginterval 100 \
  -pgnout file="$OUT"

echo
echo ">> extracting labeled positions -> $DATA"
python3 "$HERE/extract.py" "$OUT" -o "$DATA"
echo ">> next: ./cmake-build-release/askaig tune $DATA"
