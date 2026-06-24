#!/usr/bin/env bash
#
# One-time setup for the SPRT harness: makes sure a fastchess match-runner and a balanced opening book
# are available. Run once; afterwards use tools/sprt.sh. Re-running is cheap (skips done work).
# Requires: curl, unzip (and, only if fastchess must be built, git + clang++).
#
#   bash tools/setup.sh
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CXX="${CXX:-clang++}"
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

# 1) fastchess — a fast UCI match runner with a built-in SPRT, the standard tool for engine testing.
#    Use a binary if one is already present ($FASTCHESS, a manual download, or a prior build); only
#    build from source (https://github.com/Disservin/fastchess) as a last resort. The build needs
#    clang++ because Homebrew GCC cannot parse the macOS SDK headers.
FASTCHESS="${FASTCHESS:-}"
if [ -z "$FASTCHESS" ]; then
  for c in "$HOME/Downloads/fastchess-mac-arm64/fastchess" "$HERE/fastchess-src/fastchess" \
           "$(command -v fastchess 2>/dev/null || true)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then FASTCHESS="$c"; break; fi
  done
fi
if [ -z "$FASTCHESS" ] || [ ! -x "$FASTCHESS" ]; then
  echo "no fastchess binary found — building from source ..."
  if [ ! -d "$HERE/fastchess-src/.git" ]; then
    rm -rf "$HERE/fastchess-src"
    git clone --depth 1 https://github.com/Disservin/fastchess.git "$HERE/fastchess-src"
  fi
  make -C "$HERE/fastchess-src" -j"$JOBS" CXX="$CXX"
  FASTCHESS="$HERE/fastchess-src/fastchess"
fi
echo "fastchess: $FASTCHESS"
echo "           $("$FASTCHESS" --version 2>/dev/null | head -1)"

# 2) Opening book — UHO_4060_v3 (balanced "unbalanced human openings", ~0.6-pawn starts): the de-facto
#    SPRT book. A book is REQUIRED: at Threads=1 the engine is deterministic, so without varied start
#    positions every game would be identical.
mkdir -p "$HERE/books"
if [ ! -f "$HERE/books/UHO_4060_v3.epd" ]; then
  curl -L --fail -o "$HERE/books/uho.zip" \
    https://github.com/official-stockfish/books/raw/master/UHO_4060_v3.epd.zip
  unzip -o "$HERE/books/uho.zip" -d "$HERE/books"
  rm -f "$HERE/books/uho.zip"
fi
echo "book: $(wc -l < "$HERE/books/UHO_4060_v3.epd" | tr -d ' ') openings"

echo "setup OK — now run:  tools/sprt.sh"
