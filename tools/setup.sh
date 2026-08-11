#!/usr/bin/env bash
# one-time SPRT setup
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CXX="${CXX:-clang++}"
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

# prefer an existing fastchess binary
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

# opening book
mkdir -p "$HERE/books"
if [ ! -f "$HERE/books/UHO_Lichess_4852_v1.epd" ]; then
  curl -L --fail --retry 3 -o "$HERE/books/UHO_Lichess_4852_v1.epd.zip" \
    https://github.com/official-stockfish/books/raw/master/UHO_Lichess_4852_v1.epd.zip
  unzip -o "$HERE/books/UHO_Lichess_4852_v1.epd.zip" -d "$HERE/books"
  rm -f "$HERE/books/UHO_Lichess_4852_v1.epd.zip"
fi
echo "book: $(wc -l < "$HERE/books/UHO_Lichess_4852_v1.epd" | tr -d ' ') openings"

echo "setup OK — now run:  tools/sprt.sh"
