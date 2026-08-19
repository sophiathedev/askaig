#!/usr/bin/env bash
#
# Two-stage PGO (profile-guided optimization) build -> build-pgo/askaig, via the CMake pgo-generate /
# pgo-use presets. PGO orchestration (instrument -> run a training workload -> merge -> rebuild) can't
# live in CMake alone, so this script drives it; the FLAGS and build config are first-class CMake.
#
# A behaviour-NEUTRAL speed win: same node count -> same moves, just better code layout, so the engine
# reaches more depth per second in real games (~3-10% nps; Stockfish ships PGO for the same reason).
# No SPRT needed for correctness — only the node-count check below (must match your normal build).
#
# Builds into build-pgo/ ONLY (never build/ or cmake-build-release/), so it is safe to run alongside an
# SPSA/SPRT run — but the compile competes for CPU, so prefer running it when cores are free.
#
#   PGO_SYZYGY_PATH=/path/to/tables bash tools/build_pgo.sh [SOURCE_ROOT]
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${1:-$HERE/..}" && pwd)"
DIR="$ROOT/build-pgo"

# llvm-profdata: Apple ships it behind xcrun; Homebrew/LLVM puts it on PATH.
if command -v xcrun >/dev/null 2>&1 && xcrun -f llvm-profdata >/dev/null 2>&1; then
  PROFDATA=(xcrun llvm-profdata)
elif command -v llvm-profdata >/dev/null 2>&1; then
  PROFDATA=(llvm-profdata)
else
  echo "ERROR: llvm-profdata not found (needed to merge the profile). Install LLVM or Xcode CLT." >&2
  exit 1
fi
NCPU="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

cd "$ROOT"
rm -rf "$DIR"

echo ">> [1/4] instrument  (preset pgo-generate) ..."
cmake --preset pgo-generate >/dev/null
cmake --build --preset pgo-generate -j"$NCPU" >/dev/null

echo ">> [2/4] train on perft, NNUE, search, and Syzygy ..."
printf 'position startpos\ngo perft 6\nquit\n' | LLVM_PROFILE_FILE="$DIR/perft-%p.profraw" "$DIR/askaig" >/dev/null 2>&1
printf 'position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1\ngo perft 5\nquit\n' | LLVM_PROFILE_FILE="$DIR/kiwipete-%p.profraw" "$DIR/askaig" >/dev/null 2>&1
printf 'position startpos\ngo perft 5 noncache\nquit\n' | LLVM_PROFILE_FILE="$DIR/noncache-%p.profraw" "$DIR/askaig" >/dev/null 2>&1
printf 'bench evalnps\nquit\n' | LLVM_PROFILE_FILE="$DIR/evalnps-%p.profraw" "$DIR/askaig" >/dev/null 2>&1

SYZYGY_DIR="${PGO_SYZYGY_PATH:-$ROOT/tools/syzygy/3-5}"
if [ ! -f "$SYZYGY_DIR/KPvK.rtbw" ] || [ ! -f "$SYZYGY_DIR/KPvK.rtbz" ]; then
  SYZYGY_DIR="$DIR/pgo-syzygy"
  mkdir -p "$SYZYGY_DIR"
  curl -fsSL https://tablebase.sesse.net/syzygy/3-4-5/KPvK.rtbw -o "$SYZYGY_DIR/KPvK.rtbw"
  curl -fsSL https://tablebase.sesse.net/syzygy/3-4-5/KPvK.rtbz -o "$SYZYGY_DIR/KPvK.rtbz"
  WDL_HASH="$(cmake -E sha256sum "$SYZYGY_DIR/KPvK.rtbw")"
  DTZ_HASH="$(cmake -E sha256sum "$SYZYGY_DIR/KPvK.rtbz")"
  [ "${WDL_HASH%% *}" = 02fd2bdcd92821c869458f019f63524def1c9e7a2862ca8d2b16dc7712b16183 ]
  [ "${DTZ_HASH%% *}" = ad01af2076183c90ccee662ca6c5d74da3222ff1fee47c01b5d4e56ed01938de ]
fi
printf 'setoption name SyzygyPath value %s\nsetoption name SyzygyProbeLimit value 5\nbench 16\nquit\n' "$SYZYGY_DIR" \
  | LLVM_PROFILE_FILE="$DIR/search-%p.profraw" "$DIR/askaig" >/dev/null 2>&1

echo ">> [3/4] merge profile -> build-pgo/pgo.profdata ..."
"${PROFDATA[@]}" merge -output="$DIR/pgo.profdata" "$DIR"/*.profraw

echo ">> [4/4] optimize  (preset pgo-use) ..."
cmake --preset pgo-use >/dev/null
cmake --build --preset pgo-use -j"$NCPU" >/dev/null

echo
echo ">> PGO binary: $DIR/askaig"
echo ">> perft check (node count MUST match your normal build/askaig — proves behaviour-neutral):"
printf 'position startpos\ngo perft 6\nquit\n' | "$DIR/askaig" 2>/dev/null | grep -E 'Nodes searched|Speed'
echo ">> compare nodes/s to the normal build:  printf 'position startpos\\ngo perft 6\\nquit\\n' | ./build/askaig | grep Speed"
echo ">> faster + same nodes -> ship build-pgo/askaig (or wire -DPGO=use into your release build)."
