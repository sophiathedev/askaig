#!/usr/bin/env bash
#
# SPRT self-play test: decides whether the CANDIDATE build is stronger than the BASELINE, by playing
# them against each other until the Sequential Probability Ratio Test reaches a verdict (H1 = the
# candidate gains in (elo0, elo1); H0 = it does not). This is the ONLY reliable measure of strength —
# node counts and tactical spot-checks confirm *correctness*, not Elo.
#
#   tools/sprt.sh [BASE]
#   tools/sprt.sh --resume RUN_ID
#   tools/sprt.sh --adopt CONFIG [BASE]
#
#     CANDIDATE  = build-pgo/askaig — a fresh PGO build of the CURRENT source tree.
#     BASE       = the opponent. Two forms, auto-detected:
#                    - a path to an EXECUTABLE (stockfish, another askaig binary, ...) -> played
#                      directly, no build. THIS is how you test against a DIFFERENT engine.
#                    - a git ref (commit/tag/branch, default HEAD) -> fresh PGO build in a throwaway
#                      worktree of that commit.
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
#                   ALPHA/BETA (0.1), ROUNDS (5000), BOOK (UHO_4060_v4.epd), ADJUDICATE (1),
#                   DEPTH (unset), CAND_SYZYGY_PATH/BASE_SYZYGY_PATH (unset), SAME_BINARY (0).
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
WORK="$HERE/work"
RUNS="$WORK/sprt-runs"
STATE_TOOL="$HERE/sprt_state.py"

usage() {
  echo "usage: tools/sprt.sh [BASE]"
  echo "       tools/sprt.sh --resume RUN_ID"
  echo "       tools/sprt.sh --adopt CONFIG [BASE]"
}

MODE=new
BASE_REF=HEAD
RESUME_RUN=""
LEGACY_CONFIG=""
case "${1:-}" in
  --resume)
    [ "$#" -eq 2 ] || { usage; exit 2; }
    MODE=resume
    RESUME_RUN="$2"
    ;;
  --adopt)
    [ "$#" -ge 2 ] && [ "$#" -le 3 ] || { usage; exit 2; }
    MODE=adopt
    LEGACY_CONFIG="$2"
    BASE_REF="${3:-HEAD}"
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    [ "$#" -le 1 ] || { usage; exit 2; }
    BASE_REF="${1:-HEAD}"
    ;;
esac

# Locate the fastchess match-runner: $FASTCHESS env wins, else search common locations.
if [ -z "${FASTCHESS:-}" ]; then
  for c in "$HOME/Downloads/fastchess-mac-arm64/fastchess" "$HERE/fastchess-src/fastchess" \
           "$(command -v fastchess 2>/dev/null || true)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then FASTCHESS="$c"; break; fi
  done
fi
[ -n "${FASTCHESS:-}" ] && [ -x "$FASTCHESS" ] || {
  echo "fastchess not found — set FASTCHESS=/path/to/fastchess (or run 'bash tools/setup.sh')"
  exit 1
}

resolve_run_dir() {
  case "$1" in
    */*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$RUNS/$1" ;;
  esac
}

if [ "$MODE" = resume ]; then
  RUN_DIR="$(resolve_run_dir "$RESUME_RUN")"
  [ -d "$RUN_DIR" ] || { echo "SPRT run missing: $RUN_DIR"; exit 1; }
  RUN_DIR="$(cd "$RUN_DIR" && pwd -P)"
  CONFIG="$RUN_DIR/fastchess.json"
  LOG="$RUN_DIR/fastchess.log"
  [ -f "$CONFIG" ] || { echo "SPRT config missing: $CONFIG"; exit 1; }
  [ -f "$RUN_DIR/checksums.sha256" ] || { echo "SPRT checksums missing: $RUN_DIR/checksums.sha256"; exit 1; }
  (cd "$RUN_DIR" && shasum -a 256 -c checksums.sha256 >/dev/null) || {
    echo "frozen engine checksum mismatch: $RUN_DIR"
    exit 1
  }
  [ "$(python3 "$STATE_TOOL" get "$CONFIG" candidate)" = "$RUN_DIR/bin/cand" ] || {
    echo "candidate path does not match the frozen run"
    exit 1
  }
  [ "$(python3 "$STATE_TOOL" get "$CONFIG" baseline)" = "$RUN_DIR/bin/base" ] || {
    echo "baseline path does not match the frozen run"
    exit 1
  }
  python3 "$STATE_TOOL" validate "$CONFIG"
  echo ">> resuming SPRT run: $RUN_DIR"
  echo
  "$FASTCHESS" -config file="$CONFIG" outname="$CONFIG" stats=true 2>&1 | tee -a "$LOG"
  exit "${PIPESTATUS[0]}"
fi

BOOK="${BOOK:-$HERE/books/UHO_4060_v4.epd}"
# Opening-book format passed to fastchess (epd|pgn). Default epd (UHO). A balanced PGN book
# (e.g. books/8moves_v3.pgn) reaches endgames far more often — set BOOK_FORMAT=pgn for it. Pair
# with ADJUDICATE=0 when testing endgame eval, or the resign rule ends won games before they
# reach the bare-king phase those terms operate in.
BOOK_FORMAT="${BOOK_FORMAT:-epd}"
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
SAME_BINARY="${SAME_BINARY:-0}"
case "$SAME_BINARY" in
  0|1) ;;
  *) echo "SAME_BINARY must be 0 or 1"; exit 1 ;;
esac
[ "$SAME_BINARY" = 0 ] || [ "$MODE" = new ] || { echo "SAME_BINARY is only supported for new runs"; exit 1; }
[ "$SAME_BINARY" = 0 ] || [ "$BASE_REF" = HEAD ] || { echo "SAME_BINARY does not accept a baseline argument"; exit 1; }

CAND_SYZYGY_PATH="${CAND_SYZYGY_PATH:-}"
CAND_SYZYGY_PROBE_DEPTH="${CAND_SYZYGY_PROBE_DEPTH:-1}"
CAND_SYZYGY_50_MOVE_RULE="${CAND_SYZYGY_50_MOVE_RULE:-true}"
CAND_SYZYGY_PROBE_LIMIT="${CAND_SYZYGY_PROBE_LIMIT:-5}"
BASE_SYZYGY_PATH="${BASE_SYZYGY_PATH:-}"
BASE_SYZYGY_PROBE_DEPTH="${BASE_SYZYGY_PROBE_DEPTH:-1}"
BASE_SYZYGY_50_MOVE_RULE="${BASE_SYZYGY_50_MOVE_RULE:-true}"
BASE_SYZYGY_PROBE_LIMIT="${BASE_SYZYGY_PROBE_LIMIT:-5}"

CAND_ENGINE_ARGS=()
if [ -n "$CAND_SYZYGY_PATH" ]; then
  [ -d "$CAND_SYZYGY_PATH" ] || { echo "candidate Syzygy path missing: $CAND_SYZYGY_PATH"; exit 1; }
  CAND_SYZYGY_PATH="$(cd "$CAND_SYZYGY_PATH" && pwd -P)"
  CAND_ENGINE_ARGS+=(
    option.SyzygyPath="$CAND_SYZYGY_PATH"
    option.SyzygyProbeDepth="$CAND_SYZYGY_PROBE_DEPTH"
    option.Syzygy50MoveRule="$CAND_SYZYGY_50_MOVE_RULE"
    option.SyzygyProbeLimit="$CAND_SYZYGY_PROBE_LIMIT"
  )
fi

BASE_ENGINE_ARGS=()
if [ -n "$BASE_SYZYGY_PATH" ]; then
  [ -d "$BASE_SYZYGY_PATH" ] || { echo "baseline Syzygy path missing: $BASE_SYZYGY_PATH"; exit 1; }
  BASE_SYZYGY_PATH="$(cd "$BASE_SYZYGY_PATH" && pwd -P)"
  BASE_ENGINE_ARGS+=(
    option.SyzygyPath="$BASE_SYZYGY_PATH"
    option.SyzygyProbeDepth="$BASE_SYZYGY_PROBE_DEPTH"
    option.Syzygy50MoveRule="$BASE_SYZYGY_50_MOVE_RULE"
    option.SyzygyProbeLimit="$BASE_SYZYGY_PROBE_LIMIT"
  )
fi

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

mkdir -p "$RUNS"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$(git -C "$ROOT" rev-parse --short HEAD)}"
case "$RUN_ID" in
  latest|*[^A-Za-z0-9._-]*) echo "invalid RUN_ID: $RUN_ID"; exit 1 ;;
esac
RUN_DIR="$RUNS/$RUN_ID"
[ ! -e "$RUN_DIR" ] || { echo "SPRT run already exists: $RUN_DIR"; exit 1; }

LEGACY_CAND=""
LEGACY_BASE=""
LEGACY_PGN=""
if [ "$MODE" = adopt ]; then
  [ -f "$LEGACY_CONFIG" ] || { echo "legacy config missing: $LEGACY_CONFIG"; exit 1; }
  LEGACY_CAND="$(python3 "$STATE_TOOL" get "$LEGACY_CONFIG" candidate)"
  LEGACY_BASE="$(python3 "$STATE_TOOL" get "$LEGACY_CONFIG" baseline)"
  LEGACY_PGN="$(python3 "$STATE_TOOL" get "$LEGACY_CONFIG" pgn)"
  [ -f "$LEGACY_PGN" ] || { echo "legacy PGN missing: $LEGACY_PGN"; exit 1; }
fi

PGO_SCRIPT="$HERE/build_pgo.sh"
[ -f "$PGO_SCRIPT" ] || { echo "PGO helper missing: $PGO_SCRIPT"; exit 1; }

if [ -n "$LEGACY_CAND" ] && [ -x "$LEGACY_CAND" ]; then
  CAND_BIN="$LEGACY_CAND"
  echo ">> adopting candidate binary: $CAND_BIN"
else
  echo ">> building candidate (current tree, PGO) -> build-pgo/ ..."
  bash "$PGO_SCRIPT" "$ROOT"
  CAND_BIN="$ROOT/build-pgo/askaig"
fi
[ -x "$CAND_BIN" ] || { echo "candidate binary missing: $CAND_BIN"; exit 1; }

BASE_SRC=""
cleanup() {
  if [ -n "$BASE_SRC" ]; then
    git -C "$ROOT" worktree remove --force "$BASE_SRC" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [ "$SAME_BINARY" = 1 ]; then
  BASE_BIN="$CAND_BIN"
  BASE_NAME=base
  echo ">> baseline = candidate binary"
elif [ -n "$LEGACY_BASE" ] && [ -x "$LEGACY_BASE" ]; then
  BASE_BIN="$LEGACY_BASE"
  BASE_NAME="$(basename "$LEGACY_BASE")"
  echo ">> adopting baseline binary: $BASE_BIN"
elif [ -f "$BASE_REF" ] && [ -x "$BASE_REF" ]; then
  BASE_BIN="$(cd "$(dirname "$BASE_REF")" && pwd)/$(basename "$BASE_REF")"   # absolute path
  BASE_NAME="$(basename "$BASE_REF")"
  echo ">> baseline = external engine: $BASE_BIN"
else
  BASE_SRC="$WORK/base-src-$RUN_ID"
  cleanup
  BASE_NAME="base"
  echo ">> building baseline ($BASE_REF, PGO) ..."
  git -C "$ROOT" worktree add --detach -f "$BASE_SRC" "$BASE_REF" >/dev/null
  bash "$PGO_SCRIPT" "$BASE_SRC"
  BASE_BIN="$BASE_SRC/build-pgo/askaig"
fi
[ -x "$BASE_BIN" ] || { echo "baseline binary missing: $BASE_BIN"; exit 1; }

mkdir -p "$RUN_DIR/bin"
install -m 755 "$CAND_BIN" "$RUN_DIR/bin/cand"
install -m 755 "$BASE_BIN" "$RUN_DIR/bin/base"
CAND_BIN="$RUN_DIR/bin/cand"
BASE_BIN="$RUN_DIR/bin/base"
CONFIG="$RUN_DIR/fastchess.json"
PGN="$RUN_DIR/games.pgn"
LOG="$RUN_DIR/fastchess.log"

TREE_STATE=clean
git -C "$ROOT" diff --quiet HEAD -- || TREE_STATE=dirty
{
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'created_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'candidate_head=%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
  printf 'candidate_tree=%s\n' "$TREE_STATE"
  printf 'base_ref=%s\n' "$BASE_REF"
  printf 'same_binary=%s\n' "$SAME_BINARY"
  printf 'mode=%s\n' "$MODE"
  printf 'candidate_syzygy_path=%s\n' "$CAND_SYZYGY_PATH"
  printf 'candidate_syzygy_probe_depth=%s\n' "$CAND_SYZYGY_PROBE_DEPTH"
  printf 'candidate_syzygy_50_move_rule=%s\n' "$CAND_SYZYGY_50_MOVE_RULE"
  printf 'candidate_syzygy_probe_limit=%s\n' "$CAND_SYZYGY_PROBE_LIMIT"
  printf 'base_syzygy_path=%s\n' "$BASE_SYZYGY_PATH"
  printf 'base_syzygy_probe_depth=%s\n' "$BASE_SYZYGY_PROBE_DEPTH"
  printf 'base_syzygy_50_move_rule=%s\n' "$BASE_SYZYGY_50_MOVE_RULE"
  printf 'base_syzygy_probe_limit=%s\n' "$BASE_SYZYGY_PROBE_LIMIT"
} > "$RUN_DIR/run.env"
(cd "$RUN_DIR" && shasum -a 256 bin/cand bin/base > checksums.sha256)
ln -sfn "$RUN_DIR" "$RUNS/latest"

if [ "$MODE" = adopt ]; then
  python3 "$STATE_TOOL" adopt "$LEGACY_CONFIG" "$LEGACY_PGN" "$CONFIG" "$PGN" "$CAND_BIN" "$BASE_BIN"
  echo ">> adopted SPRT run: $RUN_DIR"
  echo ">> resume with: tools/sprt.sh --resume $RUN_ID"
  exit 0
fi

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
[ -z "$CAND_SYZYGY_PATH" ] || echo ">> candidate Syzygy: $CAND_SYZYGY_PATH  limit=$CAND_SYZYGY_PROBE_LIMIT"
[ -z "$BASE_SYZYGY_PATH" ] || echo ">> baseline Syzygy: $BASE_SYZYGY_PATH  limit=$BASE_SYZYGY_PROBE_LIMIT"
echo

"$FASTCHESS" \
  -engine cmd="$CAND_BIN" name=cand ${CAND_ENGINE_ARGS[@]+"${CAND_ENGINE_ARGS[@]}"} \
  -engine cmd="$BASE_BIN" name="$BASE_NAME" ${BASE_ENGINE_ARGS[@]+"${BASE_ENGINE_ARGS[@]}"} \
  -each "$LIMIT" option.Hash="$HASH" option.Threads=1 proto=uci \
  -openings file="$BOOK" format="$BOOK_FORMAT" order=random \
  -games 2 -rounds "$ROUNDS" -repeat \
  $SPRT_ARGS \
  $ADJ_ARGS \
  -recover \
  -config outname="$CONFIG" stats=true \
  -autosaveinterval 20 \
  -concurrency "$CONCURRENCY" \
  -ratinginterval 20 \
  -pgnout file="$PGN" append=true 2>&1 | tee -a "$LOG"
