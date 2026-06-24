#!/usr/bin/env python3
"""Extract labeled training positions from self-play PGNs for Texel tuning.

Methodology (the zurichess recipe, adapted):
  - only games with a real result (terminations like time forfeit / abandoned / illegal move
    produce wrong labels and are skipped)
  - skip the first 16 plies (the UHO book is deliberately unbalanced; those positions' labels
    say more about the book than the eval)
  - skip positions in check and positions whose played move was a capture or a promotion
    (tactical positions: the static eval is not supposed to explain them)
  - skip the last 4 plies of decisive games (mating sequences label "being mated" not "eval")
  - sample at most MAX_PER_GAME of the survivors per game (evenly spaced), dedupe globally
    by the board part of the FEN

Output lines:  <FEN> <result>     with result in {1.0, 0.5, 0.0} from White's point of view.

Usage:  python3 tools/extract.py [games.pgn ...] [-o data.book]
        (defaults: tools/work/games.pgn -> tools/work/data.book)
"""

import argparse
import sys
import time

import chess
import chess.pgn

MIN_PLY = 16  # skip the book/opening phase
MATE_TAIL = 4  # plies to drop before a checkmate ending
MAX_PER_GAME = 24

RESULTS = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
BAD_TERMINATIONS = {"time forfeit", "abandoned", "illegal move", "unterminated"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgns", nargs="*", default=["tools/work/games.pgn"])
    ap.add_argument("-o", "--out", default="tools/work/data.book")
    args = ap.parse_args()

    seen: set[str] = set()
    n_games = n_used = n_pos = 0
    t0 = time.time()

    with open(args.out, "w") as out:
        for path in args.pgns:
            with open(path, errors="replace") as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    n_games += 1
                    if n_games % 200 == 0:
                        el = time.time() - t0
                        print(
                            f"  {n_games} games ({n_games / el:.0f}/s) -> {n_pos} positions",
                            flush=True,
                        )

                    result = RESULTS.get(game.headers.get("Result", "*"))
                    if result is None:
                        continue
                    if game.headers.get("Termination", "").lower() in BAD_TERMINATIONS:
                        continue

                    # Collect candidate (fen, ply) pairs along the game.
                    board = game.board()
                    moves = list(game.mainline_moves())
                    total = len(moves)
                    is_mate_end = False
                    cands = []
                    for ply, move in enumerate(moves):
                        if (
                            ply >= MIN_PLY
                            and not board.is_check()
                            and not board.is_capture(move)
                            and move.promotion is None
                        ):
                            cands.append((board.fen(), ply))
                        board.push(move)
                    if board.is_checkmate():
                        is_mate_end = True

                    if is_mate_end:
                        cands = [c for c in cands if c[1] < total - MATE_TAIL]
                    if not cands:
                        continue

                    # Evenly spaced sample, then global dedupe by the board field.
                    step = max(1, len(cands) // MAX_PER_GAME)
                    n_used += 1
                    for fen, _ply in cands[::step][:MAX_PER_GAME]:
                        key = fen.split(" ")[0] + fen.split(" ")[1]  # placement + side
                        if key in seen:
                            continue
                        seen.add(key)
                        out.write(f"{fen} {result}\n")
                        n_pos += 1

    el = time.time() - t0
    print(
        f"done: {n_games} games read, {n_used} usable -> {n_pos} unique positions "
        f"in {el:.1f}s -> {args.out}"
    )
    if n_pos < 20000:
        print("warning: fewer than 20k positions - consider generating more games", file=sys.stderr)


if __name__ == "__main__":
    main()
