#pragma once

// Universal Chess Interface (UCI) front-end. Runs the command loop on stdin/stdout until
// "quit". Search/evaluation is not implemented yet; the protocol and board-state commands
// (uci, isready, ucinewgame, position, go perft, d, ...) are fully functional.
namespace uci {
  void loop();

  // Runs the fixed-position benchmark (the "bench" command / `askaig bench [depth]` CLI form) and
  // prints per-position node counts plus a total — a deterministic search signature. depth <= 0
  // uses the default.
  void bench(int depth = 0);

  // Enables/disables debug mode at startup (the `--debug` CLI flag). Same effect as the `debug on`
  // UCI command: exposes the search-tuning options to `uci`/`setoption`. Off in production.
  void set_debug(bool on);
} // namespace uci
