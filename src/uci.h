#pragma once

// Universal Chess Interface (UCI) front-end. Runs the command loop on stdin/stdout until
// "quit". Search/evaluation is not implemented yet; the protocol and board-state commands
// (uci, isready, ucinewgame, position, go perft, d, ...) are fully functional.
namespace uci {
  void loop();
} // namespace uci
