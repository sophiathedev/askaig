#include "uci.h"
#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include "position.h"
#include "search.h"
#include "tables.h"
#include "tt.h"
#include "types.h"

namespace {

  constexpr auto ENGINE_NAME   = "Askaig 20260608";
  constexpr auto ENGINE_AUTHOR = "the Askaig developers (see AUTHORS file)";

  // Number of search threads (the "Threads" UCI option).
  int g_threads = 1;

  // Largest sensible thread count to advertise/accept.
  unsigned max_threads() {
    unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : hw;
  }

  // Converts an internal Move to its UCI string. Two special cases:
  //  - castling is stored king-to-rook (e1h1 / e1a1) but UCI expects king-two-squares
  //    (e1g1 / e1c1), so the destination is recomputed from the king square;
  //  - promotions get a trailing piece letter (low 2 flag bits: 0=n 1=b 2=r 3=q).
  std::string move_to_uci(Move m) {
    Square    from = m.from();
    Square    to   = m.to();
    MoveFlags f    = m.flags();

    if (f == OO)
      to = Square(from + 2);
    else if (f == OOO)
      to = Square(from - 2);

    std::string s = SQSTR[from];
    s += SQSTR[to];

    if ((f >= PR_KNIGHT && f <= PR_QUEEN) || (f >= PC_KNIGHT && f <= PC_QUEEN)) {
      constexpr char promo[] = {'n', 'b', 'r', 'q'};
      s += promo[f & 0b11];
    }
    return s;
  }

  // Plays the legal move whose UCI string matches `mstr`. Returns false if none matches.
  template<Color Us>
  bool try_play(Position &pos, const std::string &mstr) {
    MoveList<Us> list(pos);
    for (Move m: list)
      if (move_to_uci(m) == mstr) {
        pos.play<Us>(m);
        return true;
      }
    return false;
  }

  bool play_uci_move(Position &pos, const std::string &mstr) {
    return pos.turn() == WHITE ? try_play<WHITE>(pos, mstr) : try_play<BLACK>(pos, mstr);
  }

  // Formats a search score as a UCI "score ..." field. Mate scores (within MAX_MATE_PLY of
  // MATE) are reported as "mate <n>" in moves, with the sign giving who is mating.
  std::string format_score(int score) {
    constexpr int MAX_MATE_PLY = 256;
    if (score > search::MATE - MAX_MATE_PLY)
      return "mate " + std::to_string((search::MATE - score + 1) / 2);
    if (score < -(search::MATE - MAX_MATE_PLY))
      return "mate " + std::to_string(-((search::MATE + score + 1) / 2));
    return "cp " + std::to_string(score);
  }

  // perft (move-generation node count) — used by "go perft <depth>".
  template<Color Us>
  uint64_t perft(Position &p, int depth) {
    MoveList<Us> list(p);
    if (depth <= 1)
      return static_cast<uint64_t>(list.size());

    uint64_t nodes = 0;
    for (Move m: list) {
      p.play<Us>(m);
      nodes += perft<~Us>(p, depth - 1);
      p.undo<Us>(m);
    }
    return nodes;
  }

  template<Color Us>
  void perft_divide(Position &p, int depth) {
    uint64_t     total = 0;
    MoveList<Us> list(p);

    const auto start = std::chrono::steady_clock::now();
    for (Move m: list) {
      uint64_t n = 1;
      if (depth > 1) {
        p.play<Us>(m);
        n = perft<~Us>(p, depth - 1);
        p.undo<Us>(m);
      }
      std::cout << move_to_uci(m) << ": " << n << "\n";
      total += n;
    }
    const auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();

    std::cout << "\nNodes searched: " << total << "\n";
    std::cout << "Speed: "
              << (us > 0 ? static_cast<uint64_t>(static_cast<double>(total) * 1'000'000.0 / static_cast<double>(us))
                         : 0)
              << " nodes/s\n";
  }

  void run_perft(Position &pos, int depth) {
    if (pos.turn() == WHITE)
      perft_divide<WHITE>(pos, depth);
    else
      perft_divide<BLACK>(pos, depth);
  }

  // Handles "position [startpos | fen <fen>] [moves <m1> ...]".
  void position_cmd(std::optional<Position> &pos, std::istringstream &is) {
    std::string token;
    std::string fen;
    is >> token;

    if (token == "startpos") {
      fen = DEFAULT_FEN;
      is >> token; // consume "moves" if present
    } else if (token == "fen") {
      while (is >> token && token != "moves") {
        if (!fen.empty())
          fen += ' ';
        fen += token;
      }
    } else {
      return; // malformed
    }

    pos.emplace(); // set() assumes a freshly-constructed Position
    Position::set(fen, *pos);

    if (token == "moves") {
      std::string mv;
      while (is >> mv)
        if (!play_uci_move(*pos, mv)) {
          std::cout << "info string ignoring illegal/unknown move " << mv << "\n";
          break;
        }
    }
  }

  constexpr int DEFAULT_DEPTH = 8;

  // Handles "go ...". "go perft <depth>" counts move-generation nodes; otherwise it runs an
  // iterative-deepening, multi-threaded (Lazy SMP) negamax/alpha-beta search ("go depth <n>",
  // default DEFAULT_DEPTH), printing an info line per depth. Time-control args are ignored.
  void go_cmd(Position &pos, std::istringstream &is) {
    int         depth = DEFAULT_DEPTH;
    std::string token;
    while (is >> token) {
      if (token == "perft") {
        int d = 1;
        is >> d;
        run_perft(pos, d);
        return;
      }
      if (token == "depth")
        is >> depth;
    }

    search::Result r =
            search::think(pos, depth, g_threads, [](int d, const search::Result &res, uint64_t nodes, long long ms) {
              const uint64_t nps = ms > 0 ? nodes * 1000 / static_cast<uint64_t>(ms) : nodes * 1000;
              std::cout << "info depth " << d << " score " << format_score(res.score) << " nodes " << nodes << " nps "
                        << nps << " time " << ms;
              if (!res.pv.empty()) {
                std::cout << " pv";
                for (Move m: res.pv)
                  std::cout << " " << move_to_uci(m);
              } else if (res.best.to_from() != 0)
                std::cout << " pv " << move_to_uci(res.best);
              std::cout << "\n" << std::flush;
            });

    const bool has_move = r.best.to_from() != 0;
    std::cout << "bestmove " << (has_move ? move_to_uci(r.best) : "0000") << "\n";
  }

} // namespace

void uci::loop() {
  std::optional<Position> pos;
  pos.emplace();
  Position::set(DEFAULT_FEN, *pos);

  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream is(line);
    std::string        cmd;
    is >> cmd;

    if (cmd == "uci") {
      std::cout << "id name " << ENGINE_NAME << "\n";
      std::cout << "id author " << ENGINE_AUTHOR << "\n";
      std::cout << "option name Hash type spin default " << tt::DEFAULT_HASH_MB << " min 1 max 65536\n";
      std::cout << "option name Threads type spin default 1 min 1 max " << max_threads() << "\n";
      std::cout << "uciok\n";
    } else if (cmd == "isready") {
      std::cout << "readyok\n";
    } else if (cmd == "ucinewgame") {
      pos.emplace();
      Position::set(DEFAULT_FEN, *pos);
      tt::clear();
    } else if (cmd == "position") {
      position_cmd(pos, is);
    } else if (cmd == "go") {
      go_cmd(*pos, is);
    } else if (cmd == "d" || cmd == "display") {
      std::cout << *pos;
    } else if (cmd == "setoption") {
      // setoption name <id> [value <x>]
      std::string token;
      std::string name;
      is >> token >> name >> token; // "name", <id>, "value"
      if (name == "Hash") {
        int mb = 0;
        if (is >> mb) {
          mb = mb < 1 ? 1 : (mb > 65536 ? 65536 : mb);
          tt::resize(static_cast<size_t>(mb));
        }
      } else if (name == "Threads") {
        int t = 0;
        if (is >> t)
          g_threads = t < 1 ? 1 : (t > 1024 ? 1024 : t);
      }
    } else if (cmd == "stop" || cmd == "ponderhit" || cmd == "register" || cmd.empty()) {
      // accepted but no-op
    } else if (cmd == "quit" || cmd == "exit") {
      break;
    }
    // unknown commands are silently ignored, as the UCI spec requires

    std::cout.flush();
  }
}
