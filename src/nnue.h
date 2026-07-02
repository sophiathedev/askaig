#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include "position.h"
#include "types.h"

// NNUE evaluation: a (768 -> HL)x2 -> 1 perspective network with SCReLU activation,
// evaluated incrementally (the accumulator is updated by the pieces a move touches, not
// recomputed). The accumulator lives OUTSIDE Position — Position and perft are untouched;
// each (future) search thread owns one nnue::Evaluator.
//
// Feature set: plain 768 = piece(12) x square(64), one half per perspective:
//   white pov: idx = 384*color(pc) + 64*type(pc) + sq
//   black pov: idx = 384*(color(pc)^1) + 64*type(pc) + (sq^56)   (vertical mirror)
// At eval time the halves are concatenated [stm, ~stm]; the output is centipawns for the
// side to move.
//
// Quantization (matching tools/nnue/export.py): feature-transformer weights/biases are
// int16 at scale QA, output weights int16 at scale QB, output bias int32 at scale QA*QB.
// SCReLU is computed as v*(v*w) with v = clamp(acc, 0, QA): v*w must fit int16, which the
// trainer guarantees by clipping float weights to +-1.98 (255*127 < 32768).
namespace nnue {

  constexpr int FEATURES = 768;
  constexpr int HL       = 256; // hidden layer size; a net file with a different value is rejected
  constexpr int QA       = 255;
  constexpr int QB       = 64;
  constexpr int SCALE    = 400;

  // Deepest make/unmake stack the Evaluator supports (the future search's ply ceiling).
  constexpr int MAX_PLY = 256;

  // --- Net file (.nnue) ------------------------------------------------------------------------
  // Little-endian, 32-byte header followed by the raw quantized weights:
  //   int16 ft_w[768][HL] (feature-major), int16 ft_b[HL], int16 out_w[2*HL], int32 out_b
  struct NetHeader {
    char     magic[4]; // "AKNN"
    uint32_t version; // 1
    uint32_t features; // 768
    uint32_t hl; // HL
    uint32_t buckets; // 1
    uint16_t qa, qb, scale; // 255, 64, 400
    uint8_t  activation; // 1 = SCReLU (0 = CReLU, unused)
    uint8_t  pad[5]; // zero
  };
  static_assert(sizeof(NetHeader) == 32);

  // Loads a net, replacing the current one. On failure the current net is kept and `err`
  // (when non-null) explains why. load_buffer validates the header and total size.
  // load_embedded loads the net baked into the binary (networks/default.nnue at build time).
  bool load_file(const std::string &path, std::string *err = nullptr);
  bool load_buffer(const unsigned char *data, size_t size, std::string *err = nullptr);
  bool load_embedded(std::string *err = nullptr);
  bool loaded();

  // --- Incremental evaluation ------------------------------------------------------------------

  // The feature changes one move makes: at most 2 additions and 2 removals (castling).
  struct DirtyPiece {
    int8_t n_add, n_sub;
    Piece  add_pc[2], sub_pc[2];
    Square add_sq[2], sub_sq[2];
  };

  struct Accumulator {
    alignas(64) int16_t v[2][HL]; // [Color][HL]: the two perspective halves
    DirtyPiece dp; // how to build this from the parent (when !computed)
    bool       computed;
  };

  // One per (future) search thread. Usage around make/unmake:
  //   ev.reset(rootpos);                 // once per new root
  //   ev.push(pos, m); pos.play<C>(m);   // push BEFORE play (reads the captured piece)
  //   ... int cp = ev.evaluate(pos); ... // lazy: cost is paid only if a node is evaluated
  //   pos.undo<C>(m); ev.pop();
  class Evaluator {
  public:
    Evaluator();
    void reset(const Position &pos);
    void push(const Position &before, Move m);
    void push_null(); // pairs with Position::play_null (no feature changes)
    void pop();
    int  evaluate(const Position &pos); // centipawns, side-to-move perspective

  private:
    std::unique_ptr<Accumulator[]> stack; // MAX_PLY+8 entries (~270 KB) — heap, not per-Position
    int                            top;
  };

  // One-shot evaluation via a full accumulator refresh (the UCI `eval` command, tests).
  int evaluate_refresh(const Position &pos);

} // namespace nnue
