#include "nnue.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <fstream>
#include <vector>

// The loader memcpys little-endian int16/int32 weight blocks straight into memory; a
// big-endian port would need byte swaps here, so fail loudly at compile time instead.
static_assert(std::endian::native == std::endian::little, "NNUE net loading assumes a little-endian host");

namespace {

  using namespace nnue;

  // The quantized network, in the exact layout the kernels read. One global copy (nets are
  // immutable after load; every Evaluator/thread reads the same weights).
  struct Network {
    alignas(64) int16_t ft_w[FEATURES * HL]; // feature-major: row f = ft_w[f*HL .. f*HL+HL)
    alignas(64) int16_t ft_b[HL];
    alignas(64) int16_t out_w[2 * HL]; // [0,HL) = stm half, [HL,2HL) = the other half
    int32_t out_b;
  };

  Network g_net;
  bool    g_loaded = false;

  // Feature index of (pc, s) from `persp`'s point of view (see nnue.h for the layout).
  inline int feature_index(Color persp, Piece pc, Square s) {
    const int c = color_of(pc);
    const int t = type_of(pc);
    return persp == WHITE ? 384 * c + 64 * t + s : 384 * (c ^ 1) + 64 * t + (s ^ 56);
  }

  inline const int16_t *ft_row(Color persp, Piece pc, Square s) {
    return &g_net.ft_w[feature_index(persp, pc, s) * HL];
  }

  // --- Kernels ---------------------------------------------------------------------------------
  // Scalar reference implementations. The SIMD (AVX2/NEON) versions must produce bit-identical
  // results — everything here is exact integer math, asserted by `selftest nnue`.

  // Rebuilds both perspective halves of `acc` from scratch (bias + every piece on the board).
  void acc_refresh(Accumulator &acc, const Position &pos) {
    std::memcpy(acc.v[WHITE], g_net.ft_b, sizeof(g_net.ft_b));
    std::memcpy(acc.v[BLACK], g_net.ft_b, sizeof(g_net.ft_b));
    for (int pc = WHITE_PAWN; pc <= BLACK_KING; ++pc) {
      if (pc > WHITE_KING && pc < BLACK_PAWN)
        continue; // the gap in the Piece enum (6, 7)
      Bitboard bb = pos.bitboard_of(Piece(pc));
      while (bb) {
        const Square   s = pop_lsb(&bb);
        const int16_t *w = ft_row(WHITE, Piece(pc), s);
        const int16_t *b = ft_row(BLACK, Piece(pc), s);
        for (int i = 0; i < HL; ++i)
          acc.v[WHITE][i] += w[i];
        for (int i = 0; i < HL; ++i)
          acc.v[BLACK][i] += b[i];
      }
    }
    acc.computed = true;
  }

  // Builds `child` from `parent` by applying child.dp, both perspectives in one pass.
  // Shapes: quiet/promo = (1 add, 1 sub); capture/ep = (1, 2); castling = (2, 2); null = (0, 0).
  void acc_apply(Accumulator &child, const Accumulator &parent) {
    const DirtyPiece &dp = child.dp;
    for (int p = 0; p < 2; ++p) {
      const Color    persp = Color(p);
      const int16_t *src   = parent.v[p];
      int16_t       *dst   = child.v[p];

      const int16_t *a0 = dp.n_add > 0 ? ft_row(persp, dp.add_pc[0], dp.add_sq[0]) : nullptr;
      const int16_t *a1 = dp.n_add > 1 ? ft_row(persp, dp.add_pc[1], dp.add_sq[1]) : nullptr;
      const int16_t *s0 = dp.n_sub > 0 ? ft_row(persp, dp.sub_pc[0], dp.sub_sq[0]) : nullptr;
      const int16_t *s1 = dp.n_sub > 1 ? ft_row(persp, dp.sub_pc[1], dp.sub_sq[1]) : nullptr;

      if (a1) { // castling: 2 adds, 2 subs
        for (int i = 0; i < HL; ++i)
          dst[i] = static_cast<int16_t>(src[i] + a0[i] + a1[i] - s0[i] - s1[i]);
      } else if (s1) { // capture / en passant: 1 add, 2 subs
        for (int i = 0; i < HL; ++i)
          dst[i] = static_cast<int16_t>(src[i] + a0[i] - s0[i] - s1[i]);
      } else if (a0) { // quiet / promotion: 1 add, 1 sub
        for (int i = 0; i < HL; ++i)
          dst[i] = static_cast<int16_t>(src[i] + a0[i] - s0[i]);
      } else { // null move: no feature changes
        std::memcpy(dst, src, sizeof(int16_t) * HL);
      }
    }
    child.computed = true;
  }

  // SCReLU output layer: sum over both halves of clamp(a,0,QA)^2 * w, stm half first.
  // Computed as v*(v*w) so the SIMD versions can use mullo(int16) + madd — v*w fits int16
  // because the trainer clips |w| so that QA*|w_q| < 32768.
  int output_dot(const Accumulator &acc, Color stm) {
    int32_t        sum     = 0;
    const int16_t *half[2] = {acc.v[stm], acc.v[~stm]};
    for (int h = 0; h < 2; ++h) {
      const int16_t *a = half[h];
      const int16_t *w = &g_net.out_w[h * HL];
      for (int i = 0; i < HL; ++i) {
        const int32_t v = std::clamp<int32_t>(a[i], 0, QA);
        sum += v * v * static_cast<int32_t>(w[i]); // exact: |v*v*w| <= 255*255*127 < 2^31
      }
    }
    // sum is at scale QA^2*QB; one /QA plus the bias (at QA*QB) then rescale to centipawns.
    return ((sum / QA) + g_net.out_b) * SCALE / (QA * QB);
  }

} // namespace

// --- Loading -------------------------------------------------------------------------------

bool nnue::load_buffer(const unsigned char *data, size_t size, std::string *err) {
  const auto fail = [err](const char *m) {
    if (err)
      *err = m;
    return false;
  };

  NetHeader h{};
  if (size < sizeof(NetHeader))
    return fail("file smaller than the header");
  std::memcpy(&h, data, sizeof(NetHeader));

  if (std::memcmp(h.magic, "AKNN", 4) != 0)
    return fail("bad magic (not an askaig .nnue file)");
  if (h.version != 1)
    return fail("unsupported version");
  if (h.features != FEATURES || h.hl != HL || h.buckets != 1)
    return fail("architecture mismatch (features/hl/buckets)");
  if (h.qa != QA || h.qb != QB || h.scale != SCALE || h.activation != 1)
    return fail("quantization/activation mismatch");

  constexpr size_t FT_W  = sizeof(g_net.ft_w);
  constexpr size_t FT_B  = sizeof(g_net.ft_b);
  constexpr size_t OUT_W = sizeof(g_net.out_w);
  if (size != sizeof(NetHeader) + FT_W + FT_B + OUT_W + sizeof(int32_t))
    return fail("file size does not match the architecture");

  const unsigned char *p = data + sizeof(NetHeader);
  std::memcpy(g_net.ft_w, p, FT_W);
  p += FT_W;
  std::memcpy(g_net.ft_b, p, FT_B);
  p += FT_B;
  std::memcpy(g_net.out_w, p, OUT_W);
  p += OUT_W;
  std::memcpy(&g_net.out_b, p, sizeof(int32_t));

  g_loaded = true;
  return true;
}

bool nnue::load_file(const std::string &path, std::string *err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (err)
      *err = "cannot open file";
    return false;
  }
  std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return load_buffer(buf.data(), buf.size(), err);
}

bool nnue::loaded() { return g_loaded; }

// --- Evaluator -----------------------------------------------------------------------------

nnue::Evaluator::Evaluator() : stack(new Accumulator[MAX_PLY + 8]), top(0) { stack[0].computed = false; }

void nnue::Evaluator::reset(const Position &pos) {
  top = 0;
  acc_refresh(stack[0], pos);
}

// Records the feature changes of `m` (to be played on `before`) into a new stack entry.
// Mirrors the play<C>() switch exactly — note castling squares are hardcoded per color
// (the Move stores king-to-rook, e1h1), and the captured piece is read from `before`.
void nnue::Evaluator::push(const Position &before, Move m) {
  assert(top + 1 < MAX_PLY + 8);
  Accumulator &a = stack[++top];
  a.computed     = false;
  DirtyPiece &dp = a.dp;
  dp.n_add = dp.n_sub = 0;

  const auto add = [&dp](Piece pc, Square s) {
    dp.add_pc[dp.n_add] = pc;
    dp.add_sq[dp.n_add] = s;
    ++dp.n_add;
  };
  const auto sub = [&dp](Piece pc, Square s) {
    dp.sub_pc[dp.n_sub] = pc;
    dp.sub_sq[dp.n_sub] = s;
    ++dp.n_sub;
  };

  const Color c     = before.turn();
  const Piece mover = before.at(m.from());

  switch (m.flags()) {
    case QUIET:
    case DOUBLE_PUSH:
      sub(mover, m.from());
      add(mover, m.to());
      break;
    case OO:
      if (c == WHITE) {
        sub(WHITE_KING, e1), sub(WHITE_ROOK, h1);
        add(WHITE_KING, g1), add(WHITE_ROOK, f1);
      } else {
        sub(BLACK_KING, e8), sub(BLACK_ROOK, h8);
        add(BLACK_KING, g8), add(BLACK_ROOK, f8);
      }
      break;
    case OOO:
      if (c == WHITE) {
        sub(WHITE_KING, e1), sub(WHITE_ROOK, a1);
        add(WHITE_KING, c1), add(WHITE_ROOK, d1);
      } else {
        sub(BLACK_KING, e8), sub(BLACK_ROOK, a8);
        add(BLACK_KING, c8), add(BLACK_ROOK, d8);
      }
      break;
    case EN_PASSANT:
      sub(mover, m.from());
      sub(make_piece(~c, PAWN), m.to() + (c == WHITE ? SOUTH : NORTH));
      add(mover, m.to());
      break;
    case PR_KNIGHT:
    case PR_BISHOP:
    case PR_ROOK:
    case PR_QUEEN:
      sub(mover, m.from());
      add(make_piece(c, PieceType(KNIGHT + (m.flags() & 0b11))), m.to());
      break;
    case PC_KNIGHT:
    case PC_BISHOP:
    case PC_ROOK:
    case PC_QUEEN:
      sub(mover, m.from());
      sub(before.at(m.to()), m.to());
      add(make_piece(c, PieceType(KNIGHT + (m.flags() & 0b11))), m.to());
      break;
    case CAPTURE:
      sub(mover, m.from());
      sub(before.at(m.to()), m.to());
      add(mover, m.to());
      break;
    default:
      break;
  }
}

void nnue::Evaluator::push_null() {
  assert(top + 1 < MAX_PLY + 8);
  Accumulator &a = stack[++top];
  a.computed     = false;
  a.dp.n_add = a.dp.n_sub = 0; // no feature changes; applied as a copy
}

void nnue::Evaluator::pop() {
  assert(top > 0);
  --top;
}

int nnue::Evaluator::evaluate(const Position &pos) {
  assert(g_loaded);
  if (!stack[top].computed) {
    // Walk back to the nearest computed ancestor and apply the recorded updates forward.
    // If none is close enough (or reset() was never called), a full refresh of the top from
    // `pos` is cheaper than a long chain of applies (~32 row-adds vs 2-4 per ply).
    constexpr int MAX_BACKTRACK = 24;
    int           j             = top;
    while (j > 0 && !stack[j].computed && top - j < MAX_BACKTRACK)
      --j;
    if (stack[j].computed) {
      for (int k = j + 1; k <= top; ++k)
        acc_apply(stack[k], stack[k - 1]);
    } else {
      acc_refresh(stack[top], pos);
    }
  }
  return output_dot(stack[top], pos.turn());
}

int nnue::evaluate_refresh(const Position &pos) {
  assert(g_loaded);
  Accumulator acc;
  acc_refresh(acc, pos);
  return output_dot(acc, pos.turn());
}
