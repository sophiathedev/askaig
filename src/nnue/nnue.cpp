#include "nnue.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#include "accumulator.h"
#include "features.h"
#include "network.h"
#include "position.h"
#include "types.h"

namespace nnue {

  // The single loaded network. Backed by static storage (≈1.6 MiB) so it is never on the stack; g_net
  // points at it once init()/load() fills it.
  namespace {
    Network g_storage;
  }
  const Network *g_net = nullptr;

  bool active() { return g_net != nullptr; }

  // --- accumulator refresh (from scratch) -------------------------------------------------------
  // Build both perspective accumulators directly from the board: bias + the column of every piece's
  // feature. O(pieces * L1); the incremental path (M2) avoids this on most moves.
  static void refresh(const Position &pos, Accumulator &acc) {
    for (int p = 0; p < NCOLORS; ++p)
      std::memcpy(acc.v[p], g_net->ft_bias, sizeof(g_net->ft_bias));

    const Square king[NCOLORS] = {bsf(pos.bitboard_of(WHITE, KING)), bsf(pos.bitboard_of(BLACK, KING))};
    for (int s = 0; s < NSQUARES; ++s) {
      const Piece pc = pos.at(Square(s));
      if (pc == NO_PIECE)
        continue;
      add_feature(acc, WHITE, feature_index(WHITE, pc, Square(s), king[WHITE]));
      add_feature(acc, BLACK, feature_index(BLACK, pc, Square(s), king[BLACK]));
    }
  }

  // --- forward pass ----------------------------------------------------------------------------
  // SCReLU dot for one perspective: sum over i of clamp(a[i],0,QA)^2 * w[i]. Each product fits int32
  // (x^2 <= 65025, |w| <= 32767 -> < 2^31), and integer addition is associative, so the SIMD branches
  // (int32 products, int64-lane accumulation) return a value IDENTICAL to the portable reference. This
  // is the eval analogue of perft: AVX2 == NEON == portable, bit-for-bit. L1 is a multiple of 8/16.
  [[gnu::hot]] static int64_t screlu_dot(const int16_t *a, const int16_t *w) noexcept {
#if defined(SIMD) && defined(ARCH_AVX2)
    __m256i       acc0 = _mm256_setzero_si256(); // 4 x int64
    __m256i       acc1 = _mm256_setzero_si256();
    const __m128i zero = _mm_setzero_si128();
    const __m128i qa   = _mm_set1_epi16(int16_t(QA));
    for (int i = 0; i < L1; i += 8) {
      __m128i a16       = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i)); // 8 int16
      a16               = _mm_min_epi16(_mm_max_epi16(a16, zero), qa); // clamp [0, QA]
      const __m256i x   = _mm256_cvtepi16_epi32(a16); // 8 int32
      const __m256i xsq = _mm256_mullo_epi32(x, x); // x^2 (<= 65025)
      const __m256i wv  = _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w + i)));
      const __m256i p   = _mm256_mullo_epi32(xsq, wv); // 8 int32 products
      acc0              = _mm256_add_epi64(acc0, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(p)));
      acc1              = _mm256_add_epi64(acc1, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(p, 1)));
    }
    const __m256i s = _mm256_add_epi64(acc0, acc1);
    int64_t       buf[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(buf), s);
    return buf[0] + buf[1] + buf[2] + buf[3];
#elif defined(SIMD) && defined(ARCH_ARM_NEON)
    int64x2_t       acc  = vdupq_n_s64(0);
    const int16x4_t zero = vdup_n_s16(0);
    const int16x4_t qa   = vdup_n_s16(int16_t(QA));
    for (int i = 0; i < L1; i += 4) {
      int16x4_t a16       = vld1_s16(a + i);
      a16                 = vmin_s16(vmax_s16(a16, zero), qa); // clamp [0, QA]
      const int32x4_t x   = vmovl_s16(a16);
      const int32x4_t xsq = vmulq_s32(x, x);
      const int32x4_t wv  = vmovl_s16(vld1_s16(w + i));
      const int32x4_t p   = vmulq_s32(xsq, wv); // 4 int32 products
      acc                 = vaddq_s64(acc, vmovl_s32(vget_low_s32(p)));
      acc                 = vaddq_s64(acc, vmovl_s32(vget_high_s32(p)));
    }
    return vgetq_lane_s64(acc, 0) + vgetq_lane_s64(acc, 1);
#else
    int64_t sum = 0;
    for (int i = 0; i < L1; ++i) {
      int x = a[i];
      x     = x < 0 ? 0 : (x > QA ? QA : x);
      sum += int64_t(x) * x * w[i];
    }
    return sum;
#endif
  }

  static int forward(const Accumulator &acc, Color stm) {
    const int64_t sum = screlu_dot(acc.v[stm], g_net->out_weights) + screlu_dot(acc.v[~stm], g_net->out_weights + L1);
    const int     out = int(sum / QA) + g_net->out_bias;
    return out * SCALE / (QA * QB);
  }

  int evaluate(const Position &pos) {
    // Use the accumulator the make/unmake primitives maintain incrementally (refreshed by Position::set
    // and verified == from-scratch in M2); this is the whole point of Option A — no per-eval refresh.
    return forward(pos.nnue_acc, pos.turn());
  }

  // --- net loading ------------------------------------------------------------------------------
  // v1/M1: install a deterministic PLACEHOLDER net (small magnitudes so the int16 accumulator can't
  // overflow with 32 pieces) so the whole inference path is exercisable before a real net is trained.
  // Replace with loading the embedded trained net at M6.
  void init() {
    uint64_t   s  = 0x0123456789ABCDEFULL;
    const auto nx = [&s]() noexcept {
      s ^= s << 13;
      s ^= s >> 7;
      s ^= s << 17;
      return s;
    };
    for (auto &w: g_storage.ft_weights)
      w = int16_t(int(nx() % 65) - 32);
    for (auto &b: g_storage.ft_bias)
      b = int16_t(int(nx() % 65) - 32);
    for (auto &w: g_storage.out_weights)
      w = int16_t(int(nx() % 129) - 64);
    g_storage.out_bias = 0;
    g_net              = &g_storage;
  }

  bool load(const char *path) {
    std::FILE *f = std::fopen(path, "rb");
    if (f == nullptr)
      return false;
    const size_t n = std::fread(&g_storage, 1, sizeof(g_storage), f);
    std::fclose(f);
    if (n != sizeof(g_storage))
      return false;
    g_net = &g_storage;
    return true;
  }

  bool dump(const char *path) {
    if (g_net == nullptr)
      return false;
    std::FILE *f = std::fopen(path, "wb");
    if (f == nullptr)
      return false;
    const size_t n = std::fwrite(g_net, 1, sizeof(Network), f);
    std::fclose(f);
    return n == sizeof(Network);
  }

  // --- self-test (mirror symmetry) --------------------------------------------------------------
  // A side-to-move-relative eval must satisfy eval(pos) == eval(color-mirrored pos): mirroring swaps
  // which physical colour is "us" but the perspective-relative computation is identical. This holds by
  // construction for ANY weights, so it validates the feature/perspective/mirror code with the
  // placeholder net (M1), and that SIMD == portable once SIMD lands (M3).
  namespace {
    // Vertically mirror a FEN's board (swap colours, reverse rank order) and flip the side to move.
    // Inputs use "- -" castling/ep so the mirror is a pure board+side flip.
    std::string mirror_fen(const std::string &fen) {
      std::istringstream in(fen);
      std::string        board, side, rest;
      in >> board >> side;
      std::getline(in, rest); // castling / ep / clocks (kept verbatim; tests use "- -")

      std::vector<std::string> ranks;
      std::string              cur;
      for (char c: board) {
        if (c == '/') {
          ranks.push_back(cur);
          cur.clear();
        } else {
          cur += c;
        }
      }
      ranks.push_back(cur);

      std::string out;
      for (size_t i = ranks.size(); i-- > 0;) {
        for (char c: ranks[i])
          out += char(islower(c) ? toupper(c) : (isupper(c) ? tolower(c) : c));
        if (i != 0)
          out += '/';
      }
      return out + " " + (side == "w" ? "b" : "w") + rest;
    }
  } // namespace

  int self_test() {
    static const char *FENS[] = {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - -", // startpos
            "r3k2r/8/8/8/8/8/8/R3K2R w - -", // rooks + kings
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", // perft pos3
            "8/8/4kpp1/3p1b2/p6P/2B5/6P1/6K1 b - -", // bishop endgame
            "2rq1rk1/pp1bppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/2KR1B1R w - -", // a middlegame
            "8/5k2/8/8/3Q4/8/5K2/8 w - -", // K+Q vs K (asymmetric)
    };
    int fails = 0;
    for (const char *fen: FENS) {
      Position a, b;
      Position::set(fen, a);
      Position::set(mirror_fen(fen), b);
      const int ea = evaluate(a);
      const int eb = evaluate(b);
      if (ea != eb) {
        std::printf("MIRROR FAIL  %6d vs %6d   %s\n", ea, eb, fen);
        ++fails;
      } else {
        std::printf("ok           %6d           %s\n", ea, fen);
      }
    }
    std::printf(fails ? "\nNNUE self-test: %d FAILURE(S)\n" : "\nNNUE self-test: all mirror-symmetric\n", fails);
    return fails;
  }

  // --- incremental == from-scratch (M2) ---------------------------------------------------------
  // Recursively make/unmake every legal move; at each node assert the Position's incrementally-
  // maintained accumulator equals a fresh from-scratch refresh. This proves the put/remove/move hooks
  // (incl. king-refresh, castling, promotions, ep, captures) and undo-reversal are all correct.
  template<Color C>
  static void verify_walk(Position &pos, int depth, int &bad) {
    Accumulator scratch;
    refresh(pos, scratch);
    if (std::memcmp(scratch.v, pos.nnue_acc.v, sizeof(scratch.v)) != 0)
      ++bad;
    if (depth == 0)
      return;
    Move        list[256];
    Move *const end = pos.generate_legals<C>(list);
    for (Move *m = list; m != end; ++m) {
      pos.play<C>(*m);
      verify_walk<~C>(pos, depth - 1, bad);
      pos.undo<C>(*m);
    }
  }

  int verify_incremental() {
    struct Case {
      const char *fen;
      int         depth;
    };
    static const Case cases[] = {
            {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -", 4}, // quiets / double push / captures
            {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", 3}, // castling + captures
            {"n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - -", 4}, // promotions + promotion-captures
            {"rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6", 3}, // en passant available
    };
    int total = 0;
    for (const Case &c: cases) {
      Position p;
      Position::set(c.fen, p);
      int bad = 0;
      if (p.turn() == WHITE)
        verify_walk<WHITE>(p, c.depth, bad);
      else
        verify_walk<BLACK>(p, c.depth, bad);
      std::printf("%-58s depth %d : %s\n", c.fen, c.depth, bad ? "MISMATCH" : "ok");
      total += bad;
    }
    std::printf(total ? "\nNNUE incremental: %d MISMATCH(ES)\n" : "\nNNUE incremental == from-scratch (all nodes)\n",
                total);
    return total;
  }

} // namespace nnue
