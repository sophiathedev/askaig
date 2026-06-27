#include "search.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>
#include <vector>
#include "eval.h"
#include "position.h"
#include "psqt.h"
#include "tt.h"
#include "types.h"

namespace search {

  namespace {

    constexpr int MAX_MOVES     = 218; // upper bound on legal moves (MoveList buffer)
    constexpr int MAX_MATE_PLY  = 256; // scores beyond MATE - this are treated as mates
    constexpr int TT_MOVE_SCORE = 2'000'000; // ordering score for the transposition-table move

    constexpr int MAX_PLY       = MAX_DEPTH; // killer table depth (== the public search ceiling)
    constexpr int KILLER1_SCORE = 900'000; // first killer (quiet) — ranked just below captures
    constexpr int KILLER2_SCORE = 800'000; // second killer
    constexpr int HISTORY_MAX   = 100'000; // cap on history scores (kept below the killers)

    // Forward-pruning margins for shallow non-PV nodes (futility-pruning family). NOTE: the search
    // tunables below are plain `int` (not constexpr) so they can be exposed as UCI spin options for
    // SPSA tuning (see tunables()); a normal run uses the defaults shown.
    int RFP_MAX_DEPTH           = 5; // reverse futility (static null move) up to this depth
    int RFP_MARGIN              = 25; // per-depth eval surplus over beta needed to prune
    int FUTILITY_MAX_DEPTH      = 2; // move-loop futility up to this depth
    int FUTILITY_MARGIN         = 116; // per-depth eval deficit below alpha that prunes quiets
    int LMP_MAX_DEPTH           = 5; // late-move (move-count) pruning up to this depth
    int RAZOR_MAX_DEPTH         = 4; // razoring up to this depth
    int RAZOR_MARGIN            = 248; // per-depth eval deficit below alpha that triggers a qsearch verify
    int HISTORY_PRUNE_MAX_DEPTH = 3; // history pruning up to this depth
    int HISTORY_PRUNE_MARGIN    = 5356; // per-depth quiet-history floor below which a quiet is skipped
    int CUTNODE_R               = 0; // extra LMR reduction at expected-fail-high (cut) nodes; tunable, SPSA-explore
                       // (default 0: at +1 it grew the tree ~15% here, like IIR — let SPSA decide)
    int NMP_EVAL_DIV            = 200; // null-move R gains +1 per this many cp of static eval above beta (cap +3); tunable

    // SEE pruning in the main search (quiescence has its own): at shallow depths, skip moves that
    // lose material to the exchange on their destination square. Captures get a per-depth^2
    // allowance (a deep node may sacrifice for compensation the search can still find); quiets a
    // stricter per-depth one (a quiet that just hangs the piece rarely justifies itself). Margins
    // are in psqt::VALUE scale (pawn = 100) and are unproven starting values (Ethereal-like).
    int SEE_PRUNE_MAX_DEPTH = 12;
    int SEE_QUIET_MARGIN    = 41; // * depth
    int SEE_CAPT_MARGIN     = 18; // * depth^2

    // Singular extensions: only attempted from this depth; the verification search excludes the TT
    // move and uses a window `SINGULAR_MARGIN * depth` below the TT score.
    int SINGULAR_MIN_DEPTH = 7;
    int SINGULAR_MARGIN    = 3;
    // Double extension: if the singular verification falls this far BELOW the singular window, the TT
    // move is not just singular but far better than every alternative -> extend it 2 plies, not 1.
    int DOUBLE_EXT_MARGIN = 50; // cp below sBeta (non-PV only); tunable

    // ProbCut: at depth >= PROBCUT_MIN_DEPTH, a capture whose (depth - PROBCUT_REDUCTION) search
    // already beats beta + PROBCUT_MARGIN cuts the node. The margin is in pawn=100 scale and an
    // unproven starting value (engines with this scale commonly use 100..200).
    int PROBCUT_MIN_DEPTH = 5;
    int PROBCUT_REDUCTION = 3;
    int PROBCUT_MARGIN    = 114;

    int DELTA_MARGIN = 212; // qsearch delta pruning: safety margin above the captured value

    // Log-based LMR reduction table, indexed by [depth][move index]: late moves at deep nodes are
    // reduced progressively more (log*log growth) instead of the old flat 1-3 steps. Filled once by
    // a static initializer (runs before main(), hence before any search thread spawns) and read-only
    // afterwards, so it is shared across Lazy-SMP threads without locking.
    int        LMR_TABLE[64][64];
    const bool LMR_TABLE_INIT = [] {
      for (int d = 1; d < 64; ++d)
        for (int i = 1; i < 64; ++i)
          LMR_TABLE[d][i] = static_cast<int>(1.00 + std::log(d) * std::log(i) / 1.50);
      return true;
    }();

    // Global stop flag, set by request_stop() (the UCI "stop" command) and cleared at the start of
    // each think(). Every search thread (main + helpers) points its t_stop at this.
    std::atomic<bool> g_stop{false};

    // Time control. The deadlines may be armed either at think() start (a normal timed search) or
    // mid-search by request_ponderhit() (a ponder search whose predicted move just appeared), so the
    // active flags are atomic. Each deadline time_point is written *before* its flag is set true and
    // is never modified again, so once a thread observes the flag (acquire) the time_point is stable
    // — no torn read. `g_ponder` is set while searching on the opponent's time (no limits enforced,
    // and think() must not return a move until ponderhit or stop).
    std::atomic<bool>                     g_timed{false}; // a hard deadline is active
    std::chrono::steady_clock::time_point g_deadline; // hard cut-off (abort the search)
    std::atomic<bool>                     g_soft{false}; // a soft budget is active (clock-based search)
    std::chrono::steady_clock::time_point g_soft_start; // when the soft budget started counting
    int64_t                               g_soft_ms = 0; // base soft budget (ms), scaled per iteration
    std::atomic<bool>                     g_ponder{false}; // searching on the opponent's time
    std::atomic<uint64_t>                 g_max_nodes{0}; // per-thread node cap (0 = unlimited)
    constexpr int                         TIME_CHECK_INTERVAL = 2048; // poll the clock every N nodes

    // Arms the time control relative to `from`: a hard deadline `from + hard_ms` (the safety cut-off,
    // enforced in negamax) and a soft budget `g_soft_ms` counted from `from` (the main thread scales
    // it by best-move stability and stops between iterations). Either is skipped when its budget <= 0.
    // Used both at search start and on ponderhit.
    void arm_time(std::chrono::steady_clock::time_point from, int64_t soft_ms, int64_t hard_ms) noexcept {
      if (hard_ms > 0) {
        g_deadline = from + std::chrono::milliseconds(hard_ms);
        g_timed.store(true, std::memory_order_release);
      } else
        g_timed.store(false, std::memory_order_relaxed);
      if (soft_ms > 0) {
        g_soft_start = from;
        g_soft_ms    = soft_ms;
        g_soft.store(true, std::memory_order_release);
      } else
        g_soft.store(false, std::memory_order_relaxed);
    }

    // True once the hard time budget has elapsed. Cheap when untimed (short-circuits before now()).
    [[gnu::hot]] inline bool time_up() noexcept {
      return g_timed.load(std::memory_order_acquire) && std::chrono::steady_clock::now() >= g_deadline;
    }

    // Per-thread abort flag, pointing at g_stop while a search runs so negamax/quiescence bail out
    // promptly. Left null outside a search.
    thread_local const std::atomic<bool> *t_stop = nullptr;

    bool aborted() noexcept { return t_stop != nullptr && t_stop->load(std::memory_order_relaxed); }

    // Per-thread move-ordering heuristics (each Lazy-SMP thread has its own slot, so no locking):
    //  - killers: quiet moves that caused a beta cut-off at a given ply in a sibling node
    //  - history: how often a quiet (color, from, to) move has caused a cut-off
    //  - cont:    continuation history — quiet-move history conditioned on a PREVIOUS move of the
    //             current line, one table per ply-back distance in CONT_OFFSET = {1, 2, 4, 6}:
    //             cont[0] keyed by the move 1 ply back (countermove history: "after they played
    //             that, this reply cuts"), cont[1] by our own move 2 plies back (follow-up), and
    //             cont[2]/cont[3] by the moves 4/6 plies back (longer-range plans). Each prior move
    //             is flattened to a (piece, to-square) index — piece identity matters more than the
    //             from-square (the standard formulation), and the piece encodes its color, so colors
    //             never collide. int16_t (own cap CONT_MAX) halves the cache footprint — ~7 MiB/thread.
    //
    // The slots are PERSISTENT across searches and live in g_heur, NOT in thread_locals: uci.cpp
    // spawns a fresh thread per "go", so a thread_local table would start empty every move — at a
    // fast time control a single ~100ms search is far too short to accumulate statistics (the
    // sparse continuation tables especially would be read almost entirely as zeros while their
    // cache cost is paid on every quiet scored). Instead the tables carry over from move to move
    // within a game — the gravity update self-ages stale entries — and are cleared only by
    // new_game() ("ucinewgame"). Slots are allocated in think() BEFORE any helper spawns (the
    // vector never grows while threads run); each search thread then points t_heur at its slot.
    constexpr int CONT_PLIES              = 4; // number of continuation-history tables (one per offset below)
    constexpr int CONT_OFFSET[CONT_PLIES] = {1, 2, 4, 6}; // plies-back each cont table conditions on
    constexpr int NPIECE_TO               = static_cast<int>(NPIECES) * static_cast<int>(NSQUARES);
    constexpr int CONT_MAX                = 30'000; // gravity cap for the int16_t continuation entries

    // Correction history: a per-(side, structure) running estimate of how far the static eval has
    // historically sat from the value the search actually returned, nudging the raw eval toward the
    // truth — the static eval is biased in structure-dependent ways (it under/over-values a fixed
    // pawn chain, a piece configuration) that a learned offset captures. Two independent tables,
    // summed: one keyed by the pawn skeleton (the two pawn bitboards), one by the non-pawn material
    // (the knight/bishop/rook/queen bitboards). Positions sharing that structure share an entry.
    constexpr int CORR_SIZE  = 16384; // buckets per side (power of two)
    constexpr int CORR_MASK  = CORR_SIZE - 1;
    int           CORR_LIMIT = 1284; // gravity cap on a stored entry (tunable, see tunables())
    int           CORR_GRAIN = 17; // entry / CORR_GRAIN = the cp correction applied (per table; max ±64 cp); tunable

    struct Heuristics {
      Move    killers[MAX_PLY][2];
      int     history[NCOLORS][NSQUARES][NSQUARES];
      int16_t cont[CONT_PLIES][NPIECE_TO][NPIECE_TO];
      // Capture history: how often (attacker piece, to-square, victim type) captures cut off,
      // refining the ordering WITHIN an MVV victim tier (LVA alone knows nothing about position).
      // Victims indexed P..Q (a king is never captured); same gravity/cap as `cont` (~10 KiB).
      int16_t capt[NPIECES][NSQUARES][5];
      // Capture CONTINUATION history (counter-capture): how often a capture flattened to (attacker,
      // to) cuts off given the move ONE ply back — the capture analogue of cont[0]. Keyed
      // [prev (piece,to)][this capture's (piece,to)]; same gravity/cap as `cont` (~1.8 MiB/thread).
      int16_t capt_cont[NPIECE_TO][NPIECE_TO];
      // Correction history (see above): four material-structure tables [side][structure-hash bucket]
      // (pawn skeleton / all non-pawn / minor pieces N+B / major pieces R+Q) plus a continuation
      // correction keyed by the move one ply back ([prev (piece,to) index] — the piece encodes the
      // side, so it is not indexed by color). All summed in corrected_eval, all taught by corr_update.
      int corr[NCOLORS][CORR_SIZE]; // pawn structure
      int corr_np[NCOLORS][CORR_SIZE]; // all non-pawn material
      int corr_minor[NCOLORS][CORR_SIZE]; // minor pieces (knights + bishops)
      int corr_major[NCOLORS][CORR_SIZE]; // major pieces (rooks + queens)
      int corr_cont[NPIECE_TO]; // continuation: keyed by the previous move's (piece, to)
    };
    std::vector<std::unique_ptr<Heuristics>> g_heur; // slot per thread index, persists across searches
    thread_local Heuristics                 *t_heur = nullptr; // this thread's slot, set per search

    thread_local int t_move_stack[MAX_PLY + 1]; // (piece,to) played at each ply of the current line
    thread_local int t_static_eval[MAX_PLY + 1]; // corrected static eval per ply (INF in check), for "improving"

    // Flattened (piece, to-square) index of a move; must be taken BEFORE the move is played (the
    // piece is read from its from-square).
    [[gnu::always_inline]] inline int piece_to_index(Piece pc, Square to) noexcept {
      return static_cast<int>(pc) * static_cast<int>(NSQUARES) + static_cast<int>(to);
    }

    // Combined quiet-move statistic: butterfly history plus the continuation histories against the
    // moves CONT_OFFSET = {1, 2, 4, 6} plies back. `pt_idx` is piece_to_index() of the move scored.
    [[gnu::hot]] inline int quiet_stat(Color c, Move m, int pt_idx, int ply) noexcept {
      int s = t_heur->history[c][m.from()][m.to()];
      for (int k = 0; k < CONT_PLIES; ++k) {
        const int o = CONT_OFFSET[k];
        if (ply >= o && t_move_stack[ply - o] >= 0)
          s += t_heur->cont[k][t_move_stack[ply - o]][pt_idx];
      }
      return s;
    }

    // Triangular principal-variation table: t_pv[ply] holds the best line found from `ply` down,
    // and t_pv_len[ply] its length. A new best move at `ply` prepends itself to the child's line.
    thread_local Move t_pv[MAX_PLY + 1][MAX_PLY + 1];
    thread_local int  t_pv_len[MAX_PLY + 1];

    // Selective depth: the deepest ply any line reached this iteration (including quiescence). Reset
    // once per iteration in aspiration_search and reported on each "info" line as "seldepth".
    thread_local int t_seldepth = 0;

    // Per-thread node countdown to the next clock poll. A plain `nodes & MASK == 0` test is unsafe:
    // quiescence makes the node counter jump in big strides, so a negamax entry rarely lands exactly
    // on a multiple and the clock can go unread for a very long time (a search that ignores its time
    // limit). Counting down one-per-node fires every TIME_CHECK_INTERVAL nodes regardless of strides.
    // Checked in BOTH negamax and quiescence so a big capture subtree is still interruptible.
    thread_local int t_time_count = 0;

    // Approximate per-thread node count since this search began (advanced in TIME_CHECK_INTERVAL steps
    // when the countdown fires), for the optional node cap (`g_max_nodes`). Reset to 0 when a thread
    // starts a search.
    thread_local uint64_t t_nodes = 0;

    // Decrement the countdown; every TIME_CHECK_INTERVAL nodes, read the clock and (if a node cap is
    // armed) the node count, setting the global stop if the hard deadline passed or the cap is reached.
    // Returns true if the search should abort now (stop / time up / node cap). Cheap when untimed and
    // uncapped (the inner block runs once per TIME_CHECK_INTERVAL nodes).
    [[gnu::hot, gnu::always_inline]] inline bool stop_or_time_up() noexcept {
      if (--t_time_count <= 0) [[unlikely]] { // the clock poll fires once per TIME_CHECK_INTERVAL nodes
        t_time_count = TIME_CHECK_INTERVAL;
        t_nodes += TIME_CHECK_INTERVAL;
        const uint64_t cap = g_max_nodes.load(std::memory_order_relaxed);
        if (time_up() || (cap != 0 && t_nodes >= cap))
          g_stop.store(true, std::memory_order_relaxed);
      }
      return aborted();
    }

    // Records `m` followed by the child's PV as the PV at `ply`.
    [[gnu::always_inline]] inline void update_pv(int ply, Move m) noexcept {
      t_pv[ply][0] = m;
      std::memcpy(static_cast<void *>(&t_pv[ply][1]), static_cast<const void *>(&t_pv[ply + 1][0]),
                  static_cast<size_t>(t_pv_len[ply + 1]) * sizeof(Move));
      t_pv_len[ply] = t_pv_len[ply + 1] + 1;
    }

    // A quiet move is a non-capture, non-promotion move (only these use killer/history ordering).
    [[gnu::const]] bool is_quiet(Move m) noexcept {
      if (m.is_capture())
        return false;
      MoveFlags f = m.flags();
      return !(f >= PR_KNIGHT && f <= PR_QUEEN);
    }

    // History "gravity" update: pull the score toward +/-HISTORY_MAX in proportion to how far it
    // still is from that bound. Self-limiting (|h| can never exceed HISTORY_MAX, no manual cap) and
    // self-aging: a big stale score moves quickly once the evidence turns against it.
    [[gnu::always_inline]] inline void history_update(int &h, int bonus) noexcept {
      h += bonus - h * (bonus < 0 ? -bonus : bonus) / HISTORY_MAX;
    }
    // Same gravity formula for the int16_t continuation entries (own cap CONT_MAX; the arithmetic
    // promotes to int, and the result is bounded by ±CONT_MAX so it always fits back).
    [[gnu::always_inline]] inline void cont_update(int16_t &h, int bonus) noexcept {
      h = static_cast<int16_t>(h + bonus - h * (bonus < 0 ? -bonus : bonus) / CONT_MAX);
    }

    // --- Correction history -----------------------------------------------------------------
    // Bucket index for the current pawn structure (both colors' pawns mixed into a hash).
    [[gnu::pure, gnu::always_inline]] inline int corr_index(const Position &p) noexcept {
      const uint64_t wp = p.bitboard_of(WHITE, PAWN);
      const uint64_t bp = p.bitboard_of(BLACK, PAWN);
      const uint64_t k  = wp * 0x9E3779B97F4A7C15ULL ^ bp * 0xC2B2AE3D27D4EB4FULL;
      return static_cast<int>((k >> 47) & CORR_MASK);
    }

    // Bucket index for the non-pawn material configuration (the eight knight/bishop/rook/queen
    // bitboards mixed into a hash; the king is omitted — its bias is largely the pawn-shield term,
    // already captured by the pawn table).
    [[gnu::pure, gnu::always_inline]] inline int corr_np_index(const Position &p) noexcept {
      uint64_t k =
              p.bitboard_of(WHITE, KNIGHT) * 0xFF51AFD7ED558CCDULL ^
              p.bitboard_of(BLACK, KNIGHT) * 0xC4CEB9FE1A85EC53ULL ^
              p.bitboard_of(WHITE, BISHOP) * 0x9E3779B97F4A7C15ULL ^
              p.bitboard_of(BLACK, BISHOP) * 0xC2B2AE3D27D4EB4FULL ^
              p.bitboard_of(WHITE, ROOK) * 0x165667B19E3779F9ULL ^ p.bitboard_of(BLACK, ROOK) * 0x27D4EB2F165667C5ULL ^
              p.bitboard_of(WHITE, QUEEN) * 0x2545F4914F6CDD1DULL ^ p.bitboard_of(BLACK, QUEEN) * 0xD6E8FEB86659FD93ULL;
      return static_cast<int>((k >> 47) & CORR_MASK);
    }

    // Bucket index for the minor pieces (knights + bishops, both colors).
    [[gnu::pure, gnu::always_inline]] inline int corr_minor_index(const Position &p) noexcept {
      const uint64_t k = p.bitboard_of(WHITE, KNIGHT) * 0xFF51AFD7ED558CCDULL ^
                         p.bitboard_of(BLACK, KNIGHT) * 0xC4CEB9FE1A85EC53ULL ^
                         p.bitboard_of(WHITE, BISHOP) * 0x9E3779B97F4A7C15ULL ^
                         p.bitboard_of(BLACK, BISHOP) * 0xC2B2AE3D27D4EB4FULL;
      return static_cast<int>((k >> 47) & CORR_MASK);
    }

    // Bucket index for the major pieces (rooks + queens, both colors).
    [[gnu::pure, gnu::always_inline]] inline int corr_major_index(const Position &p) noexcept {
      const uint64_t k =
              p.bitboard_of(WHITE, ROOK) * 0x165667B19E3779F9ULL ^ p.bitboard_of(BLACK, ROOK) * 0x27D4EB2F165667C5ULL ^
              p.bitboard_of(WHITE, QUEEN) * 0x2545F4914F6CDD1DULL ^ p.bitboard_of(BLACK, QUEEN) * 0xD6E8FEB86659FD93ULL;
      return static_cast<int>((k >> 47) & CORR_MASK);
    }

    // The raw eval nudged by this side's learned corrections: four material-structure tables (pawn /
    // non-pawn / minor / major) plus the continuation correction keyed by the move `prev_pt` one ply
    // back (skipped when prev_pt < 0). Summed (each entry / CORR_GRAIN) and clamped clear of the mate
    // range so it can never masquerade as a forced mate.
    [[gnu::hot, gnu::always_inline]] inline int corrected_eval(const Position &p, Color c, int raw,
                                                               int prev_pt) noexcept {
      int corr = t_heur->corr[c][corr_index(p)] + t_heur->corr_np[c][corr_np_index(p)] +
                 t_heur->corr_minor[c][corr_minor_index(p)] + t_heur->corr_major[c][corr_major_index(p)];
      if (prev_pt >= 0)
        corr += t_heur->corr_cont[prev_pt];
      return std::clamp(raw + corr / CORR_GRAIN, -(MATE - MAX_MATE_PLY) + 1, MATE - MAX_MATE_PLY - 1);
    }

    // Pull all of this side's correction tables toward the (search - static) error, with the same
    // self-capping/self-aging gravity formula the other history tables use.
    [[gnu::always_inline]] inline void corr_update(const Position &p, Color c, int depth, int diff,
                                                   int prev_pt) noexcept {
      const int  bonus = std::clamp(diff * depth / 8, -CORR_LIMIT / 4, CORR_LIMIT / 4);
      const int  absb  = bonus < 0 ? -bonus : bonus;
      const auto pull  = [&](int &e) noexcept { e += bonus - e * absb / CORR_LIMIT; };
      pull(t_heur->corr[c][corr_index(p)]);
      pull(t_heur->corr_np[c][corr_np_index(p)]);
      pull(t_heur->corr_minor[c][corr_minor_index(p)]);
      pull(t_heur->corr_major[c][corr_major_index(p)]);
      if (prev_pt >= 0)
        pull(t_heur->corr_cont[prev_pt]);
    }

    // The victim type of a capture, read on the unmade position (en passant -> a pawn).
    [[gnu::pure]] inline PieceType victim_type_of(const Position &p, Move m) noexcept {
      const Piece v = p.at(m.to());
      return v == NO_PIECE ? PAWN : type_of(v);
    }

    // On a beta cut-off: reward the cut-off move's history by ~depth^2 and penalise (malus) the
    // moves ALREADY TRIED at this node that failed to cut (they were refuted where this move
    // wasn't, so they should sort later next time). Only moves actually searched are punished;
    // moves skipped by LMP/futility/SEE were never refuted. A quiet cut-off move also becomes a
    // killer and feeds the butterfly + continuation histories (against the moves 1 and 2 plies
    // back); the capture history is fed for the tried captures either way (all moves are unmade
    // here, so each move's piece is back on its from-square).
    void update_heuristics(const Position &p, Color c, int ply, Move m, int depth, const Move *quiets, int nq,
                           const Move *capts, int nc) noexcept {
      const int bonus = depth * depth;
      // Move one ply back — the counter-capture continuation key (-1 = none / right after a null move).
      const int prev_pt = ply >= 1 ? t_move_stack[ply - 1] : -1;

      // Capture history: bonus for a capture that cut off, malus for every capture searched before the
      // cut-off move (whatever kind that move was). The same bonus/malus also feeds the counter-capture
      // continuation table, conditioned on the move one ply back.
      if (m.is_capture()) {
        cont_update(t_heur->capt[p.at(m.from())][m.to()][victim_type_of(p, m)], bonus);
        if (prev_pt >= 0)
          cont_update(t_heur->capt_cont[prev_pt][piece_to_index(p.at(m.from()), m.to())], bonus);
      }
      for (int j = 0; j < nc; ++j)
        if (!(capts[j] == m)) {
          cont_update(t_heur->capt[p.at(capts[j].from())][capts[j].to()][victim_type_of(p, capts[j])], -bonus);
          if (prev_pt >= 0)
            cont_update(t_heur->capt_cont[prev_pt][piece_to_index(p.at(capts[j].from()), capts[j].to())], -bonus);
        }

      if (!is_quiet(m))
        return;
      if (ply < MAX_PLY && !(m == t_heur->killers[ply][0])) {
        t_heur->killers[ply][1] = t_heur->killers[ply][0];
        t_heur->killers[ply][0] = m;
      }
      int prev[CONT_PLIES];
      for (int k = 0; k < CONT_PLIES; ++k)
        prev[k] = ply >= CONT_OFFSET[k] ? t_move_stack[ply - CONT_OFFSET[k]] : -1;
      const auto bump = [&](Move q, int b) noexcept {
        history_update(t_heur->history[c][q.from()][q.to()], b);
        const int idx = piece_to_index(p.at(q.from()), q.to());
        for (int k = 0; k < CONT_PLIES; ++k)
          if (prev[k] >= 0)
            cont_update(t_heur->cont[k][prev[k]][idx], b);
      };
      bump(m, bonus);
      for (int j = 0; j < nq; ++j)
        if (!(quiets[j] == m))
          bump(quiets[j], -bonus);
    }

    // Mate scores are stored in the TT relative to the node (not the root): a forced mate is
    // worth more the closer it is, so the distance from the current node must be folded in on the
    // way in and out.
    [[gnu::const]] int score_to_tt(int s, int ply) noexcept {
      if (s > MATE - MAX_MATE_PLY)
        return s + ply;
      if (s < -(MATE - MAX_MATE_PLY))
        return s - ply;
      return s;
    }
    [[gnu::const]] int score_from_tt(int s, int ply) noexcept {
      if (s > MATE - MAX_MATE_PLY)
        return s - ply;
      if (s < -(MATE - MAX_MATE_PLY))
        return s + ply;
      return s;
    }

    // True if `c` has any piece beyond pawns and the king (used to avoid null-move pruning in
    // pawns-only endgames, where passing can be the only good move — zugzwang).
    [[gnu::pure]] bool has_non_pawn_material(const Position &p, Color c) noexcept {
      return p.bitboard_of(c, KNIGHT) || p.bitboard_of(c, BISHOP) || p.bitboard_of(c, ROOK) || p.bitboard_of(c, QUEEN);
    }

    // Move-ordering score (higher = searched earlier):
    //   TT move (added by the caller) > captures/promotions (MVV-LVA) > killers > history quiets.
    [[gnu::hot]] int move_score(const Position &pos, Move m, int ply) noexcept {
      MoveFlags f       = m.flags();
      bool      capture = m.is_capture();
      bool      promo   = (f >= PR_KNIGHT && f <= PR_QUEEN) || (f >= PC_KNIGHT && f <= PC_QUEEN);

      if (capture || promo) {
        int score = 1'000'000;
        if (capture) {
          Piece           victim      = pos.at(m.to()); // NO_PIECE for en passant -> count the victim as a pawn
          const PieceType victim_type = victim == NO_PIECE ? PAWN : type_of(victim);
          score += psqt::VALUE[victim_type] * 16 - psqt::VALUE[type_of(pos.at(m.from()))];
          // Capture history, scaled so it can only reorder WITHIN an MVV victim tier (±CONT_MAX/16
          // = ±1875 < the 3520 gap between adjacent victim values ×16): the victim hierarchy stays
          // absolute, the history breaks LVA ties by what actually cut off here before.
          score += t_heur->capt[pos.at(m.from())][m.to()][victim_type] / 16;
          // Counter-capture continuation: refine further by how this capture fared after the move one
          // ply back. Scaled /32 (±937) so the combined capture history (±1875 + ±937 = ±2812) still
          // stays under the 3520 victim-tier gap — the MVV hierarchy remains absolute.
          if (ply >= 1 && t_move_stack[ply - 1] >= 0)
            score += t_heur->capt_cont[t_move_stack[ply - 1]][piece_to_index(pos.at(m.from()), m.to())] / 32;
        }
        if (promo)
          score += psqt::VALUE[KNIGHT + (static_cast<int>(f) & 0b11)];
        return score;
      }

      // Quiet move: killers for this ply, then the combined history statistic (butterfly +
      // continuation). Its magnitude is bounded by 3*HISTORY_MAX = 300'000, safely below the killers.
      if (ply < MAX_PLY) {
        if (m == t_heur->killers[ply][0])
          return KILLER1_SCORE;
        if (m == t_heur->killers[ply][1])
          return KILLER2_SCORE;
      }
      return quiet_stat(pos.turn(), m, piece_to_index(pos.at(m.from()), m.to()), ply);
    }

    // --- Static exchange evaluation (SEE) ---------------------------------------------------
    // Piece values for the exchange swap (psqt::VALUE, except the king: it must never be counted
    // as winnable material, so it gets a sentinel no exchange sequence can profit from).
    constexpr int SEE_VALUE[NPIECE_TYPES] = {psqt::VALUE[PAWN], psqt::VALUE[KNIGHT], psqt::VALUE[BISHOP],
                                             psqt::VALUE[ROOK], psqt::VALUE[QUEEN],  20000};

    // True if the exchange sequence started by the move `m` wins at least `threshold` cp for the
    // mover, assuming both sides keep capturing on m.to() with their least valuable attacker (the
    // classic swap algorithm). Plain captures AND plain quiet moves (victim value 0 — "is the moved
    // piece simply lost on that square?") are evaluated; everything the swap doesn't model (en
    // passant, promotions — material swings; castling — the king-to-rook encoding) is treated
    // optimistically (never reported as losing). X-ray attackers are revealed as pieces are removed
    // from `occ` (the static `diag`/`orth` masks may re-add an already-used slider, but
    // `attackers &= occ` at the top of each round filters it before use). The king may only
    // "recapture" if the opponent has no attacker left to answer.
    [[gnu::hot]] bool see_ge(const Position &p, Move m, int threshold) noexcept {
      const MoveFlags fl = m.flags();
      if (fl != CAPTURE && fl != QUIET && fl != DOUBLE_PUSH) [[unlikely]]
        return 0 >= threshold;
      const Square from = m.from();
      const Square to   = m.to();

      int swap = (fl == CAPTURE ? SEE_VALUE[type_of(p.at(to))] : 0) - threshold;
      if (swap < 0)
        return false; // taking the victim even for free is below the threshold
      swap = SEE_VALUE[type_of(p.at(from))] - swap;
      if (swap <= 0)
        return true; // losing the moved piece outright still meets the threshold

      const Bitboard diag  = p.bitboard_of(WHITE, BISHOP) | p.bitboard_of(BLACK, BISHOP) | p.bitboard_of(WHITE, QUEEN) |
                             p.bitboard_of(BLACK, QUEEN);
      const Bitboard orth  = p.bitboard_of(WHITE, ROOK) | p.bitboard_of(BLACK, ROOK) | p.bitboard_of(WHITE, QUEEN) |
                             p.bitboard_of(BLACK, QUEEN);
      const Bitboard kings = p.bitboard_of(WHITE, KING) | p.bitboard_of(BLACK, KING);
      const Bitboard side_bb[NCOLORS] = {p.all_pieces<WHITE>(), p.all_pieces<BLACK>()};

      // The mover leaves `from` and stands on `to` (replacing any victim); it is deliberately NOT
      // in `occ` — a piece on the contested square cannot block sliders aimed at that square. The
      // `& ~to` (rather than `^ to`) also clears the square when the move is quiet (`to` empty).
      Bitboard occ = ((side_bb[WHITE] | side_bb[BLACK]) ^ SQUARE_BB[from]) & ~SQUARE_BB[to];
      Color    stm = color_of(p.at(from));
      // attackers_from omits kings (it serves in_check, where a king never gives check) -> OR them.
      Bitboard attackers =
              p.attackers_from<WHITE>(to, occ) | p.attackers_from<BLACK>(to, occ) | (attacks<KING>(to, occ) & kings);

      int res = 1; // 1 = the ORIGINAL mover is winning the exchange so far (flips each round)
      while (true) {
        stm = ~stm;
        attackers &= occ; // drop used attackers (and any stale x-ray re-adds from last round)
        const Bitboard mine = attackers & side_bb[stm];
        if (!mine)
          break; // no attacker left: the side to move loses the race
        res ^= 1;
        Bitboard bb;
        if ((bb = mine & (p.bitboard_of(WHITE, PAWN) | p.bitboard_of(BLACK, PAWN)))) {
          if ((swap = SEE_VALUE[PAWN] - swap) < res)
            break;
          occ ^= SQUARE_BB[bsf(bb)];
          attackers |= attacks<BISHOP>(to, occ) & diag; // a pawn leaving may reveal a diagonal x-ray
        } else if ((bb = mine & (p.bitboard_of(WHITE, KNIGHT) | p.bitboard_of(BLACK, KNIGHT)))) {
          if ((swap = SEE_VALUE[KNIGHT] - swap) < res)
            break;
          occ ^= SQUARE_BB[bsf(bb)]; // knights reveal no x-rays
        } else if ((bb = mine & (p.bitboard_of(WHITE, BISHOP) | p.bitboard_of(BLACK, BISHOP)))) {
          if ((swap = SEE_VALUE[BISHOP] - swap) < res)
            break;
          occ ^= SQUARE_BB[bsf(bb)];
          attackers |= attacks<BISHOP>(to, occ) & diag;
        } else if ((bb = mine & (p.bitboard_of(WHITE, ROOK) | p.bitboard_of(BLACK, ROOK)))) {
          if ((swap = SEE_VALUE[ROOK] - swap) < res)
            break;
          occ ^= SQUARE_BB[bsf(bb)];
          attackers |= attacks<ROOK>(to, occ) & orth;
        } else if ((bb = mine & (p.bitboard_of(WHITE, QUEEN) | p.bitboard_of(BLACK, QUEEN)))) {
          if ((swap = SEE_VALUE[QUEEN] - swap) < res)
            break;
          occ ^= SQUARE_BB[bsf(bb)];
          attackers |= (attacks<BISHOP>(to, occ) & diag) | (attacks<ROOK>(to, occ) & orth);
        } else {
          // King: it may only recapture if the opponent has no attacker waiting behind it.
          return (attackers & ~side_bb[stm]) ? (res ^ 1) != 0 : res != 0;
        }
      }
      return res != 0;
    }

    // Selection-sort step: bring the highest-scoring entry in [i, n) to index i. Done lazily inside
    // the move loop so that, on a cut-off, only the moves actually tried are ever sorted.
    [[gnu::hot]] void pick(Move *moves, int *scores, int i, int n) noexcept {
      int b = i;
      for (int j = i + 1; j < n; ++j)
        if (scores[j] > scores[b])
          b = j;
      if (b != i) {
        std::swap(moves[i], moves[b]);
        std::swap(scores[i], scores[b]);
      }
    }

    // Quiescence search: at the search horizon, keep resolving captures (in MVV-LVA order) until the
    // position is "quiet", so the static eval is not taken mid-capture-sequence (horizon effect).
    // The TT is probed and stored here too (depth-0 entries): this was tried and rejected when the
    // table was single-slot always-replace (qsearch junk evicted deep analysis, +31% nodes), but
    // the bucketed, aged table keeps depth-0 entries from displacing anything valuable — and the
    // stored static eval replaces evaluate(), the dominant qsearch cost, on every revisit.
    template<Color Us>
    [[gnu::hot]] int quiescence(Position &p, int alpha, int beta, uint64_t &nodes, int ply) {
      if (stop_or_time_up()) [[unlikely]]
        return alpha; // stop / time up: the value is discarded (whole iteration thrown away)
      if (p.ply() >= Position::MAX_HISTORY - 2) [[unlikely]]
        return eval::evaluate(p); // absolute undo-stack backstop (see Position::MAX_HISTORY)
      if (ply > t_seldepth)
        t_seldepth = ply; // quiescence reaches beyond the nominal depth -> it sets the selective depth

      const uint64_t key    = p.get_hash();
      tt::Entry     *tte    = tt::probe(key);
      Move           ttMove = Move{};
      int            ttEval = tt::EVAL_NONE;
      if (tte != nullptr) { // fields copied out (lockless table — see the negamax probe)
        const int       ttScore = score_from_tt(tte->score, ply);
        const tt::Bound ttBound = tte->bound();
        ttMove                  = Move(tte->move);
        ttEval                  = tte->eval;
        // Any stored entry is at least as deep as a quiescence node (depth >= 0) -> usable bound.
        if (ttBound == tt::EXACT || (ttBound == tt::LOWER && ttScore >= beta) ||
            (ttBound == tt::UPPER && ttScore <= alpha))
          return ttScore;
      }

      // Generate before the stand-pat: the MoveList refreshes p.checkers, and a node IN CHECK has
      // no stand-pat option (passing is not legal in check) — it must search every evasion, quiet
      // ones included. This is what catches horizon mates: previously an in-check node here stood
      // pat on its static eval and only looked at captures, so a mating attack one ply past the
      // nominal depth could be "evaluated away".
      MoveList<Us> list(p);
      const bool   in_check = p.checkers != 0;
      if (in_check && list.size() == 0) [[unlikely]]
        return -(MATE - ply); // checkmated at the horizon

      // Fail-soft: track and return the true best score even outside (alpha, beta) — the TT then
      // carries tighter bounds than the window edges (negamax/search_root already work this way;
      // quiescence was the fail-hard holdout).
      const int alpha_orig = alpha;
      int       stand_pat  = -INF; // in check: no stand-pat floor, and no eval to cache
      int       best       = -INF; // in check: only the searched evasions set it (all are searched)
      int       raw_eval   = tt::EVAL_NONE; // uncorrected eval for the TT field (EVAL_NONE in check)
      if (!in_check) {
        raw_eval = ttEval != tt::EVAL_NONE ? ttEval : eval::evaluate(p);
        // Correction-adjusted, like the main search, but with no continuation term (prev_pt = -1):
        // quiescence does not maintain t_move_stack below its entry node, so the move-one-ply-back
        // key would be stale deeper in. The four material-structure corrections still apply.
        stand_pat = corrected_eval(p, Us, raw_eval, -1);
        if (stand_pat >= beta) {
          if (!aborted())
            tt::store(key, Move{}, score_to_tt(stand_pat, ply), 0, tt::LOWER, raw_eval);
          return stand_pat;
        }
        best = stand_pat;
        if (stand_pat > alpha)
          alpha = stand_pat;
      }

      // Big-delta pruning (whole-node): if even capturing the most valuable enemy piece — plus the
      // delta margin — cannot raise the stand-pat to alpha, then no capture here can, and the
      // per-move delta below would `continue` past every one and return this same stand-pat. Bail
      // out now to skip generating, scoring and sorting captures that are all hopeless. This is
      // behaviour- AND node-neutral (no capture is ever played either way, so the node count is
      // unchanged) — a pure time saving, not a search change. The guards mirror the per-move delta's
      // exemptions: only out of check, and disabled when a promotion is possible (a pawn on the
      // relative 7th can swing by a queen regardless of the victim, and promotions skip delta).
      if (!in_check && !(p.bitboard_of(Us, PAWN) & MASK_RANK[Us == WHITE ? RANK7 : RANK2])) {
        const Color them = ~Us;
        int         maxGain; // value of the most valuable enemy piece on the board (an upper bound on any victim)
        if (p.bitboard_of(them, QUEEN))
          maxGain = psqt::VALUE[QUEEN];
        else if (p.bitboard_of(them, ROOK))
          maxGain = psqt::VALUE[ROOK];
        else
          maxGain = std::max({psqt::VALUE[PAWN], p.bitboard_of(them, BISHOP) ? psqt::VALUE[BISHOP] : 0,
                              p.bitboard_of(them, KNIGHT) ? psqt::VALUE[KNIGHT] : 0});
        // Only fires when stand_pat <= alpha_orig (else alpha was already raised to stand_pat and the
        // test cannot pass), so the node is a fail-low -> store an UPPER bound, exactly as the loop would.
        if (stand_pat + maxGain + DELTA_MARGIN <= alpha) {
          if (!aborted())
            tt::store(key, Move{}, score_to_tt(stand_pat, ply), 0, tt::UPPER, raw_eval);
          return stand_pat;
        }
      }

      Move moves[MAX_MOVES];
      int  scores[MAX_MOVES];
      int  n = 0;
      for (Move m: list)
        if (in_check || m.is_capture()) { // all evasions when in check, else captures only
          moves[n]  = m;
          scores[n] = m == ttMove ? TT_MOVE_SCORE : move_score(p, m, 0); // ply 0: no real killers, ordering noise only
          ++n;
        }

      for (int i = 0; i < n; ++i) {
        pick(moves, scores, i, n);
        Move m = moves[i];

        // Delta pruning: if winning the captured piece (plus a margin) still cannot reach alpha,
        // this capture is hopeless — skip it. Promotions are exempt (they can swing by a queen),
        // and so are check evasions (every evasion must be considered — there is no stand-pat to
        // fall back on, and "losing" SEE is irrelevant when the alternative may be mate).
        const MoveFlags f     = m.flags();
        const bool      promo = f >= PC_KNIGHT && f <= PC_QUEEN;
        if (!promo && !in_check) {
          const Piece victim = p.at(m.to()); // NO_PIECE for en passant -> the victim is a pawn
          const int   gain   = victim == NO_PIECE ? psqt::VALUE[PAWN] : psqt::VALUE[type_of(victim)];
          if (stand_pat + gain + DELTA_MARGIN <= alpha)
            continue;
          // SEE pruning: a capture that loses material in the full exchange sequence cannot rescue
          // the stand-pat (resolving captures is the only job here — losing ones don't resolve up).
          if (!see_ge(p, m, 0))
            continue;
        }

        p.play<Us>(m);
        ++nodes;
        int score = -quiescence<~Us>(p, -beta, -alpha, nodes, ply + 1);
        p.undo<Us>(m);

        if (score > best)
          best = score;
        if (score >= beta) {
          if (!aborted())
            tt::store(key, m, score_to_tt(best, ply), 0, tt::LOWER, raw_eval);
          return best;
        }
        if (score > alpha)
          alpha = score;
      }

      // No cut-off: the window was entered -> exact; never reached -> an upper bound.
      if (!aborted())
        tt::store(key, Move{}, score_to_tt(best, ply), 0, best > alpha_orig ? tt::EXACT : tt::UPPER, raw_eval);
      return best;
    }

    // Fills the move/score buffers from `list`, giving the transposition-table move top priority.
    [[gnu::hot]] void score_moves(const Position &p, const Move *src, Move *moves, int *scores, int n, Move tt_move,
                                  int ply) noexcept {
      for (int i = 0; i < n; ++i) {
        moves[i]  = src[i];
        scores[i] = moves[i] == tt_move ? TT_MOVE_SCORE : move_score(p, moves[i], ply);
      }
    }

    // Negamax with alpha-beta pruning and a transposition table. `ply` is the distance from the
    // root; it makes a checkmate found nearer the root score higher, so the engine prefers the
    // fastest mate (and, when losing, drags the mate out as long as possible).
    template<Color Us>
    [[gnu::hot]] int negamax(Position &p, int depth, int ply, int alpha, int beta, uint64_t &nodes, bool null_ok,
                             bool cutNode, Move excluded = Move{}) {
      if (stop_or_time_up()) [[unlikely]]
        return 0; // stop requested / hard deadline hit: value is discarded, last completed depth kept

      // Clear this node's PV row BEFORE any other return path. A row left over from a previous
      // sibling line is a line from a DIFFERENT branch: if an early return below (draw, ply cap)
      // scored above the parent's alpha, the parent's update_pv would memcpy that stale row and the
      // reported PV would contain moves that are illegal in this line (seen as fastchess "Illegal
      // PV move" warnings once repetition detection made draw returns frequent in self-play).
      t_pv_len[ply] = 0;

      if (ply > t_seldepth)
        t_seldepth = ply; // track the deepest ply reached, for the reported selective depth
      if (ply >= MAX_PLY || p.ply() >= Position::MAX_HISTORY - MAX_PLY) [[unlikely]]
        return eval::evaluate(p); // hard cap: bound runaway extensions, and never index history[] OOB
                                  // (the search stacks its plies onto game_ply; in a very long game
                                  //  that could otherwise overrun the undo stack — reserve MAX_PLY for
                                  //  the quiescence that may still run below this node)

      // Draw by repetition / fifty-move rule: score it 0 immediately (never at the root, ply 0, where
      // a move must still be returned). This collapses shuffling/perpetual lines instead of searching
      // them to the depth ceiling — it both plays draws correctly (no repeating in won positions) and
      // is what keeps the PV short (an unbounded repetition PV could otherwise flood stdout).
      if (ply > 0 && p.is_draw()) [[unlikely]]
        return 0;

      const uint64_t key        = p.get_hash();
      const int      alpha_orig = alpha;
      const bool     is_pv      = beta - alpha > 1; // a full window => principal-variation node

      // Mate distance pruning: a mate can be no faster than `ply` from the root, so tighten the
      // window to the best/worst still-possible mate scores; if it collapses, nothing here can beat
      // an already-found shorter mate. Exact and free of Elo risk.
      if (alpha < -MATE + ply)
        alpha = -MATE + ply;
      if (beta > MATE - ply - 1)
        beta = MATE - ply - 1;
      if (alpha >= beta)
        return alpha;

      // Transposition-table probe: reuse a deep-enough stored result, and remember its move/score.
      // Every needed field is COPIED out here: the table is lockless, so the slot can be
      // overwritten by descendant/helper-thread stores while this node runs (the singular check
      // reads these deep inside the move loop — re-reading the entry there could mix in fields
      // from a completely different position).
      tt::Entry *tte     = tt::probe(key);
      const bool tt_hit  = tte != nullptr;
      Move       ttMove  = Move{};
      int        ttScore = 0;
      int        ttDepth = 0;
      int        ttEval  = tt::EVAL_NONE;
      tt::Bound  ttBound = tt::NONE;
      if (tt_hit) {
        ttMove  = Move(tte->move);
        ttScore = score_from_tt(tte->score, ply);
        ttDepth = tte->depth;
        ttEval  = tte->eval;
        ttBound = tte->bound();
        // During a singular verification search (excluded set) the stored result is for the full
        // move set, so it must not cut this (reduced) search off.
        if (excluded == Move{} && ttDepth >= depth) {
          if (ttBound == tt::EXACT || (ttBound == tt::LOWER && ttScore >= beta) ||
              (ttBound == tt::UPPER && ttScore <= alpha))
            return ttScore;
        }
      }

      MoveList<Us> list(p);
      const int    n = static_cast<int>(list.size());

      if (n == 0) [[unlikely]]
        return p.checkers ? -(MATE - ply) : 0; // checkmate (lose) : stalemate (draw)
      if (depth == 0)
        return quiescence<Us>(p, alpha, beta, nodes, ply);

      // Captured now because the null-move / child searches below overwrite p.checkers.
      const bool in_check = p.checkers != 0;
      // (piece, to) of the move one ply back — the continuation-correction key (valid here: the
      // parent set it before recursing; this node only writes t_move_stack[ply]). -1 = none.
      const int prev_pt = ply >= 1 ? t_move_stack[ply - 1] : -1;

      // Static eval at every non-check interior node (the TT carries it across revisits, so this is
      // cheap). Two forms: `raw_eval` (uncorrected, stored in the TT eval field) and `static_eval`
      // (nudged by the correction histories — see corrected_eval — and used for every pruning
      // decision and for the post-search correction update). INF means "not computed" (in check).
      int        raw_eval    = INF;
      int        static_eval = INF;
      const bool can_prune   = !is_pv && !in_check && beta < MATE - MAX_MATE_PLY && beta > -(MATE - MAX_MATE_PLY);
      if (!in_check) [[likely]] { // the vast majority of interior nodes are not in check
        // Reuse the static eval stored in the TT when the position was seen before — evaluate() is
        // the dominant cost at these nodes. (The stored eval may have been computed at a different
        // halfmove clock — the hash omits it — so the fifty-move damping can be slightly off; the
        // same staleness is already accepted for stored scores.)
        raw_eval    = ttEval != tt::EVAL_NONE ? ttEval : eval::evaluate(p);
        static_eval = corrected_eval(p, Us, raw_eval, prev_pt);
        // Reverse futility pruning (static null move): a depth-scaled margin above beta -> prune.
        if (can_prune && depth <= RFP_MAX_DEPTH && static_eval - RFP_MARGIN * depth >= beta)
          return static_eval;
        // Razoring: when the static eval is far enough BELOW alpha that even resolving every capture
        // is unlikely to recover, drop straight to a quiescence search; if that also fails to reach
        // alpha, prune (return its fail-low score). The qsearch verification is what keeps this safe
        // against a tactic the static eval missed — we only bail when the captures don't rescue it.
        if (can_prune && depth <= RAZOR_MAX_DEPTH && static_eval + RAZOR_MARGIN * depth <= alpha) {
          const int v = quiescence<Us>(p, alpha, beta, nodes, ply);
          if (v <= alpha)
            return v;
        }
      }

      // Record this node's static eval (INF when in check) on the per-ply stack and derive the
      // "improving" flag: our static eval is higher than it was two plies ago (same side to move),
      // i.e. the position is trending our way. Used ONLY to reduce late quiet moves one ply more
      // when NOT improving (a stagnant/worsening line spends less on its unpromising tail). NOTE: an
      // improving heuristic was previously rejected for RFP-margin / LMP-count tuning (SPRT -23.7);
      // this is a deliberately narrower, distinct application (LMR depth only) and is SPRT-gated.
      t_static_eval[ply] = static_eval;
      const bool improving =
              !in_check && ply >= 2 && t_static_eval[ply - 2] != INF && static_eval > t_static_eval[ply - 2];

      // Null-move pruning: if passing the turn (a free move for the opponent) still leaves us at
      // >= beta after a shallower search, the position is so good we can prune. Skipped when in
      // check (can't pass), shallow, right after a null move, during a singular search, near mate
      // bounds, or in a pawns-only (zugzwang-prone) position.
      bool mate_threat = false;
      if (null_ok && excluded == Move{} && depth >= 3 && !in_check && beta < MATE - MAX_MATE_PLY &&
          has_non_pawn_material(p, Us)) {
        // Null-move reduction: the depth-scaled base, plus a bonus when the static eval already sits
        // comfortably above beta — the more the position looks won without moving, the safer it is to
        // reduce the verification harder (capped at +3). static_eval is valid here (this branch is
        // !in_check, where it was computed above).
        int R = 2 + depth / 6;
        if (static_eval >= beta)
          R += std::min((static_eval - beta) / NMP_EVAL_DIV, 3);
        // Clamp the reduced null-search depth to >= 0: the eval bonus can push R past depth-1 (e.g. at
        // depth 3, base 2 + bonus 3 = 5), and a negative depth would skip the `depth == 0` quiescence
        // dispatch in the child and recurse to the ply cap (a node blow-up — see the LMR clamp note).
        const int null_depth = depth - 1 - R > 0 ? depth - 1 - R : 0;
        p.play_null();
        t_move_stack[ply] = -1; // a pass carries no (piece, to) to condition continuation history on
        int score         = -negamax<~Us>(p, null_depth, ply + 1, -beta, -beta + 1, nodes, false, !cutNode, Move{});
        p.undo_null();
        if (score >= beta) // fail-high: prune this node (fail-soft, except a reduced null-window
          return score >= MATE - MAX_MATE_PLY ? beta : score; // result is never trusted as a MATE bound)
        // Mate-threat extension: if passing lets the opponent force mate, this line is dangerous —
        // search it deeper rather than trusting the shallow reductions.
        if (score <= -(MATE - MAX_MATE_PLY))
          mate_threat = true;
      }

      // ProbCut: if a capture, searched shallow with the window raised by a wide margin, already
      // fails high, the full-depth search would almost certainly fail high too — cut now. Gated
      // like the futility family (can_prune: non-PV, not in check, non-mate), at real depth only,
      // never during singular verification, and skipped when the TT already holds counter-evidence
      // (a deep-enough entry whose score sits BELOW the raised beta). Each candidate must first
      // beat the raised window in a plain qsearch (cheap) before the reduced negamax verification
      // is paid; a confirmed cut is stored as a depth-(R-1) lower bound so revisits cut instantly.
      const int pc_beta = beta + PROBCUT_MARGIN;
      if (can_prune && excluded == Move{} && depth >= PROBCUT_MIN_DEPTH && pc_beta < MATE - MAX_MATE_PLY &&
          !(tt_hit && ttDepth >= depth - (PROBCUT_REDUCTION - 1) && ttScore < pc_beta)) {
        for (Move m: list) {
          if (!m.is_capture())
            continue;
          t_move_stack[ply] = piece_to_index(p.at(m.from()), m.to()); // continuation-history context
          p.play<Us>(m);
          ++nodes;
          int v = -quiescence<~Us>(p, -pc_beta, -pc_beta + 1, nodes, ply + 1);
          if (v >= pc_beta) // qsearch agrees -> pay for the reduced full verification
            v = -negamax<~Us>(p, depth - PROBCUT_REDUCTION, ply + 1, -pc_beta, -pc_beta + 1, nodes, true, !cutNode,
                              Move{});
          p.undo<Us>(m);
          if (aborted()) [[unlikely]]
            return 0;
          if (v >= pc_beta) {
            tt::store(key, m, score_to_tt(v, ply), depth - (PROBCUT_REDUCTION - 1), tt::LOWER,
                      raw_eval == INF ? tt::EVAL_NONE : raw_eval);
            return v;
          }
        }
      }

      // Internal iterative reductions: with no TT move the move ordering has no best-move hint and is
      // unreliable, so a full-depth search is wasteful — reduce by one ply. The shallower search fills
      // the TT move, so the (re-)visit is better ordered. PV and expected-cut nodes of sufficient depth
      // only; never in check or during singular verification (which already has its excluded TT move).
      if (depth >= 4 && ttMove == Move{} && excluded == Move{} && !in_check && (is_pv || cutNode))
        --depth;

      Move moves[MAX_MOVES];
      int  scores[MAX_MOVES];
      score_moves(p, list.begin(), moves, scores, n, ttMove, ply);

      // Node-level extension (one ply): in check (every reply forced), a null-move-detected mate
      // threat (shallow reductions can't be trusted), or a single legal move (one-reply — also
      // forced) — search the whole node deeper.
      const int node_ext = (in_check || mate_threat || n == 1) ? 1 : 0;

      int  best     = -INF;
      Move bestMove = Move{};
      Move quiets_tried[MAX_MOVES]; // quiets actually searched at this node (for the history malus)
      Move capts_tried[MAX_MOVES]; // captures actually searched (for the capture-history malus)
      int  nq = 0;
      int  nc = 0;
      for (int i = 0; i < n; ++i) {
        pick(moves, scores, i, n);
        Move m = moves[i];
        if (m == excluded) [[unlikely]]
          continue; // singular verification search: skip the move being tested

        // Forward pruning of late, quiet moves at shallow non-PV nodes (best already set => at
        // least one move searched, and not pruning into a mate). Moves are tried best-first, so by
        // this point the remaining quiets are the unpromising tail.
        if (can_prune && is_quiet(m) && best > -(MATE - MAX_MATE_PLY)) {
          // Late move pruning: past a depth-scaled move count, skip the rest of the quiet moves.
          if (depth <= LMP_MAX_DEPTH && i >= 3 + depth * depth)
            continue;
          // Futility pruning: if the static eval plus a margin can't reach alpha, skip the quiet.
          if (depth <= FUTILITY_MAX_DEPTH && static_eval + FUTILITY_MARGIN * depth <= alpha)
            continue;
          // History pruning: a quiet whose combined history score is strongly negative has been
          // refuted here repeatedly — at shallow depth skip it. scores[i] IS that history value for
          // a non-killer quiet (killers carry a large positive score, TT/captures larger still, so
          // none of those are ever caught by the negative threshold).
          if (depth <= HISTORY_PRUNE_MAX_DEPTH && scores[i] < -HISTORY_PRUNE_MARGIN * depth)
            continue;
        }
        // SEE pruning: skip a move that loses material on its destination square (same gates as
        // above, but captures included). `best > mated` means at least one move was searched, so
        // the node always returns a real score; the first move (best == -INF) is never pruned, and
        // see_ge is optimistic about what the swap can't model (promotions, ep, castling).
        if (can_prune && best > -(MATE - MAX_MATE_PLY) && depth <= SEE_PRUNE_MAX_DEPTH &&
            !see_ge(p, m, is_quiet(m) ? -SEE_QUIET_MARGIN * depth : -SEE_CAPT_MARGIN * depth * depth))
          continue;

        // Per-move extension (node-level check/mate-threat or a singular TT move; up to two plies for
        // a decisively-singular move — see below). (A recapture extension was tried and removed: it
        // cost ~+40% nodes while quiescence and the check extension already cover exchange resolution.)
        int ext = node_ext;
        if (!ext && depth >= SINGULAR_MIN_DEPTH && m == ttMove && excluded == Move{} && tt_hit &&
            ttDepth >= depth - 3 && ttBound != tt::UPPER && ttScore > -(MATE - MAX_MATE_PLY) &&
            ttScore < MATE - MAX_MATE_PLY) {
          // Singular extension: search every move *except* the TT move at reduced depth with a
          // window just below the TT score. If none reaches it, the TT move is singular -> extend.
          const int sBeta  = ttScore - SINGULAR_MARGIN * depth;
          const int sDepth = (depth - 1) / 2;
          const int v      = negamax<Us>(p, sDepth, ply, sBeta - 1, sBeta, nodes, false, cutNode, m);
          if (v < sBeta) {
            ext = 1;
            // Double extension: the best alternative falls not just short of the singular window but
            // well below it -> the TT move is decisively the only good move, search it two plies
            // deeper. Non-PV only (PV lines are already searched to full depth) and gated by the
            // singular conditions (depth >= 7, a deep TT move) so it stays rare.
            if (!is_pv && v < sBeta - DOUBLE_EXT_MARGIN)
              ext = 2;
          }
          // Multi-cut pruning: the TT move is NOT singular (v >= sBeta, so some OTHER move also reached
          // the singular threshold) and that threshold already sits at/above beta. Two distinct moves
          // therefore fail high here, so this expected cut-node is genuinely a cut-node — prune the
          // whole subtree, returning the (fail-soft) bound. Reuses the singular verification search just
          // done, at no extra cost (the capture-only ProbCut above is a separate, complementary prune).
          // Non-PV only: a PV node is searched to full depth rather than cut on a reduced result.
          else if (!is_pv && sBeta >= beta)
            return sBeta;
        }
        if (!ext && type_of(p.at(m.from())) == PAWN &&
            ((Us == WHITE && rank_of(m.to()) == RANK7) || (Us == BLACK && rank_of(m.to()) == RANK2)) &&
            eval::is_passed_pawn(p, Us, m.to()))
          ext = 1; // passed pawn pushed to the 7th rank (one from promotion) -> search deeper

        const int nd = depth - 1 + ext;

        // (piece, to) of this move, read BEFORE playing it; pushed onto the line stack so child
        // nodes can condition their continuation-history lookups on it.
        const int pt_idx  = piece_to_index(p.at(m.from()), m.to());
        t_move_stack[ply] = pt_idx;

        p.play<Us>(m);
        tt::prefetch(p.get_hash()); // start the child's TT cache-line fetch before it probes
        ++nodes;

        int score;
        if (i == 0) {
          // Principal variation: search the (well-ordered) first move with the full window.
          score = -negamax<~Us>(p, nd, ply + 1, -beta, -alpha, nodes, true, false, Move{});
        } else {
          // Late move reductions: a late, quiet move is unlikely to beat alpha, so scout it at a
          // reduced depth first; only re-search at full depth if that surprises us (beats alpha).
          // Tactical moves (captures/promotions), shallow nodes, and in-check nodes aren't reduced;
          // PV nodes are reduced one step less. The clamp to [0, nd] is load-bearing: a negative
          // child depth would skip the `depth == 0` quiescence dispatch and recurse to the ply cap.
          int reduction = 0;
          if (depth >= 3 && i >= 3 && !in_check && is_quiet(m)) {
            reduction = LMR_TABLE[depth < 63 ? depth : 63][i < 63 ? i : 63];
            if (is_pv)
              --reduction;
            if (!improving)
              ++reduction; // a stagnant/worsening line reduces its late quiets one ply more
            if (cutNode)
              reduction += CUTNODE_R; // expected-fail-high node whose 1st move didn't cut -> reduce late ones more
            // History-informed adjustment: quiets with a strong (signed) history record are reduced
            // less, persistent fail-lows more. Deliberately the butterfly history ONLY: feeding the
            // combined stat (with a rescaled divisor) in here was SPRT-tested together with the
            // continuation-history ordering and trended ~-30 Elo — early in a search the sparse
            // continuation tables dilute a proven signal. Re-couple only with SPRT evidence.
            reduction -= std::clamp(t_heur->history[Us][m.from()][m.to()] / 8'192, -2, 2);
            reduction = std::clamp(reduction, 0, nd);
          }
          // PVS scout at the (possibly reduced) depth with a null window. A scout child is the
          // opposite node type to its parent (cut <-> all), so it carries !cutNode.
          score = -negamax<~Us>(p, nd - reduction, ply + 1, -alpha - 1, -alpha, nodes, true, !cutNode, Move{});
          // Reduced scout beat alpha: re-search at full depth (still a null window).
          if (reduction > 0 && score > alpha)
            score = -negamax<~Us>(p, nd, ply + 1, -alpha - 1, -alpha, nodes, true, !cutNode, Move{});
          // PVS: inside the window -> re-search with the full window.
          if (score > alpha && score < beta)
            score = -negamax<~Us>(p, nd, ply + 1, -beta, -alpha, nodes, true, false, Move{});
        }
        p.undo<Us>(m);

        if (is_quiet(m))
          quiets_tried[nq++] = m; // searched (not pruned away) -> eligible for the malus
        else if (m.is_capture())
          capts_tried[nc++] = m;

        if (score > best) {
          best     = score;
          bestMove = m;
          if (best > alpha) {
            alpha = best;
            // PV nodes only: null-window searches (PVS scouts, the singular verification — which
            // runs at the SAME ply as its parent and would scribble over the parent's row) don't
            // produce lines anyone reports. Reporting-only, so the search itself is unaffected.
            if (is_pv)
              update_pv(ply, m); // a new best move at this node -> extend the PV
          }
        }
        if (alpha >= beta) {
          // Reward the quiet cut-off move; penalise the quiets tried before it (history malus).
          update_heuristics(p, Us, ply, m, depth, quiets_tried, nq, capts_tried, nc);
          break;
        }
      }

      if (!aborted() && excluded == Move{}) { // don't store a partial (aborted) or singular result
        tt::Bound bound = best <= alpha_orig ? tt::UPPER : (best >= beta ? tt::LOWER : tt::EXACT);
        tt::store(key, bestMove, score_to_tt(best, ply), depth, bound, raw_eval == INF ? tt::EVAL_NONE : raw_eval);
        // Correction history: when not in check and the best move is quiet, and the search bound
        // corroborates the direction of the (search - static) gap, teach every correction table
        // that gap so future static evals of the same structure are less biased. Captures are
        // excluded (their swing is tactical, not a structural eval bias), as are mate-range scores.
        if (!in_check && static_eval != INF && !bestMove.is_capture() && best < MATE - MAX_MATE_PLY &&
            best > -(MATE - MAX_MATE_PLY) &&
            (bound == tt::EXACT || (bound == tt::LOWER && best > static_eval) ||
             (bound == tt::UPPER && best < static_eval)))
          corr_update(p, Us, depth, best - static_eval, prev_pt);
      }
      return best;
    }

    // Root search over the window (alpha, beta). Records which move produced the best score and
    // stores it (with the correct bound, since aspiration windows can fail high/low at the root);
    // it never takes a TT cut-off (it must return a move) but uses the TT move for ordering.
    template<Color Us>
    Result search_root(Position &p, int depth, int alpha, int beta) {
      const int    alpha_orig = alpha;
      Result       r{Move{}, -INF, 0};
      MoveList<Us> list(p);
      const int    n = static_cast<int>(list.size());

      if (n == 0) {
        r.score = p.checkers ? -MATE : 0; // already checkmated (loss) or stalemate (draw)
        return r; // r.best stays null -> caller emits "bestmove 0000"
      }

      const uint64_t key    = p.get_hash();
      tt::Entry     *tte    = tt::probe(key);
      Move           ttMove = tte != nullptr ? Move(tte->move) : Move{};

      Move moves[MAX_MOVES];
      int  scores[MAX_MOVES];
      score_moves(p, list.begin(), moves, scores, n, ttMove, 0);

      t_pv_len[0]         = 0;
      const int new_depth = depth - 1 + (p.checkers || n == 1 ? 1 : 0); // check / one-reply extension

      for (int i = 0; i < n; ++i) {
        pick(moves, scores, i, n);
        if (aborted())
          return r; // helper thread is stopping; result is discarded, don't store
        Move m          = moves[i];
        t_move_stack[0] = piece_to_index(p.at(m.from()), m.to()); // root move: continuation context for ply 1
        p.play<Us>(m);
        tt::prefetch(p.get_hash()); // start the child's TT cache-line fetch before it probes
        ++r.nodes;
        const uint64_t nodes_before = r.nodes; // attribute this move's subtree to the effort count

        int score;
        if (i == 0) {
          score = -negamax<~Us>(p, new_depth, 1, -beta, -alpha, r.nodes, true, false,
                                Move{}); // PV child: not a cut node
        } else {
          score = -negamax<~Us>(p, new_depth, 1, -alpha - 1, -alpha, r.nodes, true, true, Move{}); // scout: cut node
          if (score > alpha && score < beta)
            score = -negamax<~Us>(p, new_depth, 1, -beta, -alpha, r.nodes, true, false, Move{}); // re-search (PV)
        }
        p.undo<Us>(m);

        if (score > r.score) {
          r.score      = score;
          r.best       = m;
          r.best_nodes = r.nodes - nodes_before; // nodes this (new best) move's subtree consumed
          if (score > alpha) {
            alpha = score;
            update_pv(0, m); // a new best root move -> record the PV
          }
        }
        if (alpha >= beta) {
          update_heuristics(p, Us, 0, m, depth, nullptr, 0, nullptr,
                            0); // root fail-high: no malus (aspiration artifact)
          break; // fail-high: the caller re-searches with a wider window
        }
      }

      if (!aborted()) {
        tt::Bound bound = r.score <= alpha_orig ? tt::UPPER : (r.score >= beta ? tt::LOWER : tt::EXACT);
        tt::store(key, r.best, score_to_tt(r.score, 0), depth, bound, tt::EVAL_NONE);
      }

      // Export the collected principal variation (fall back to just the best move if none was set).
      if (t_pv_len[0] > 0)
        r.pv.assign(&t_pv[0][0], &t_pv[0][t_pv_len[0]]);
      else if (!(r.best == Move{}))
        r.pv = {r.best};
      r.seldepth = std::max(depth, t_seldepth); // never report a selective depth below the nominal one
      return r;
    }

    // Runs one root search to `depth` over the window (alpha, beta), dispatching on side to move.
    Result search_to_depth(Position &p, int depth, int alpha, int beta) {
      return p.turn() == WHITE ? search_root<WHITE>(p, depth, alpha, beta) : search_root<BLACK>(p, depth, alpha, beta);
    }

    // Searches one depth with an aspiration window around `prev_score` (the previous iteration's
    // score), re-searching with a progressively wider window on a fail-high/low. Shallow depths and
    // mate scores fall back to a full window. The returned result carries the cumulative node count.
    Result aspiration_search(Position &p, int depth, int prev_score) {
      t_seldepth = 0; // selective depth is measured per iteration; accumulate across this depth's re-searches
      if (depth <= 4 || prev_score > MATE - MAX_MATE_PLY || prev_score < -(MATE - MAX_MATE_PLY))
        return search_to_depth(p, depth, -INF, INF);

      int      delta = 50;
      int      alpha = std::max(prev_score - delta, -INF);
      int      beta  = std::min(prev_score + delta, INF);
      uint64_t nodes = 0;
      while (true) {
        Result r = search_to_depth(p, depth, alpha, beta);
        nodes += r.nodes;
        if (aborted()) { // stop/time-up: the result is incomplete (search_root may return -INF), so
          r.nodes = nodes; // don't widen on it (that would spin forever) — hand it back; the caller discards it
          return r;
        }
        if (r.score <= alpha) // fail low: widen the lower bound
          alpha = std::max(r.score - delta, -INF);
        else if (r.score >= beta) // fail high: widen the upper bound
          beta = std::min(r.score + delta, INF);
        else { // inside the window
          r.nodes = nodes;
          return r;
        }
        delta *= 2; // widen exponentially -> reaches the full window in a few tries
      }
    }

    bool is_mate(int score) { return score > MATE - MAX_MATE_PLY || score < -(MATE - MAX_MATE_PLY); }

    // Returns true for an obviously illegal position that the (legality-assuming) generators must
    // not be run on: a side missing its king, or the side NOT to move being in check. Searching
    // such a position would let the engine "capture the king", after which generate_legals reads
    // its attack tables out of bounds. The `&&` short-circuits so in_check (which does bsf on the
    // king bitboard) is only reached once both kings are known to exist.
    bool is_illegal(const Position &pos) {
      if (!pos.bitboard_of(WHITE, KING) || !pos.bitboard_of(BLACK, KING))
        return true;
      return pos.turn() == WHITE ? pos.in_check<BLACK>() : pos.in_check<WHITE>();
    }

    // Byte-clones a position so each search thread can make/unmake on its own copy. A plain copy
    // would invoke UndoInfo's copy constructor (which resets epsq/captured) and corrupt history.
    Position clone(const Position &src) {
      Position dst;
      // Byte copy (cast to void* to acknowledge the non-trivial copy ctor): every Position member
      // is plain data, so the bit pattern is a faithful, independent copy.
      std::memcpy(static_cast<void *>(&dst), static_cast<const void *>(&src), sizeof(Position));
      return dst;
    }

    // The first legal move (any), used as a safe fallback bestmove if the search is stopped before
    // even depth 1 completes.
    template<Color Us>
    Move first_legal(Position &p) {
      MoveList<Us> list(p);
      return list.size() ? *list.begin() : Move{};
    }
    Move any_legal_move(Position &p) { return p.turn() == WHITE ? first_legal<WHITE>(p) : first_legal<BLACK>(p); }

  } // namespace

  // SPSA-tunable search constants, with [min, max] bounds for the tuner. The names double as UCI
  // option names. Defined here (in namespace search, after the anonymous namespace) so it can take
  // the addresses of the internal-linkage constants above. Ranges are deliberately wide.
  const std::vector<Tunable> &tunables() {
    static const std::vector<Tunable> T = {
            {"RFP_MARGIN", &RFP_MARGIN, 20, 200},
            {"RFP_MAX_DEPTH", &RFP_MAX_DEPTH, 4, 10},
            {"FUTILITY_MARGIN", &FUTILITY_MARGIN, 40, 220},
            {"FUTILITY_MAX_DEPTH", &FUTILITY_MAX_DEPTH, 2, 8},
            {"LMP_MAX_DEPTH", &LMP_MAX_DEPTH, 2, 8},
            {"RAZOR_MARGIN", &RAZOR_MARGIN, 80, 400},
            {"RAZOR_MAX_DEPTH", &RAZOR_MAX_DEPTH, 1, 6},
            {"HISTORY_PRUNE_MARGIN", &HISTORY_PRUNE_MARGIN, 1024, 16384},
            {"HISTORY_PRUNE_MAX_DEPTH", &HISTORY_PRUNE_MAX_DEPTH, 2, 8},
            {"CUTNODE_R", &CUTNODE_R, 0, 3},
            {"NMP_EVAL_DIV", &NMP_EVAL_DIV, 80, 400},
            {"SEE_QUIET_MARGIN", &SEE_QUIET_MARGIN, 20, 150},
            {"SEE_CAPT_MARGIN", &SEE_CAPT_MARGIN, 5, 60},
            {"SEE_PRUNE_MAX_DEPTH", &SEE_PRUNE_MAX_DEPTH, 5, 14},
            {"SINGULAR_MIN_DEPTH", &SINGULAR_MIN_DEPTH, 5, 12},
            {"SINGULAR_MARGIN", &SINGULAR_MARGIN, 1, 6},
            {"DOUBLE_EXT_MARGIN", &DOUBLE_EXT_MARGIN, 8, 200},
            {"PROBCUT_MARGIN", &PROBCUT_MARGIN, 60, 300},
            {"PROBCUT_MIN_DEPTH", &PROBCUT_MIN_DEPTH, 3, 8},
            {"PROBCUT_REDUCTION", &PROBCUT_REDUCTION, 3, 6},
            {"DELTA_MARGIN", &DELTA_MARGIN, 80, 400},
            {"CORR_GRAIN", &CORR_GRAIN, 8, 96},
            {"CORR_LIMIT", &CORR_LIMIT, 256, 4096},
    };
    return T;
  }

  bool set_tunable(const std::string &name, int value) {
    for (const auto &t: tunables())
      if (name == t.name) {
        *t.ptr = std::clamp(value, t.min, t.max);
        return true;
      }
    return false;
  }

  void request_stop() { g_stop.store(true, std::memory_order_relaxed); }

  void new_game() {
    // Forget the previous game's move-ordering statistics (the tables persist across "go"s within
    // a game — see Heuristics). Called from "ucinewgame", i.e. never while a search runs.
    for (auto &h: g_heur)
      if (h)
        std::memset(static_cast<void *>(h.get()), 0, sizeof(Heuristics));
  }

  void request_ponderhit(int64_t soft_ms, int64_t hard_ms) {
    // The predicted move was actually played, so our clock starts now: arm the time control relative
    // to this moment and leave ponder mode. The in-progress search then runs as a normal timed one.
    arm_time(std::chrono::steady_clock::now(), soft_ms, hard_ms);
    g_ponder.store(false, std::memory_order_relaxed);
  }

  Result think(const Position &pos, int max_depth, int threads, const InfoCallback &on_iteration, int64_t soft_ms,
               int64_t hard_ms, bool ponder, uint64_t max_nodes) {
    if (is_illegal(pos))
      return Result{Move{}, 0, 0}; // refuse to search -> caller emits "bestmove 0000"
    if (max_depth < 1)
      max_depth = 1;
    if (max_depth > MAX_DEPTH) // "go infinite" passes MAX_DEPTH; never iterate past the ply ceiling
      max_depth = MAX_DEPTH;
    if (threads < 1)
      threads = 1;

    g_stop.store(false, std::memory_order_relaxed); // fresh search: clear any leftover stop request
    tt::new_search(); // age the table: this search's entries outrank earlier moves' leftovers

    // Arm the time control before any thread starts (so all threads see the flags/deadlines). While
    // pondering we run unbounded (like "infinite"); request_ponderhit() arms the clock later. The
    // same `t0` is reused for the reported elapsed time.
    const auto t0 = std::chrono::steady_clock::now();
    g_ponder.store(ponder, std::memory_order_relaxed);
    if (ponder) {
      g_timed.store(false, std::memory_order_relaxed);
      g_soft.store(false, std::memory_order_relaxed);
    } else
      arm_time(t0, soft_ms, hard_ms);
    g_max_nodes.store(ponder ? 0 : max_nodes, std::memory_order_relaxed); // node cap (unbounded while pondering)

    // Persistent per-thread heuristics: make sure a slot exists for every thread index. Done before
    // any helper spawns (and "go" is serialised by uci.cpp), so the vector never grows while a
    // search thread might read it. The tables deliberately carry over from the previous searches of
    // this game; new_game() ("ucinewgame") is what clears them.
    if (g_heur.size() < static_cast<size_t>(threads))
      g_heur.resize(static_cast<size_t>(threads));
    for (int t = 0; t < threads; ++t)
      if (!g_heur[static_cast<size_t>(t)])
        g_heur[static_cast<size_t>(t)] = std::make_unique<Heuristics>(); // value-init -> zeroed

    // Lazy SMP: helper threads search the same root on their own board copies, all sharing the
    // (lockless) transposition table. Their work fills the TT, giving the main thread extra
    // cut-offs. Every thread aborts when g_stop is set.
    std::atomic<uint64_t>    total_nodes{0}; // nodes summed across all threads (for the reported nps)
    std::vector<std::thread> helpers;
    helpers.reserve(static_cast<size_t>(threads - 1));
    for (int t = 1; t < threads; ++t)
      helpers.emplace_back([&pos, max_depth, t, &total_nodes]() {
        t_stop          = &g_stop; // abort promptly when a stop is requested
        t_nodes         = 0; // fresh node count for the node cap
        t_heur          = g_heur[static_cast<size_t>(t)].get(); // this thread's persistent slot
        Position  local = clone(pos);
        const int start = 1 + (t % 2); // stagger start depth so helpers don't march in lockstep
        while (!g_stop.load(std::memory_order_relaxed))
          for (int d = start; d <= max_depth && !g_stop.load(std::memory_order_relaxed); ++d)
            total_nodes.fetch_add(search_to_depth(local, d, -INF, INF).nodes, std::memory_order_relaxed);
      });

    // Main thread: iterative deepening, reporting each completed depth.
    t_stop              = &g_stop; // the main search thread is interruptible too
    t_nodes             = 0; // fresh node count for the node cap
    t_heur              = g_heur[0].get();
    Position main_pos   = clone(pos);
    int      prev_score = 0;
    Result   best{};
    best.best = any_legal_move(main_pos); // guarantees a real move even if stopped before depth 1 finishes

    // Adaptive time management state (main thread, clock-based searches): how many iterations in a
    // row the best move has been the same, and the previous iteration's score.
    Move last_best    = Move{};
    int  stable       = 0;
    int  prev_iter_sc = 0;
    bool have_prev_sc = false;

    for (int d = 1; d <= max_depth; ++d) {
      Result r = aspiration_search(main_pos, d, prev_score);
      total_nodes.fetch_add(r.nodes, std::memory_order_relaxed);
      if (aborted()) // this iteration was interrupted: discard it, keep the last completed one
        break;
      best       = r;
      prev_score = r.score;
      const auto ms =
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
      const uint64_t nodes = total_nodes.load(std::memory_order_relaxed); // all threads so far
      best.nodes           = nodes;
      on_iteration(d, best, nodes, static_cast<long long>(ms));
      if (is_mate(best.score))
        break; // forced mate found; deeper search cannot find a faster one

      // Adaptive soft limit (clock searches only): scale the optimum by how settled the search is.
      // A best move that keeps changing — or a score that just dropped — earns more time; a move
      // that has been stable for several iterations lets us stop early and bank time for later moves.
      // The hard deadline (negamax) is the safety net, so a long scale never risks flagging.
      if (g_soft.load(std::memory_order_acquire)) {
        stable    = best.best == last_best ? std::min(stable + 1, 10) : 0;
        last_best = best.best;

        double scale = 1.2 - 0.07 * stable; // 1.2 (just changed) down to ~0.5 (stable for ~10 iters)
        if (have_prev_sc && best.score + 30 < prev_iter_sc)
          scale += 0.3; // the position got worse — look harder for something better
        // Node-effort: a best move that consumed a large share of this iteration's nodes is clearly
        // best -> bank time; a small share (the choice was contested or needed aspiration re-searches)
        // -> spend more. Centred near the typical effort (~0.30) so the average move is left unchanged.
        const double effort = r.nodes ? static_cast<double>(r.best_nodes) / static_cast<double>(r.nodes) : 0.30;
        scale *= std::clamp(1.0 + 0.6 * (0.30 - effort), 0.85, 1.20);
        // Tighter ceiling than before (was 1.4 base / 1.8 cap): the opening is full of unstable
        // best moves, and spending ~1.8x on each of those front-loaded the clock and left the engine
        // on increment by the middlegame. A 1.5 cap banks that time for later instead.
        scale = std::clamp(scale, 0.5, 1.5);

        prev_iter_sc = best.score;
        have_prev_sc = true;

        const int64_t soft_elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - g_soft_start)
                        .count();
        if (static_cast<double>(soft_elapsed) >= static_cast<double>(g_soft_ms) * scale)
          break; // the (scaled) optimum is spent: starting another iteration would likely overrun it
      }
    }

    // If the search ran out (mate / depth ceiling / soft limit) while still pondering, UCI forbids
    // emitting a move until the GUI says so, so wait for ponderhit (clears g_ponder) or stop.
    while (g_ponder.load(std::memory_order_relaxed) && !g_stop.load(std::memory_order_relaxed))
      std::this_thread::sleep_for(std::chrono::milliseconds(1));

    g_stop.store(true, std::memory_order_relaxed); // stop the helpers
    for (std::thread &h: helpers)
      h.join();
    t_stop = nullptr;
    g_timed.store(false, std::memory_order_relaxed); // disarm so a later untimed search isn't affected
    g_soft.store(false, std::memory_order_relaxed);
    g_ponder.store(false, std::memory_order_relaxed);
    return best;
  }

} // namespace search
