#include "search.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
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

    // Forward-pruning margins for shallow non-PV nodes (futility-pruning family).
    constexpr int RFP_MAX_DEPTH      = 6; // reverse futility (static null move) up to this depth
    constexpr int RFP_MARGIN         = 80; // per-depth eval surplus over beta needed to prune
    constexpr int FUTILITY_MAX_DEPTH = 4; // move-loop futility up to this depth
    constexpr int FUTILITY_MARGIN    = 100; // per-depth eval deficit below alpha that prunes quiets
    constexpr int LMP_MAX_DEPTH      = 4; // late-move (move-count) pruning up to this depth

    // Singular extensions: only attempted from this depth; the verification search excludes the TT
    // move and uses a window `SINGULAR_MARGIN * depth` below the TT score.
    constexpr int SINGULAR_MIN_DEPTH = 8;
    constexpr int SINGULAR_MARGIN    = 2;

    constexpr int DELTA_MARGIN = 200; // qsearch delta pruning: safety margin above the captured value

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

    // Per-thread move-ordering heuristics (Lazy-SMP threads keep their own, so no locking):
    //  - killers: quiet moves that caused a beta cut-off at a given ply in a sibling node
    //  - history: how often a quiet (color, from, to) move has caused a cut-off
    thread_local Move t_killers[MAX_PLY][2];
    thread_local int  t_history[NCOLORS][NSQUARES][NSQUARES];

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

    // Decrement the countdown; every TIME_CHECK_INTERVAL nodes, read the clock and set the global
    // stop if the hard deadline passed. Returns true if the search should abort now (stop or time up).
    [[gnu::hot, gnu::always_inline]] inline bool stop_or_time_up() noexcept {
      if (--t_time_count <= 0) {
        t_time_count = TIME_CHECK_INTERVAL;
        if (time_up())
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

    void clear_heuristics() noexcept {
      std::memset(static_cast<void *>(t_killers), 0, sizeof(t_killers));
      std::memset(t_history, 0, sizeof(t_history));
    }

    // A quiet move is a non-capture, non-promotion move (only these use killer/history ordering).
    [[gnu::const]] bool is_quiet(Move m) noexcept {
      if (m.is_capture())
        return false;
      MoveFlags f = m.flags();
      return !(f >= PR_KNIGHT && f <= PR_QUEEN);
    }

    // On a beta cut-off by a quiet move, reward it: store it as a killer for this ply and bump its
    // history score by depth^2 (capped, so it stays below the killer/capture tiers).
    void update_heuristics(Color c, int ply, Move m, int depth) noexcept {
      if (!is_quiet(m))
        return;
      if (ply < MAX_PLY && !(m == t_killers[ply][0])) {
        t_killers[ply][1] = t_killers[ply][0];
        t_killers[ply][0] = m;
      }
      int &h = t_history[c][m.from()][m.to()];
      h += depth * depth;
      if (h > HISTORY_MAX)
        h = HISTORY_MAX;
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
          Piece victim     = pos.at(m.to()); // NO_PIECE for en passant -> count the victim as a pawn
          int   victim_val = victim == NO_PIECE ? psqt::VALUE[PAWN] : psqt::VALUE[type_of(victim)];
          score += victim_val * 16 - psqt::VALUE[type_of(pos.at(m.from()))];
        }
        if (promo)
          score += psqt::VALUE[KNIGHT + (static_cast<int>(f) & 0b11)];
        return score;
      }

      // Quiet move: killers for this ply, then the history score.
      if (ply < MAX_PLY) {
        if (m == t_killers[ply][0])
          return KILLER1_SCORE;
        if (m == t_killers[ply][1])
          return KILLER2_SCORE;
      }
      return t_history[pos.turn()][m.from()][m.to()];
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
    template<Color Us>
    [[gnu::hot]] int quiescence(Position &p, int alpha, int beta, uint64_t &nodes, int ply) {
      if (stop_or_time_up())
        return alpha; // stop / time up: the value is discarded (whole iteration thrown away)
      if (p.ply() >= Position::MAX_HISTORY - 2)
        return eval::evaluate(p); // absolute undo-stack backstop (see Position::MAX_HISTORY)
      if (ply > t_seldepth)
        t_seldepth = ply; // quiescence reaches beyond the nominal depth -> it sets the selective depth
      int stand_pat = eval::evaluate(p);
      if (stand_pat >= beta)
        return beta;
      if (stand_pat > alpha)
        alpha = stand_pat;

      MoveList<Us> list(p);
      Move         moves[MAX_MOVES];
      int          scores[MAX_MOVES];
      int          n = 0;
      for (Move m: list)
        if (m.is_capture()) {
          moves[n]  = m;
          scores[n] = move_score(p, m, 0); // captures only -> ply unused (no killer/history)
          ++n;
        }

      for (int i = 0; i < n; ++i) {
        pick(moves, scores, i, n);
        Move m = moves[i];

        // Delta pruning: if winning the captured piece (plus a margin) still cannot reach alpha,
        // this capture is hopeless — skip it. Promotions are exempt (they can swing by a queen).
        const MoveFlags f     = m.flags();
        const bool      promo = f >= PC_KNIGHT && f <= PC_QUEEN;
        if (!promo) {
          const Piece victim = p.at(m.to()); // NO_PIECE for en passant -> the victim is a pawn
          const int   gain   = victim == NO_PIECE ? psqt::VALUE[PAWN] : psqt::VALUE[type_of(victim)];
          if (stand_pat + gain + DELTA_MARGIN <= alpha)
            continue;
        }

        p.play<Us>(m);
        ++nodes;
        int score = -quiescence<~Us>(p, -beta, -alpha, nodes, ply + 1);
        p.undo<Us>(m);

        if (score >= beta)
          return beta;
        if (score > alpha)
          alpha = score;
      }
      return alpha;
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
                             Square recap_sq = NO_SQUARE, Move excluded = Move{}) {
      if (stop_or_time_up())
        return 0; // stop requested / hard deadline hit: value is discarded, last completed depth kept
      if (ply > t_seldepth)
        t_seldepth = ply; // track the deepest ply reached, for the reported selective depth
      if (ply >= MAX_PLY || p.ply() >= Position::MAX_HISTORY - MAX_PLY)
        return eval::evaluate(p); // hard cap: bound runaway extensions, and never index history[] OOB
                                  // (the search stacks its plies onto game_ply; in a very long game
                                  //  that could otherwise overrun the undo stack — reserve MAX_PLY for
                                  //  the quiescence that may still run below this node)

      // Draw by repetition / fifty-move rule: score it 0 immediately (never at the root, ply 0, where
      // a move must still be returned). This collapses shuffling/perpetual lines instead of searching
      // them to the depth ceiling — it both plays draws correctly (no repeating in won positions) and
      // is what keeps the PV short (an unbounded repetition PV could otherwise flood stdout).
      if (ply > 0 && p.is_draw())
        return 0;

      t_pv_len[ply] = 0; // no PV from here yet (so an early return leaves an empty line)

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
      tt::Entry *tte     = tt::probe(key);
      Move       ttMove  = Move{};
      int        ttScore = 0;
      const bool tt_hit  = tte->key == key && tte->bound != tt::NONE;
      if (tt_hit) {
        ttMove  = Move(tte->move);
        ttScore = score_from_tt(tte->score, ply);
        // During a singular verification search (excluded set) the stored result is for the full
        // move set, so it must not cut this (reduced) search off.
        if (excluded == Move{} && tte->depth >= depth) {
          if (tte->bound == tt::EXACT || (tte->bound == tt::LOWER && ttScore >= beta) ||
              (tte->bound == tt::UPPER && ttScore <= alpha))
            return ttScore;
        }
      }

      MoveList<Us> list(p);
      const int    n = static_cast<int>(list.size());

      if (n == 0)
        return p.checkers ? -(MATE - ply) : 0; // checkmate (lose) : stalemate (draw)
      if (depth == 0)
        return quiescence<Us>(p, alpha, beta, nodes, ply);

      // Captured now because the null-move / child searches below overwrite p.checkers.
      const bool in_check = p.checkers != 0;

      // Static eval, computed once for the futility-pruning heuristics (only at shallow non-PV,
      // not-in-check, non-mate nodes — where these prunings apply). INF means "not computed".
      int        static_eval = INF;
      const bool can_prune   = !is_pv && !in_check && beta < MATE - MAX_MATE_PLY && beta > -(MATE - MAX_MATE_PLY);
      if (can_prune && depth <= RFP_MAX_DEPTH) {
        static_eval = eval::evaluate(p);
        // Reverse futility pruning (static null move): a depth-scaled margin above beta -> prune.
        if (static_eval - RFP_MARGIN * depth >= beta)
          return static_eval;
      }

      // Null-move pruning: if passing the turn (a free move for the opponent) still leaves us at
      // >= beta after a shallower search, the position is so good we can prune. Skipped when in
      // check (can't pass), shallow, right after a null move, during a singular search, near mate
      // bounds, or in a pawns-only (zugzwang-prone) position.
      bool mate_threat = false;
      if (null_ok && excluded == Move{} && depth >= 3 && !in_check && beta < MATE - MAX_MATE_PLY &&
          has_non_pawn_material(p, Us)) {
        const int R = 2 + depth / 6;
        p.play_null();
        int score = -negamax<~Us>(p, depth - 1 - R, ply + 1, -beta, -beta + 1, nodes, false, NO_SQUARE, Move{});
        p.undo_null();
        if (score >= beta)
          return beta; // fail-high: prune this node
        // Mate-threat extension: if passing lets the opponent force mate, this line is dangerous —
        // search it deeper rather than trusting the shallow reductions.
        if (score <= -(MATE - MAX_MATE_PLY))
          mate_threat = true;
      }

      Move moves[MAX_MOVES];
      int  scores[MAX_MOVES];
      score_moves(p, list.begin(), moves, scores, n, ttMove, ply);

      // Node-level extension (one ply): in check (every reply forced), a null-move-detected mate
      // threat (shallow reductions can't be trusted), or a single legal move (one-reply — also
      // forced) — search the whole node deeper.
      const int node_ext = (in_check || mate_threat || n == 1) ? 1 : 0;

      int  best     = -INF;
      Move bestMove = Move{};
      for (int i = 0; i < n; ++i) {
        pick(moves, scores, i, n);
        Move m = moves[i];
        if (m == excluded)
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
        }

        // Per-move extension (capped at one ply): node-level (check / mate threat), singular TT
        // move, or a recapture on the square the previous move captured on.
        int ext = node_ext;
        if (!ext && depth >= SINGULAR_MIN_DEPTH && m == ttMove && excluded == Move{} && tt_hit &&
            tte->depth >= depth - 3 && tte->bound != tt::UPPER && ttScore > -(MATE - MAX_MATE_PLY) &&
            ttScore < MATE - MAX_MATE_PLY) {
          // Singular extension: search every move *except* the TT move at reduced depth with a
          // window just below the TT score. If none reaches it, the TT move is singular -> extend.
          const int sBeta  = ttScore - SINGULAR_MARGIN * depth;
          const int sDepth = (depth - 1) / 2;
          const int v      = negamax<Us>(p, sDepth, ply, sBeta - 1, sBeta, nodes, false, NO_SQUARE, m);
          if (v < sBeta)
            ext = 1;
        }
        if (!ext && recap_sq != NO_SQUARE && m.is_capture() && m.to() == recap_sq)
          ext = 1; // recapture on the previous capture square -> resolve the exchange one ply deeper
        if (!ext && type_of(p.at(m.from())) == PAWN &&
            ((Us == WHITE && rank_of(m.to()) == RANK7) || (Us == BLACK && rank_of(m.to()) == RANK2)) &&
            eval::is_passed_pawn(p, Us, m.to()))
          ext = 1; // passed pawn pushed to the 7th rank (one from promotion) -> search deeper

        const int    nd          = depth - 1 + ext;
        const Square child_recap = m.is_capture() ? m.to() : NO_SQUARE;

        p.play<Us>(m);
        ++nodes;

        int score;
        if (i == 0) {
          // Principal variation: search the (well-ordered) first move with the full window.
          score = -negamax<~Us>(p, nd, ply + 1, -beta, -alpha, nodes, true, child_recap, Move{});
        } else {
          // Late move reductions: a late, quiet move is unlikely to beat alpha, so scout it at a
          // reduced depth first; only re-search at full depth if that surprises us (beats alpha).
          // Tactical moves (captures/promotions), shallow nodes, and in-check nodes aren't reduced.
          int reduction = 0;
          if (depth >= 3 && i >= 3 && !in_check && is_quiet(m)) {
            reduction = 1 + (depth >= 6 ? 1 : 0) + (i >= 6 ? 1 : 0);
            if (reduction > nd)
              reduction = nd;
          }
          // PVS scout at the (possibly reduced) depth with a null window.
          score = -negamax<~Us>(p, nd - reduction, ply + 1, -alpha - 1, -alpha, nodes, true, child_recap, Move{});
          // Reduced scout beat alpha: re-search at full depth (still a null window).
          if (reduction > 0 && score > alpha)
            score = -negamax<~Us>(p, nd, ply + 1, -alpha - 1, -alpha, nodes, true, child_recap, Move{});
          // PVS: inside the window -> re-search with the full window.
          if (score > alpha && score < beta)
            score = -negamax<~Us>(p, nd, ply + 1, -beta, -alpha, nodes, true, child_recap, Move{});
        }
        p.undo<Us>(m);

        if (score > best) {
          best     = score;
          bestMove = m;
          if (best > alpha) {
            alpha = best;
            update_pv(ply, m); // a new best move at this node -> extend the PV
          }
        }
        if (alpha >= beta) {
          update_heuristics(Us, ply, m, depth); // reward the quiet move that caused the cut-off
          break;
        }
      }

      if (!aborted() && excluded == Move{}) { // don't store a partial (aborted) or singular result
        tt::Bound bound = best <= alpha_orig ? tt::UPPER : (best >= beta ? tt::LOWER : tt::EXACT);
        tt::store(key, bestMove, score_to_tt(best, ply), depth, bound);
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
      Move           ttMove = tte->key == key && tte->bound != tt::NONE ? Move(tte->move) : Move{};

      Move moves[MAX_MOVES];
      int  scores[MAX_MOVES];
      score_moves(p, list.begin(), moves, scores, n, ttMove, 0);

      t_pv_len[0]         = 0;
      const int new_depth = depth - 1 + (p.checkers || n == 1 ? 1 : 0); // check / one-reply extension

      for (int i = 0; i < n; ++i) {
        pick(moves, scores, i, n);
        if (aborted())
          return r; // helper thread is stopping; result is discarded, don't store
        Move         m           = moves[i];
        const Square child_recap = m.is_capture() ? m.to() : NO_SQUARE; // enable recapture extension
        p.play<Us>(m);
        ++r.nodes;

        int score;
        if (i == 0) {
          score = -negamax<~Us>(p, new_depth, 1, -beta, -alpha, r.nodes, true, child_recap, Move{});
        } else {
          score = -negamax<~Us>(p, new_depth, 1, -alpha - 1, -alpha, r.nodes, true, child_recap, Move{}); // PVS scout
          if (score > alpha && score < beta)
            score = -negamax<~Us>(p, new_depth, 1, -beta, -alpha, r.nodes, true, child_recap, Move{}); // re-search
        }
        p.undo<Us>(m);

        if (score > r.score) {
          r.score = score;
          r.best  = m;
          if (score > alpha) {
            alpha = score;
            update_pv(0, m); // a new best root move -> record the PV
          }
        }
        if (alpha >= beta) {
          update_heuristics(Us, 0, m, depth);
          break; // fail-high: the caller re-searches with a wider window
        }
      }

      if (!aborted()) {
        tt::Bound bound = r.score <= alpha_orig ? tt::UPPER : (r.score >= beta ? tt::LOWER : tt::EXACT);
        tt::store(key, r.best, score_to_tt(r.score, 0), depth, bound);
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

  void request_stop() { g_stop.store(true, std::memory_order_relaxed); }

  void request_ponderhit(int64_t soft_ms, int64_t hard_ms) {
    // The predicted move was actually played, so our clock starts now: arm the time control relative
    // to this moment and leave ponder mode. The in-progress search then runs as a normal timed one.
    arm_time(std::chrono::steady_clock::now(), soft_ms, hard_ms);
    g_ponder.store(false, std::memory_order_relaxed);
  }

  Result think(const Position &pos, int max_depth, int threads, const InfoCallback &on_iteration, int64_t soft_ms,
               int64_t hard_ms, bool ponder) {
    if (is_illegal(pos))
      return Result{Move{}, 0, 0}; // refuse to search -> caller emits "bestmove 0000"
    if (max_depth < 1)
      max_depth = 1;
    if (max_depth > MAX_DEPTH) // "go infinite" passes MAX_DEPTH; never iterate past the ply ceiling
      max_depth = MAX_DEPTH;
    if (threads < 1)
      threads = 1;

    g_stop.store(false, std::memory_order_relaxed); // fresh search: clear any leftover stop request

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

    // Lazy SMP: helper threads search the same root on their own board copies, all sharing the
    // (lockless) transposition table. Their work fills the TT, giving the main thread extra
    // cut-offs. Every thread aborts when g_stop is set.
    std::atomic<uint64_t>    total_nodes{0}; // nodes summed across all threads (for the reported nps)
    std::vector<std::thread> helpers;
    helpers.reserve(static_cast<size_t>(threads - 1));
    for (int t = 1; t < threads; ++t)
      helpers.emplace_back([&pos, max_depth, t, &total_nodes]() {
        t_stop          = &g_stop; // abort promptly when a stop is requested
        Position  local = clone(pos);
        const int start = 1 + (t % 2); // stagger start depth so helpers don't march in lockstep
        while (!g_stop.load(std::memory_order_relaxed))
          for (int d = start; d <= max_depth && !g_stop.load(std::memory_order_relaxed); ++d)
            total_nodes.fetch_add(search_to_depth(local, d, -INF, INF).nodes, std::memory_order_relaxed);
      });

    // Main thread: iterative deepening, reporting each completed depth. (Helper threads are freshly
    // spawned, so their thread_local killer/history tables are already zero-initialised.)
    t_stop = &g_stop; // the main search thread is interruptible too
    clear_heuristics();
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

        double scale = 1.4 - 0.08 * stable; // 1.4 (just changed) down to 0.6 (very stable)
        if (have_prev_sc && best.score + 30 < prev_iter_sc)
          scale += 0.3; // the position got worse — look harder for something better
        scale = std::clamp(scale, 0.5, 1.8);

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
