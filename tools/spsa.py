#!/usr/bin/env python3
"""SPSA tuner for askaig's search constants.

SPSA (Simultaneous Perturbation Stochastic Approximation) tunes parameters that have NO per-position
ground-truth label — the search margins / depth gates, whose only effect is on playing strength. Each
iteration it perturbs ALL parameters at once by a random +/- step, plays a short self-play match
between the "+" and "-" configurations, and nudges every parameter toward whichever side scored
better. Over many iterations the noise averages out and the parameters drift toward a strength optimum.

The tunable list is discovered automatically from the engine's `uci` output (the `option ... type
spin` lines it advertises, minus Hash/Threads), so it always matches what the binary exposes — no
hardcoding. Parameters are driven per game via fastchess `option.NAME=value`.

Usage:
    tools/spsa.py                       # defaults: 6000 iters, 8 games/iter, tc 4+0.04
    ITERS=3000 GAMES=8 TC=5+0.05 tools/spsa.py
    ENGINE=cmake-build-release/askaig FASTCHESS=~/fastchess tools/spsa.py

Env knobs: ENGINE, FASTCHESS, BOOK, TC, HASH, CONCURRENCY, ITERS, GAMES, STEP_FRAC, PERT_FRAC, SEED.
Writes tools/work/spsa.tuned (paste-ready) every 25 iterations and at the end. Ctrl-C saves and exits.
"""
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORK = os.path.join(HERE, "work")


def env(name, default):
    return os.environ.get(name, default)


def find_engine():
    if env("ENGINE", ""):
        return os.path.abspath(env("ENGINE", ""))
    for c in ("cmake-build-release/askaig", "build/askaig", "build-dev/askaig"):
        p = os.path.join(ROOT, c)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    sys.exit("engine not found — build it, or set ENGINE=/path/to/askaig")


def find_fastchess():
    if env("FASTCHESS", ""):
        return os.path.abspath(env("FASTCHESS", ""))
    for c in (
        os.path.expanduser("~/Downloads/fastchess-mac-arm64/fastchess"),
        os.path.join(HERE, "fastchess-src", "fastchess"),
        shutil.which("fastchess") or "",
    ):
        if c and os.access(c, os.X_OK):
            return c
    sys.exit("fastchess not found — set FASTCHESS=/path/to/fastchess (or run tools/setup.sh)")


def discover_params(engine):
    """Run `uci`, return [(name, default, min, max)] for the search tunables (spin options)."""
    out = subprocess.run([engine], input="uci\nquit\n", capture_output=True, text=True, timeout=30).stdout
    pat = re.compile(r"option name (\S+) type spin default (-?\d+) min (-?\d+) max (-?\d+)")
    params = []
    for name, d, lo, hi in pat.findall(out):
        if name in ("Hash", "Threads"):
            continue
        params.append((name, int(d), int(lo), int(hi)))
    if not params:
        sys.exit("no tunable spin options found — does this build expose them? (rebuild current tree)")
    return params


# fastchess prints a "Results of plus vs minus (...)" block ending in this line; W/L/D are from the
# FIRST engine's (plus's) point of view. With -ratinginterval there are several — the last is the total.
_SCORE = re.compile(r"Wins:\s*(\d+),\s*Losses:\s*(\d+),\s*Draws:\s*(\d+)")


def play_match(fc, engine, book, tc, hash_mb, conc, games, plus, minus):
    """Play `games` games (color-balanced) between configs `plus` and `minus` (name->int). Returns
    (wins, losses, draws) from plus's point of view."""
    rounds = max(1, games // 2)

    def opts(cfg):
        return [f"option.{k}={v}" for k, v in cfg.items()]

    cmd = [
        fc,
        "-engine", f"cmd={engine}", "name=plus", *opts(plus),
        "-engine", f"cmd={engine}", "name=minus", *opts(minus),
        "-each", f"tc={tc}", f"option.Hash={hash_mb}", "option.Threads=1", "proto=uci",
        "-openings", f"file={book}", "format=epd", "order=random",
        "-games", "2", "-rounds", str(rounds), "-repeat",
        "-concurrency", str(conc), "-recover",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = None
    for m in _SCORE.finditer(out):  # keep the last (final) score line
        pass
    if not m:
        return 0, 0, 0  # match failed / no result -> treat as no signal
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def save(path, params, theta):
    with open(path, "w") as f:
        for (name, _d, _lo, _hi) in params:
            f.write(f"{name} = {int(round(theta[name]))}\n")


def main():
    engine = find_engine()
    fc = find_fastchess()
    book = env("BOOK", os.path.join(HERE, "books", "UHO_4060_v3.epd"))
    tc = env("TC", "4+0.04")
    hash_mb = int(env("HASH", "16"))
    conc = int(env("CONCURRENCY", "6"))
    iters = int(env("ITERS", "6000"))
    games = int(env("GAMES", "8"))
    step_frac = float(env("STEP_FRAC", "0.05"))  # initial learning step as a fraction of each range
    pert_frac = float(env("PERT_FRAC", "0.10"))  # perturbation size as a fraction of each range
    random.seed(int(env("SEED", "20260614")))

    if not os.path.isfile(book):
        sys.exit(f"opening book missing: {book} (run tools/setup.sh)")
    os.makedirs(WORK, exist_ok=True)
    out_path = os.path.join(WORK, "spsa.tuned")

    params = discover_params(engine)
    theta = {n: float(d) for (n, d, _lo, _hi) in params}
    bounds = {n: (lo, hi) for (n, _d, lo, hi) in params}
    c0 = {n: max(1.0, pert_frac * (hi - lo)) for (n, _d, lo, hi) in params}
    a0 = {n: max(1.0, step_frac * (hi - lo)) for (n, _d, lo, hi) in params}
    A = 0.1 * iters
    alpha, gamma = 0.602, 0.101

    print(f"SPSA: {len(params)} params, engine={os.path.relpath(engine, ROOT)}, tc={tc}, "
          f"{games} games/iter x {iters} iters (~{games*iters} games)")
    print("       params:", ", ".join(n for (n, *_ ) in params))
    print(f"       defaults: {{{', '.join(f'{n}={int(theta[n])}' for (n,*_) in params)}}}\n", flush=True)

    t0 = time.time()
    try:
        for k in range(1, iters + 1):
            ak = {n: a0[n] * ((1 + A) / (k + A)) ** alpha for n in theta}
            ck = {n: max(1.0, c0[n] * (1.0 / k) ** gamma) for n in theta}
            delta = {n: random.choice((-1, 1)) for n in theta}

            def clip(n, v):
                lo, hi = bounds[n]
                return max(lo, min(hi, int(round(v))))

            plus = {n: clip(n, theta[n] + ck[n] * delta[n]) for n in theta}
            minus = {n: clip(n, theta[n] - ck[n] * delta[n]) for n in theta}

            w, l, d = play_match(fc, engine, book, tc, hash_mb, conc, games, plus, minus)
            n_games = w + l + d
            y = (w - l) / n_games if n_games else 0.0  # plus's score in [-1, 1]

            for n in theta:  # nudge every parameter toward the better-scoring side
                theta[n] = max(bounds[n][0], min(bounds[n][1], theta[n] + ak[n] * y * delta[n]))

            rate = k * games / max(1e-9, time.time() - t0)
            print(f"[{k:4d}/{iters}] plus {w}-{l}-{d} (y={y:+.2f})  {rate:.1f} g/s", flush=True)
            if k % 25 == 0:
                save(out_path, params, theta)
                print("  current: " + " ".join(f"{n}={int(round(theta[n]))}" for (n, *_) in params), flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted — saving current values")

    save(out_path, params, theta)
    print("\n=== tuned search constants (saved to tools/work/spsa.tuned) ===")
    for (name, d, _lo, _hi) in params:
        v = int(round(theta[name]))
        print(f"{name:24s} = {v:6d}   (default {d}{'  <-- changed' if v != d else ''})")


if __name__ == "__main__":
    main()
