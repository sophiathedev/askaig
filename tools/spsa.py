#!/usr/bin/env python3
"""Self-play SPSA. RUN_ID starts a run; RESUME restores its checkpoint.

ENGINE, FASTCHESS, BOOK, PARAMS, INIT select inputs for a new run.
TC, HASH, CONCURRENCY, ITERS, GAMES, STEP_FRAC, PERT_FRAC, SEED and
MATCH_TIMEOUT are frozen in checkpoint.json. See tools/README.md.
"""

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from sprt_state import write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "work" / "spsa-runs"
SETTINGS = {
    "TC": (str, "4+0.04"), "HASH": (int, 16), "CONCURRENCY": (int, 6),
    "ITERS": (int, 6000), "GAMES": (int, 8), "STEP_FRAC": (float, 0.05),
    "PERT_FRAC": (float, 0.10), "SEED": (int, 20260614), "MATCH_TIMEOUT": (float, 3600),
}


def executable(env_name, candidates):
    for candidate in ([os.environ[env_name]] if env_name in os.environ else candidates):
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return Path(candidate).resolve()
    raise ValueError(f"set {env_name} to an executable file")


def discover_params(engine):
    result = subprocess.run([str(engine), "--debug"], input="uci\nquit\n", capture_output=True,
                            text=True, timeout=30, check=True)
    if "uciok" not in result.stdout:
        raise ValueError("engine did not finish UCI discovery")
    pattern = r"option name (\S+) type spin default (-?\d+) min (-?\d+) max (-?\d+)"
    wanted = [p.strip() for p in os.environ.get("PARAMS", "").split(",") if p.strip()]
    params = []
    for name, default, lo, hi in re.findall(pattern, result.stdout):
        if name in ("Hash", "Threads", "Contempt") or name.startswith("Syzygy"):
            continue
        if wanted and not any(name.startswith(p) for p in wanted):
            continue
        params.append([name, int(default), int(lo), int(hi)])
    if not params or len({p[0] for p in params}) != len(params):
        raise ValueError("no tunable parameters, or duplicate option names")
    for name, default, lo, hi in params:
        if not lo <= default <= hi or lo == hi:
            raise ValueError(f"invalid parameter bounds: {name}")
    if wanted and any(not any(p[0].startswith(w) for p in params) for w in wanted):
        raise ValueError("PARAMS contains an unmatched prefix")
    return params


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initial_values(params, path):
    theta = {name: float(default) for name, default, lo, hi in params}
    if not path:
        return theta
    bounds = {name: (lo, hi) for name, default, lo, hi in params}
    seen = set()
    for line in Path(path).read_text().splitlines():
        match = re.fullmatch(r"\s*(\w+)\s*=\s*(-?\d+)\s*", line)
        if not line.strip():
            continue
        if not match or match[1] not in bounds or match[1] in seen:
            raise ValueError(f"invalid INIT entry: {line}")
        name, value = match[1], int(match[2])
        if not bounds[name][0] <= value <= bounds[name][1]:
            raise ValueError(f"INIT value outside bounds: {name}")
        theta[name] = float(value)
        seen.add(name)
    if not seen:
        raise ValueError("INIT contains no parameters")
    return theta


def validate_config(config):
    for key in ("HASH", "CONCURRENCY", "ITERS", "GAMES"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if config["GAMES"] % 2:
        raise ValueError("GAMES must be even for color pairs")
    for key in ("STEP_FRAC", "PERT_FRAC", "MATCH_TIMEOUT"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")


def new_state(run):
    config = {key: convert(os.environ.get(key, default))
              for key, (convert, default) in SETTINGS.items()}
    validate_config(config)
    engine = executable("ENGINE", [ROOT / "build-pgo/askaig", ROOT / "cmake-build-release/askaig"])
    fc = executable("FASTCHESS", [Path.home() / "Downloads/fastchess-mac-arm64/fastchess",
                                 HERE / "fastchess-src/fastchess", shutil.which("fastchess")])
    book = Path(os.environ.get("BOOK", HERE / "books/UHO_4060_v4.epd")).resolve()
    files = {}
    for name, source in (("engine", engine), ("fastchess", fc), ("book.epd", book)):
        shutil.copy2(source, run / name)
        files[name] = checksum(run / name)
    params = discover_params(run / "engine")
    theta = initial_values(params, os.environ.get("INIT"))
    return {"version": 1, "config": config, "params": params, "theta": theta, "iteration": 0,
            "rng": random.Random(config["SEED"]).getstate(), "files": files,
            "tuner_sha256": checksum(__file__)}


def restore_rng(state):
    version, values, gaussian = state["rng"]
    rng = random.Random()
    rng.setstate((version, tuple(values), gaussian))
    return rng


def load_state(run):
    state = json.loads((run / "checkpoint.json").read_text())
    if state["version"] != 1 or state["tuner_sha256"] != checksum(__file__):
        raise ValueError("checkpoint/tuner version mismatch")
    config = state["config"]
    validate_config(config)
    for key, (convert, default) in SETTINGS.items():
        if key in os.environ and convert(os.environ[key]) != config[key]:
            raise ValueError(f"cannot change {key} on resume; start a new run")
    if any(key in os.environ for key in ("ENGINE", "FASTCHESS", "BOOK", "PARAMS", "INIT")):
        raise ValueError("resume uses frozen inputs; unset ENGINE/FASTCHESS/BOOK/PARAMS/INIT")
    for name in ("engine", "fastchess", "book.epd"):
        if checksum(run / name) != state["files"][name]:
            raise ValueError(f"frozen input checksum mismatch: {name}")
    if not 0 <= state["iteration"] <= config["ITERS"]:
        raise ValueError("invalid checkpoint iteration")
    if set(state["theta"]) != {p[0] for p in state["params"]}:
        raise ValueError("checkpoint parameter mismatch")
    for name, default, lo, hi in state["params"]:
        value = state["theta"][name]
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"invalid checkpoint value: {name}")
    restore_rng(state)
    return state


def trial(state):
    config, theta = state["config"], state["theta"]
    k, A = state["iteration"] + 1, 0.1 * config["ITERS"]
    rng = restore_rng(state)
    plus, minus, steps = {}, {}, {}
    for name, default, lo, hi in state["params"]:
        ak = max(1.0, config["STEP_FRAC"] * (hi - lo)) * ((1 + A) / (k + A)) ** 0.602
        ck = max(1.0, max(1.0, config["PERT_FRAC"] * (hi - lo)) / k ** 0.101)
        delta = rng.choice((-1, 1))
        plus[name] = max(lo, min(hi, round(theta[name] + ck * delta)))
        minus[name] = max(lo, min(hi, round(theta[name] - ck * delta)))
        steps[name] = ak * delta
    seed = rng.randrange(1, 2 ** 31)
    return plus, minus, steps, seed, rng.getstate()


def parse_score(output, returncode, games):
    if returncode:
        raise ValueError(f"fastchess exited with status {returncode}")
    failures = r"(?:Crashed|Timeouts):\s*[1-9]\d*|loses on time|stall|disconnect|illegal move|\bcrash\b|engine.*crashed"
    if re.search(failures, output, re.IGNORECASE):
        raise ValueError("engine crash, timeout or protocol failure")
    if re.search(r"(?:unknown|invalid|unrecognized) option|option.*(?:rejected|ignored)", output, re.IGNORECASE):
        raise ValueError("engine/fastchess rejected an option")
    blocks = re.findall(r"Results of plus vs minus[^\n]*\n(.*?)(?=Results of|\Z)", output, re.DOTALL)
    scores = re.findall(r"Games:\s*(\d+),\s*Wins:\s*(\d+),\s*Losses:\s*(\d+),\s*Draws:\s*(\d+)",
                        blocks[-1] if blocks else "")
    if not scores or "Finished match" not in output:
        raise ValueError("missing final plus-vs-minus result")
    total, wins, losses, draws = map(int, scores[-1])
    if total != games or wins + losses + draws != games:
        raise ValueError(f"incomplete match: expected {games}, got {total}/{wins + losses + draws}")
    return wins, losses, draws


def play_match(run, config, plus, minus, seed, log):
    def options(values):
        return [f"option.{key}={value}" for key, value in values.items()]

    cmd = [str(run / "fastchess"),
           "-engine", f"cmd={run / 'engine'}", "name=plus", "args=--debug", *options(plus),
           "-engine", f"cmd={run / 'engine'}", "name=minus", "args=--debug", *options(minus),
           "-each", f"tc={config['TC']}", f"option.Hash={config['HASH']}", "option.Threads=1", "proto=uci",
           "-openings", f"file={run / 'book.epd'}", "format=epd", "order=random",
           "-games", "2", "-rounds", str(config["GAMES"] // 2), "-repeat",
           "-concurrency", str(config["CONCURRENCY"]), "-srand", str(seed),
           "-output", "format=fastchess", "-ratinginterval", str(config["GAMES"])]
    with log.open("w") as output:
        proc = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT, start_new_session=True, cwd=run)
        try:
            returncode = proc.wait(timeout=config["MATCH_TIMEOUT"])
        except subprocess.TimeoutExpired:
            raise ValueError(f"batch exceeded MATCH_TIMEOUT={config['MATCH_TIMEOUT']} seconds") from None
        finally:
            # include engines, not just the match runner
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    return parse_score(log.read_text(errors="replace"), returncode, config["GAMES"])


def export_values(run, state):
    temp = run / "spsa.tuned.tmp"
    temp.write_text("".join(f"{name} = {round(state['theta'][name])}\n" for name, *_ in state["params"]))
    os.replace(temp, run / "spsa.tuned")


def run_iterations(run, state, match=play_match):
    export_values(run, state)
    started, first = time.monotonic(), state["iteration"]
    config = state["config"]
    print(f"SPSA: {len(state['params'])} params, iteration {first}/{config['ITERS']}, run={run}", flush=True)
    while state["iteration"] < config["ITERS"]:
        k = state["iteration"] + 1
        plus, minus, steps, seed, rng = trial(state)
        fd, name = tempfile.mkstemp(prefix=f"match-{k:06d}-", suffix=".log", dir=run)
        os.close(fd)
        log = Path(name)
        try:
            wins, losses, draws = match(run, config, plus, minus, seed, log)
            if wins + losses + draws != config["GAMES"]:
                raise ValueError("incomplete batch")
        except BaseException:
            print(f"batch {k} not committed; log={log}\nresume: RESUME={run} python3 tools/spsa.py", flush=True)
            raise
        y = (wins - losses) / config["GAMES"]
        theta = {name: max(lo, min(hi, state["theta"][name] + steps[name] * y))
                 for name, default, lo, hi in state["params"]}
        updated = {**state, "theta": theta, "iteration": k, "rng": rng,
                   "last_match": {"log": log.name, "seed": seed, "score": [wins, losses, draws]}}
        write_json(run / "checkpoint.json", updated)
        state = updated
        export_values(run, state)
        rate = (k - first) * config["GAMES"] / max(1e-9, time.monotonic() - started)
        print(f"[{k}/{config['ITERS']}] plus {wins}-{losses}-{draws} (y={y:+.2f}) {rate:.1f} g/s", flush=True)
    print(f"finished; values={run / 'spsa.tuned'}", flush=True)


def main():
    resume = os.environ.get("RESUME")
    if resume:
        if "RUN_ID" in os.environ:
            raise ValueError("use RESUME without RUN_ID")
        path = Path(resume)
        if path.suffix == ".tuned":
            raise ValueError("old .tuned files are warm starts: use INIT, not RESUME")
        if path.name == "checkpoint.json":
            path = path.parent
        run = (path if path.is_dir() else RUNS / path).resolve()
        if not (run / "checkpoint.json").is_file():
            raise ValueError(f"checkpoint missing: {run}")
    else:
        RUNS.mkdir(parents=True, exist_ok=True)
        run_id = os.environ.get("RUN_ID")
        if run_id:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
                raise ValueError("RUN_ID must be a simple directory name")
            run = RUNS / run_id
            run.mkdir()
        else:
            run = Path(tempfile.mkdtemp(prefix="run-", dir=RUNS))
    with (run / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"run already active: {run}") from None
        state = load_state(run) if resume else new_state(run)
        if not resume:
            write_json(run / "checkpoint.json", state)
        run_iterations(run, state)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted; resume from the last committed batch", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f"SPSA stopped: {exc}", file=sys.stderr)
        sys.exit(1)
