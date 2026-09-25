#!/usr/bin/env python3
import os
from pathlib import Path
import select
import subprocess
import sys
import time


def check(engine, turn, limits):
    with subprocess.Popen([str(engine)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT) as proc:
        pending = b""

        def send(command):
            proc.stdin.write((command + "\n").encode())
            proc.stdin.flush()

        def wait_for(prefix, timeout):
            nonlocal pending
            deadline = time.monotonic() + timeout
            while True:
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if line.startswith(prefix):
                        return line.decode()
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([proc.stdout], [], [], remaining)[0]:
                    raise AssertionError(f"{turn}: go {limits}: no {prefix.decode()} within {timeout}s")
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    raise AssertionError(f"engine exited: {proc.poll()}")
                pending += chunk

        try:
            send("isready")
            wait_for(b"readyok", 10)
            send("position startpos" + (" moves e2e4" if turn == "black" else ""))
            send("go " + limits)
            result = wait_for(b"bestmove ", 1)
            assert result.split()[1] != "0000", result
        finally:
            if proc.poll() is None:
                send("quit")
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
        assert proc.returncode == 0, f"engine exited: {proc.returncode}"
        print(f"PASS {turn}: go {limits}: {result}")


if __name__ == "__main__":
    engine = Path(sys.argv[1] if len(sys.argv) > 1 else "cmake-build-release/askaig").resolve()
    for turn in ("white", "black"):
        for limits in ("movetime 0 depth 64", "wtime 0 btime 0 depth 64",
                       f"{'wtime' if turn == 'white' else 'btime'} 0 depth 64",
                       "movetime 50 depth 64", "wtime 100 btime 100 depth 64",
                       "depth 3", "nodes 1000"):
            check(engine, turn, limits)
