#!/usr/bin/env python3
import contextlib
import copy
import fcntl
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import spsa


VALID = """Results of plus vs minus (1+0.01):
Games: 8, Wins: 3, Losses: 2, Draws: 3, Points: 4.5
Player: plus
  Timeouts: 0
  Crashed: 0
Finished match
"""


class SpsaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def state(self, run=None):
        run = run or self.run
        config = {key: default for key, (_, default) in spsa.SETTINGS.items()}
        config.update(ITERS=4, GAMES=8)
        for name in ("engine", "fastchess", "book.epd"):
            (run / name).write_text(name)
        state = {"version": 1, "config": config,
                 "params": [["LMR_BASE", 80, 30, 150], ["MAT_BASE", 736, 550, 950]],
                 "theta": {"LMR_BASE": 80.125, "MAT_BASE": 736.375}, "iteration": 0,
                 "rng": random.Random(42).getstate(),
                 "files": {name: spsa.checksum(run / name) for name in ("engine", "fastchess", "book.epd")},
                 "tuner_sha256": spsa.checksum(spsa.__file__)}
        spsa.write_json(run / "checkpoint.json", state)
        return state

    def test_match_validation(self):
        self.assertEqual(spsa.parse_score(VALID, 0, 8), (3, 2, 3))
        for output, code in [(VALID, 1), ("", 0), (VALID.replace("plus vs minus", "minus vs plus"), 0),
                             (VALID.replace("Games: 8", "Games: 6"), 0),
                             (VALID.replace("Draws: 3", "Draws: 2"), 0),
                             (VALID.replace("Finished match", ""), 0),
                             (VALID.replace("Crashed: 0", "Crashed: 1"), 0),
                             (VALID.replace("Timeouts: 0", "Timeouts: 1"), 0),
                             (VALID + "White loses on time", 0), (VALID + "engine disconnected", 0),
                             (VALID + "illegal move", 0), (VALID + "Unknown option LMR_BASE", 0)]:
            with self.subTest(output=output, code=code), self.assertRaises(ValueError):
                spsa.parse_score(output, code, 8)

    def test_discovery_excludes_operational_options(self):
        output = "\n".join(f"option name {name} type spin default 3 min 1 max 10"
                           for name in ("Hash", "Threads", "Contempt", "SyzygyProbeDepth",
                                        "SyzygyProbeLimit", "LMR_BASE", "MAT_BASE")) + "\nuciok\n"
        with patch.object(spsa.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, output)):
            self.assertEqual([p[0] for p in spsa.discover_params("engine")], ["LMR_BASE", "MAT_BASE"])
            with patch.dict(os.environ, {"PARAMS": "LMR_,TYPO"}), self.assertRaises(ValueError):
                spsa.discover_params("engine")

    def test_resume_replays_uncommitted_trial(self):
        for error in (ValueError("missing games"), KeyboardInterrupt()):
            state = self.state()
            first, interrupted = [], []

            def record(target):
                def match(run, config, plus, minus, seed, log):
                    target.append((plus, minus, seed))
                    return 3, 2, 3
                return match

            with contextlib.redirect_stdout(io.StringIO()):
                spsa.run_iterations(self.run, copy.deepcopy(state), record(first))
            reference = json.loads((self.run / "checkpoint.json").read_text())
            spsa.write_json(self.run / "checkpoint.json", state)

            def fail_second(run, config, plus, minus, seed, log):
                interrupted.append((plus, minus, seed))
                if len(interrupted) == 2:
                    raise error
                return 3, 2, 3

            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(type(error)):
                spsa.run_iterations(self.run, state, fail_second)
            saved = spsa.load_state(self.run)
            self.assertEqual(saved["iteration"], 1)
            self.assertTrue(any(v != round(v) for v in saved["theta"].values()))
            resumed = []
            with contextlib.redirect_stdout(io.StringIO()):
                spsa.run_iterations(self.run, saved, record(resumed))
            actual = spsa.load_state(self.run)
            self.assertEqual(interrupted[1], resumed[0])
            self.assertEqual(first, interrupted[:1] + resumed)
            for key in ("iteration", "theta", "rng", "config"):
                self.assertEqual(actual[key], reference[key])

    def test_failed_batch_does_not_write_checkpoint(self):
        state = self.state()
        before = (self.run / "checkpoint.json").read_bytes()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            spsa.run_iterations(self.run, state, lambda *args: (0, 0, 0))
        self.assertEqual(before, (self.run / "checkpoint.json").read_bytes())

    def test_failed_checkpoint_replace_keeps_old_state(self):
        state = self.state()
        path = self.run / "checkpoint.json"
        before = path.read_bytes()
        with patch.object(spsa.os, "replace", side_effect=OSError("disk failure")), self.assertRaises(OSError):
            spsa.write_json(path, {**state, "iteration": 1})
        self.assertEqual(path.read_bytes(), before)

    def test_resume_guards(self):
        self.state()
        with patch.dict(os.environ, {"ITERS": "10"}), self.assertRaises(ValueError):
            spsa.load_state(self.run)
        with patch.dict(os.environ, {"PARAMS": "LMR_"}), self.assertRaises(ValueError):
            spsa.load_state(self.run)
        (self.run / "engine").write_text("modified")
        with self.assertRaises(ValueError):
            spsa.load_state(self.run)

    def test_config_and_legacy_init(self):
        state = self.state()
        for key, value in (("GAMES", 3), ("ITERS", 0), ("STEP_FRAC", float("nan")), ("MATCH_TIMEOUT", -1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                spsa.validate_config({**state["config"], key: value})
        old = self.run / "old.tuned"
        old.write_text("LMR_BASE = 81\n")
        self.assertEqual(spsa.initial_values(state["params"], old)["LMR_BASE"], 81.0)
        with patch.dict(os.environ, {"RESUME": str(old)}), self.assertRaises(ValueError):
            spsa.main()

    def test_exclusive_run_lock(self):
        self.state()
        with (self.run / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = subprocess.run([sys.executable, "-B", str(Path(spsa.__file__))],
                                    env={"RESUME": str(self.run)}, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already active", result.stderr)

    def test_timeout_kills_process_group(self):
        state = self.state()
        process = unittest.mock.Mock(pid=123456)
        process.wait.side_effect = [subprocess.TimeoutExpired("fastchess", 1), 0, 0]
        with patch.object(spsa.subprocess, "Popen", return_value=process), \
             patch.object(spsa.os, "killpg") as kill, self.assertRaisesRegex(ValueError, "MATCH_TIMEOUT"):
            spsa.play_match(self.run, state["config"], {}, {}, 123, self.run / "match.log")
        self.assertEqual(kill.call_count, 2)


if __name__ == "__main__":
    unittest.main()
