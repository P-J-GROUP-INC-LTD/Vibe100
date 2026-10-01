"""tools/test_setup_numa.py - setup.py's --numa setting (Vibe100 WP-F): carried by a config, never passed to an engine that does
not have the option (the ready-made upstream engine exits "unknown argument").  No GPU, no network.

    python3 tools/test_setup_numa.py

(`--setup` run again and a new folder adopting the install: tools/test_setup_main.py.)
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import setup  # noqa: E402


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **kw), buf.getvalue()


class Numa(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def engine(self, text):
        exe = self.tmp / "strata"
        exe.write_bytes(text)
        return str(exe)

    OURS = b"\x7fELF...strata generate: unknown argument: %s\n\0--numa\0--shared-experts\0"
    UPSTREAM = b"\x7fELF...strata generate: unknown argument: %s\n\0--shared-experts\0--no-numa-here\0"

    def test_engine_has_option(self):
        self.assertTrue(setup.engine_has_option(self.engine(self.OURS), "--numa"))
        self.assertFalse(setup.engine_has_option(self.engine(self.UPSTREAM), "--numa"))
        self.assertFalse(setup.engine_has_option(self.tmp / "nothing", "--numa"))
        self.assertFalse(setup.engine_has_option(self.engine(b""), "--numa"))          # (an empty file cannot be mapped)
        self.assertFalse(setup.engine_has_option("", "--numa"))

    def test_off_and_mirror_reach_an_engine_that_has_the_option(self):
        for mode in ("off", "mirror"):
            cfg = {"exe": self.engine(self.OURS), "args": ["--max-context", "32768"], "numa": mode}
            r, out = quiet(setup.apply_numa, cfg)
            self.assertTrue(r)
            self.assertEqual(cfg["args"], ["--max-context", "32768", "--numa", mode])
            self.assertEqual(out, "")

    def test_the_ready_made_engine_is_not_given_it(self):
        cfg = {"exe": self.engine(self.UPSTREAM), "args": ["--max-context", "32768"], "numa": "mirror"}
        r, out = quiet(setup.apply_numa, cfg)
        self.assertFalse(r)
        self.assertEqual(cfg["args"], ["--max-context", "32768"])
        self.assertEqual(cfg["numa"], "mirror")                      # still saved, for an engine that has it
        self.assertIn("has no such option", out)
        self.assertIn("--setup --build", out)

    def test_a_leftover_numa_argument_is_removed_for_an_engine_without_it(self):
        cfg = {"exe": self.engine(self.UPSTREAM), "args": ["--numa", "off", "--max-context", "32768"], "numa": "off"}
        r, out = quiet(setup.apply_numa, cfg)
        self.assertTrue(r)
        self.assertEqual(cfg["args"], ["--max-context", "32768"])

    def test_auto_passes_nothing_and_needs_no_engine_check(self):
        cfg = {"exe": self.engine(self.UPSTREAM), "args": ["--numa", "off", "--max-context", "32768"], "numa": "auto"}
        r, out = quiet(setup.apply_numa, cfg)
        self.assertTrue(r)
        self.assertEqual(cfg["args"], ["--max-context", "32768"])
        self.assertEqual(out, "")

    def test_a_config_without_the_setting_is_left_alone(self):
        cfg = {"exe": self.engine(self.UPSTREAM), "args": ["--numa", "off"]}       # written by hand
        self.assertFalse(setup.apply_numa(cfg))
        self.assertEqual(cfg["args"], ["--numa", "off"])

    def test_the_engine_is_found_again_after_it_is_compiled(self):
        cfg = {"exe": self.engine(self.UPSTREAM), "args": ["--x"], "numa": "off"}
        quiet(setup.apply_numa, cfg)
        self.assertEqual(cfg["args"], ["--x"])
        Path(cfg["exe"]).write_bytes(self.OURS)                       # --build later
        self.assertTrue(quiet(setup.apply_numa, cfg)[0])
        self.assertEqual(cfg["args"], ["--x", "--numa", "off"])

    def test_choices_from_config_carries_it(self):
        for numa in ("off", "mirror", "auto", None):
            cfg = {"exe": "e", "args": ["--max-context", "65536", "--kv", "int8"], "port": 8080, "model_name": "m"}
            if numa:
                cfg["numa"] = numa
            path = self.tmp / "strata-q2_0.json"
            path.write_text(json.dumps(cfg))
            self.assertEqual(setup.choices_from_config(path)["numa"], numa)

    def test_saved_numa(self):
        path = self.tmp / "strata-q2_0.json"
        self.assertIsNone(setup.saved_numa(path, None))                # nothing saved, nothing asked
        self.assertEqual(setup.saved_numa(path, "mirror"), "mirror")
        path.write_text(json.dumps({"numa": "off"}))
        self.assertEqual(setup.saved_numa(path, None), "off")          # a `--setup` run again keeps it
        self.assertEqual(setup.saved_numa(path, "auto"), "auto")       # what is asked wins
        path.write_text("not json")
        self.assertIsNone(setup.saved_numa(path, None))


if __name__ == "__main__":
    unittest.main(verbosity=1)
