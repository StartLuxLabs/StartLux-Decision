"""Reading model folders: kinds, AutoThink defaults and the auto_think block, and the checks."""
from __future__ import annotations

import shutil
import tempfile
import unittest

from startlux_decide.backends.base import Sampling
from startlux_decide.modelcfg import AutoThink, load_model_folder, think_mode

from .folders import make_folder


class FolderTest(unittest.TestCase):
    def setUp(self):
        self.paths = []

    def tearDown(self):
        for p in self.paths:
            shutil.rmtree(p, ignore_errors=True)

    def folder(self, *args, **kwargs):
        self.paths.append(make_folder(*args, **kwargs))
        return self.paths[-1]

    def test_release_defaults(self):
        f = load_model_folder(self.folder())
        self.assertEqual(f.kind, "auto_think")
        a = f.auto_think
        self.assertEqual((a.think_mode, a.gate, a.mix, a.budget, a.tail_chars, a.chunk_tokens),
                         ("auto", 0.6, 0.75, 4096, 4000, 1024))
        self.assertEqual(a.prefix, "Let me work through this carefully, step by step.\n")
        self.assertEqual(dict(a.think_temperature), {"choice": 1.0, "noul": 1.0, "score": 1.0})
        self.assertEqual(a.sampling, Sampling(0.6, 0.95, 20))
        self.assertEqual(f.temperature("noul"), 1.376)
        self.assertEqual((f.wide_group, f.wide_keep, f.wide_residual), (25, 3, 0.001))

    def test_instant(self):
        f = load_model_folder(self.folder("instant"))
        self.assertEqual((f.kind, f.auto_think, f.bidirectional), ("instant", None, True))
        with self.assertRaisesRegex(ValueError, "does not think"):
            load_model_folder(self.folder("instant", {"auto_think": {"gate": 0.5}}))

    def test_letter_bias(self):
        self.assertEqual(load_model_folder(self.folder()).letter_bias, ())
        self.assertEqual(load_model_folder(self.folder(extra={"letter_bias": [0] * 26})).letter_bias, ())
        bias = [i / 10 for i in range(26)]
        self.assertEqual(load_model_folder(self.folder(extra={"letter_bias": bias})).letter_bias, tuple(bias))
        for bad in (None, {}, "0", [], [0] * 25, [0] * 27, [True] * 26, ["0"] * 26,
                    [float("nan")] * 26, [float("inf")] * 26):
            with self.subTest(bias=bad), self.assertRaisesRegex(ValueError, "letter_bias"):
                load_model_folder(self.folder(extra={"letter_bias": bad}))

    def test_auto_think_block(self):
        f = load_model_folder(self.folder(extra={"auto_think": {
            "think_mode": "plain", "gate": 0.8, "mix": 0.5, "budget": 2048, "prefix": "", "tail_chars": 0,
            "chunk_tokens": 512, "think_temperature": {"choice": 1.2}, "sampling": {"top_k": 40}}}))
        self.assertEqual(f.auto_think, AutoThink(think_mode="never", gate=0.8, mix=0.5, budget=2048, prefix="",
                                                 tail_chars=0, chunk_tokens=512, think_temperature={"choice": 1.2},
                                                 sampling=Sampling(0.6, 0.95, 40)))
        for bad in ({"gate": 1.5}, {"budget": 0}, {"budget": 1.5}, {"gates": 0.5}, {"think_mode": "sometimes"},
                    {"think_temperature": {"rank": 1.0}}, {"sampling": {"temperature": 0}}, {"mix": True},
                    {"sampling": {"top_p": "0.9"}}, {"sampling": {"top_k": 2.5}}, {"sampling": {"temperature": True}}):
            with self.assertRaises(ValueError, msg=str(bad)):
                load_model_folder(self.folder(extra={"auto_think": bad}))

    def test_checks(self):
        with self.assertRaisesRegex(ValueError, "differs|but"):
            load_model_folder(self.folder(extra={"prompt_attention": "bidirectional"}))
        with self.assertRaisesRegex(ValueError, "max_options_per_pass"):
            load_model_folder(self.folder(extra={"max_options_per_pass": 32}))
        with self.assertRaisesRegex(ValueError, "temperature_by_type"):
            load_model_folder(self.folder(extra={"temperature_by_type": {"choice": 0}}))
        with self.assertRaisesRegex(ValueError, "wide_choice"):
            load_model_folder(self.folder(extra={"wide_choice": {"group": 30}}))
        with self.assertRaisesRegex(ValueError, "jev_prompt_attention"):
            load_model_folder(self.folder(config={"jev_prompt_attention": "sideways"}))
        empty = tempfile.mkdtemp()
        self.paths.append(empty)
        with self.assertRaises(FileNotFoundError):
            load_model_folder(empty)

    def test_release_date(self):
        dated = load_model_folder(self.folder(extra={"release_date": "2026-10-20"}))
        self.assertEqual(dated.release_date, "2026-10-20")
        self.assertRegex(load_model_folder(self.folder()).release_date, r"^\d{4}-\d\d-\d\d$")

    def test_think_mode_names(self):
        self.assertEqual([think_mode(m) for m in ("auto", "never", "always", "hybrid", "plain", "think")],
                         ["auto", "never", "always", "auto", "never", "always"])
        for bad in ("Auto", "", None, 1):
            with self.assertRaises(ValueError):
                think_mode(bad)


if __name__ == "__main__":
    unittest.main()
