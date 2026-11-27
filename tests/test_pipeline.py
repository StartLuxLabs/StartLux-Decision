"""The pipeline on the fake engine: details and timings, Instant models, knock-out lists, concurrency, errors."""
from __future__ import annotations

import asyncio
import math
import random
import string
import time
import unittest
from dataclasses import replace

from startlux_decide.backends.base import BackendRequestError
from startlux_decide.backends.fake import FakeBackend
from startlux_decide.controls import Controls, RequestError
from startlux_decide.modelcfg import AutoThink, ModelFolder
from startlux_decide.pipeline import CUT, Decider, default_controls

AUTO = ModelFolder(path="(test)", prompt_attention="causal", temperature_by_type={"choice": 1.0, "noul": 1.2},
                   letter_token_ids=tuple(range(65, 91)), wide_group=25, wide_keep=3, wide_residual=0.001,
                   auto_think=AutoThink(), release_date="2026-10-11")
INSTANT = replace(AUTO, prompt_attention="bidirectional", auto_think=None)
QUESTIONS = {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments",
                                                                                  "shipping": "Delivery"}},
             "urgent": {"type": "noul", "instructions": "Answer today?"},
             "severity": {"type": "score", "instructions": "How bad?", "criteria": ["minor", "major", "critical"]}}


def closing(prompt, seed, max_tokens):
    """A thought that closes after a few words."""
    return ["Billing ", "handles ", "double ", "charges.\n", "</think>", "ignored"], False


def endless(prompt, seed, max_tokens):
    """A thought of random letter words: never closed, never a loop."""
    r = random.Random(seed)
    return ["".join(r.choice(string.ascii_lowercase) for _ in range(5)) + " " for _ in range(max_tokens)], False


def run(coro):
    return asyncio.run(coro)


class DetailsTest(unittest.TestCase):
    def test_thinking_question(self):
        backend = FakeBackend(script=closing)
        d = Decider(AUTO, backend)
        out = run(d.decide("charged twice", QUESTIONS, Controls(think_mode="always", details=True, thought_text=True)))
        self.assertEqual(set(out.timing_ms), {"wall", "first_pass_sum", "thought_span", "after_thought_sum"})
        for key, ans in out.answers.items():
            det = ans["details"]
            self.assertEqual(set(det), {"first_pass", "thought", "after_thought", "gate", "mix", "total_ms"})
            self.assertTrue(det["thought"]["used"])
            self.assertEqual(det["thought"]["ended"], "closed")
            self.assertEqual(det["thought"]["tokens"], 5)                 # four words and the stop string
            self.assertEqual(det["thought"]["text"],
                             "Let me work through this carefully, step by step.\nBilling handles double charges.")
            self.assertIsNone(det["gate"])                                # always: the gate was not consulted
            self.assertEqual(det["mix"], 0.75)
            self.assertEqual(set(det["after_thought"]), {"probabilities", "ms"})
            self.assertGreaterEqual(det["total_ms"], det["first_pass"]["ms"])
        self.assertEqual(out.usage["output_tokens"], 15)
        reads = [c[1] for c in backend.calls if c[0] == "readout" and "</think>\n\n" in c[1] and "Billing" in c[1]]
        self.assertEqual(len(reads), 3)
        self.assertTrue(all(r.endswith("Billing handles double charges.\n</think>\n\n") for r in reads))

    def test_cut_thought_and_tail(self):
        backend = FakeBackend(script=endless)
        d = Decider(replace(AUTO, auto_think=AutoThink(budget=300, chunk_tokens=100, tail_chars=200)), backend)
        q = {"q": QUESTIONS["urgent"]}
        out = run(d.decide("x", q, Controls(think_mode="always", budget=300, details=True)))
        det = out.answers["q"]["details"]
        self.assertEqual((det["thought"]["ended"], det["thought"]["tokens"]), ("budget", 300))
        self.assertNotIn("text", det["thought"])
        gens = [c for c in backend.calls if c[0] == "generate"]
        self.assertEqual([c[2] for c in gens], [100, 100, 100])                # chunks of 100 tokens
        read = [c[1] for c in backend.calls if c[0] == "readout"][-1]
        self.assertTrue(read.endswith(CUT))
        self.assertIn("<think>\n[...]\n", read)
        self.assertEqual(len(read.split("<think>\n[...]\n")[1]) - len(CUT), 200)

    def test_gate_and_never(self):
        d = Decider(AUTO, FakeBackend(script=closing))
        never = run(d.decide("x", QUESTIONS, Controls(think_mode="never", details=True)))
        for ans in never.answers.values():
            self.assertEqual(ans["details"]["thought"], {"used": False, "tokens": 0, "ended": "skipped", "ms": 0.0})
            self.assertIsNone(ans["details"]["after_thought"])
            self.assertIsNone(ans["details"]["mix"])
        self.assertEqual(never.timing_ms["thought_span"], 0.0)
        for gate in (0.0, 1.0):
            out = run(d.decide("x", QUESTIONS, Controls(think_mode="auto", gate=gate, details=True)))
            for ans in out.answers.values():
                self.assertEqual(ans["details"]["gate"], gate)
                self.assertEqual(ans["details"]["thought"]["used"], gate == 1.0)
        plain = run(d.decide("x", QUESTIONS))
        self.assertNotIn("details", plain.answers["team"])
        self.assertEqual(set(plain.timing_ms), {"wall", "first_pass_sum", "thought_span", "after_thought_sum"})

    def test_no_room_no_thought(self):
        d = Decider(AUTO, FakeBackend(script=closing), context_length=300)
        out = run(d.decide("x" * 200, {"q": QUESTIONS["urgent"]}, Controls(think_mode="always", details=True)))
        self.assertFalse(out.answers["q"]["details"]["thought"]["used"])


class InstantTest(unittest.TestCase):
    def test_one_pass_only(self):
        backend = FakeBackend(script=closing)
        d = Decider(INSTANT, backend)
        self.assertEqual(d.defaults.think_mode, "never")
        out = run(d.decide("x", QUESTIONS, d.controls({"return_details": True})))
        self.assertFalse(any(c[0] == "generate" for c in backend.calls))
        self.assertEqual(len(backend.calls), 3)
        with self.assertRaises(RequestError):
            d.controls({"think_mode": "always"})
        with self.assertRaises(RequestError):
            run(d.decide("x", QUESTIONS, Controls(think_mode="auto")))
        self.assertEqual(out.usage["output_tokens"], 0)

    def test_default_controls(self):
        self.assertEqual(default_controls(INSTANT), Controls(think_mode="never", gate=0.6, budget=4096, mix=0.75))
        self.assertEqual(default_controls(AUTO, gate=0.8, mix=None), Controls(gate=0.8))


class KnockoutTest(unittest.TestCase):
    def test_wide_list(self):
        crit = {f"opt{i}": f"option number {i}" for i in range(60)}
        backend = FakeBackend()
        d = Decider(AUTO, backend)
        out = run(d.decide("x", {"q": {"type": "choice", "instructions": "Which?", "criteria": crit}},
                           Controls(think_mode="never", details=True)))
        ans = out.answers["q"]
        self.assertEqual(list(ans["probabilities"])[9:], [k for k in crit if k not in list(ans["probabilities"])[:9]])
        self.assertEqual(set(ans["probabilities"]), set(crit))
        self.assertAlmostEqual(sum(ans["probabilities"].values()), 1.0)
        self.assertEqual(set(ans["details"]["first_pass"]["probabilities"]), set(crit))
        self.assertEqual(len(backend.calls), 4)                          # three groups of 20, then the final round
        self.assertEqual([len(c[2]) for c in backend.calls], [20, 20, 20, 9])

    def test_single_option(self):
        out = run(Decider(AUTO, FakeBackend()).decide("x", {"q": {"type": "choice", "criteria": {"only": None}}},
                                                     Controls(details=True)))
        self.assertEqual(out.answers["q"]["choice"], "only")
        self.assertFalse(out.answers["q"]["details"]["thought"]["used"])


class ErrorsAndConcurrencyTest(unittest.TestCase):
    def test_bad_question_is_reported_with_its_key(self):
        d = Decider(AUTO, FakeBackend())
        with self.assertRaises(RequestError) as cm:
            run(d.decide("x", {"ok": QUESTIONS["urgent"], "bad": {"type": "score", "instructions": "q",
                                                                  "criteria": list("abcdefghijk")}}))
        self.assertEqual(cm.exception.loc, ["body", "questions", "bad"])
        for questions in (None, {}, [], {"q": "text"}):
            with self.assertRaises(RequestError):
                run(d.decide("x", questions))

    def test_engine_refusal_becomes_a_request_error(self):
        class Refusing(FakeBackend):
            async def readout(self, prompt, letter_ids):
                raise BackendRequestError("prompt too long")
        with self.assertRaises(RequestError) as cm:
            run(Decider(AUTO, Refusing()).decide("x", {"q": QUESTIONS["urgent"]}))
        self.assertIn("prompt too long", str(cm.exception))

    def test_questions_run_concurrently(self):
        d = Decider(AUTO, FakeBackend(delay=0.2))
        t0 = time.perf_counter()
        run(d.decide("x", {f"q{i}": QUESTIONS["urgent"] for i in range(10)}, Controls(think_mode="never")))
        self.assertLess(time.perf_counter() - t0, 1.0)                  # ten 0.2 s read-outs at once


class LetterBiasTest(unittest.TestCase):
    def test_bias_precedes_temperature_for_all_question_types(self):
        class EqualLogits(FakeBackend):
            async def readout(self, prompt, letter_ids):
                return [0.0] * len(letter_ids)

        folder = replace(AUTO, temperature_by_type={t: 2.0 for t in ("choice", "noul", "score")},
                         letter_bias=(0.0, 2 * math.log(2), 2 * math.log(4)) + (0.0,) * 23)
        out = run(Decider(folder, EqualLogits()).decide("x", QUESTIONS, Controls(think_mode="never")))
        self.assertAlmostEqual(out.answers["team"]["probabilities"]["billing"], 1 / 3)
        self.assertAlmostEqual(out.answers["team"]["probabilities"]["shipping"], 2 / 3)
        self.assertAlmostEqual(out.answers["urgent"]["noul"], 1 / 3)
        for i, expected in enumerate((1 / 7, 2 / 7, 4 / 7)):
            self.assertAlmostEqual(out.answers["severity"]["probabilities"][str(i)], expected)

    def test_bias_is_used_by_the_autothink_gate_and_second_readout(self):
        class EqualLogits(FakeBackend):
            async def readout(self, prompt, letter_ids):
                return [0.0] * len(letter_ids)

        folder = replace(AUTO, temperature_by_type={"choice": 1.0},
                         letter_bias=(0.0, math.log(99)) + (0.0,) * 24)
        backend = EqualLogits(script=closing)
        decider = Decider(folder, backend)
        question = {"q": QUESTIONS["team"]}
        automatic = run(decider.decide("x", question, Controls(think_mode="auto", details=True)))
        self.assertFalse(automatic.answers["q"]["details"]["thought"]["used"])
        forced = run(decider.decide("x", question, Controls(think_mode="always", details=True)))
        self.assertTrue(forced.answers["q"]["details"]["thought"]["used"])
        for stage in ("first_pass", "after_thought"):
            self.assertAlmostEqual(forced.answers["q"]["details"][stage]["probabilities"]["shipping"], 0.99)


if __name__ == "__main__":
    unittest.main()
