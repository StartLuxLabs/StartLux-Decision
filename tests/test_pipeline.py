"""The pipeline on the fake engine: details and timings, the first read's prompt, Instant models, knock-out lists,
concurrency, errors."""
from __future__ import annotations

import asyncio
import random
import string
import time
import unittest
from dataclasses import replace

from startlux_decide.backends.base import BackendRequestError
from startlux_decide.backends.fake import FakeBackend, FakeTokenizer
from startlux_decide.controls import Controls, RequestError
from startlux_decide.modelcfg import AutoThink, ModelFolder
from startlux_decide.pipeline import CUT, SHARE_MIN, Decider, default_controls
from startlux_decide.prompt import THINK_OFF_SUFFIX, count_tokens

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

    def test_every_call_fits_the_context_to_its_end(self):
        # A thought cut by the room the context leaves it gives the longest read-out after a thought. With NOTE the
        # thinking prompt is longer than the one-pass prompt, as with the release template.
        def letters(prompt, seed, max_tokens):          # one fake token each, never closed, never a loop
            r = random.Random(seed)
            return [r.choice(string.ascii_lowercase) for _ in range(max_tokens)], False

        tok, context = FakeTokenizer(think_note=NOTE), 2000
        for first_read in ("plain", "thinking"):
            ctl = Controls(think_mode="always", first_read=first_read, details=True)
            cut = []
            for n in range(1350, 1680, 3):
                backend = FakeBackend(tokenizer=tok, script=letters)
                out = run(Decider(AUTO, backend, context_length=context).decide("x" * n, {"q": QUESTIONS["urgent"]},
                                                                                ctl))
                thought = out.answers["q"]["details"]["thought"]
                cut.append(thought["used"] and thought["tokens"] < ctl.budget)
                for call in backend.calls:
                    generated = call[2] if call[0] == "generate" else 1
                    self.assertLessEqual(count_tokens(tok, call[1]) + generated, context, (first_read, n, call[0]))
            self.assertGreater(sum(cut), 50, first_read)        # thoughts cut short by the context were exercised
            self.assertFalse(cut[-1], first_read)               # and the last state leaves no room for one


NOTE = "Reasoning effort is set to high.\n\n"          # what the release template starts a thinking system line with


class FirstReadTest(unittest.TestCase):
    """first_read "thinking": the one-pass prompts of a request that may think are thinking prompts with an empty
    thought, so each thought continues its question's read-out, and a long state is read once with the thoughts."""

    def calls(self, ctl, state="charged twice", questions=QUESTIONS, **engine):
        backend = FakeBackend(tokenizer=FakeTokenizer(think_note=NOTE), script=closing, **engine)
        run(Decider(AUTO, backend).decide(state, questions, ctl))
        firsts = [c[1] for c in backend.calls if c[0] == "readout" and "Billing" not in c[1]]   # not after a thought
        thoughts = [c[1] for c in backend.calls if c[0] == "generate"]
        return firsts, thoughts, [c[1] for c in backend.calls if c[0] == "prefill"]

    def test_thoughts_continue_the_read_outs(self):
        firsts, thoughts, _ = self.calls(Controls(think_mode="always", first_read="thinking"))
        self.assertEqual((len(firsts), len(thoughts)), (3, 3))
        for read in firsts:
            self.assertTrue(read.endswith(THINK_OFF_SUFFIX))
            self.assertEqual(sum(t.startswith(read[:-len("\n</think>\n\n")]) for t in thoughts), 1)

    def test_plain_and_never_read_the_trained_prompt(self):
        firsts, thoughts, _ = self.calls(Controls(think_mode="always"))
        self.assertEqual((len(firsts), len(thoughts)), (3, 3))
        self.assertTrue(all(NOTE not in r for r in firsts) and all(NOTE in t for t in thoughts))
        firsts, thoughts, _ = self.calls(Controls(think_mode="never", first_read="thinking"))
        self.assertEqual((len(firsts), len(thoughts)), (3, 0))
        self.assertTrue(all(NOTE not in r and r.endswith(THINK_OFF_SUFFIX) for r in firsts))

    def test_a_long_state_is_read_once_with_its_thoughts(self):
        state = "x" * SHARE_MIN                         # one fake token per character
        firsts, thoughts, prefills = self.calls(Controls(think_mode="always", first_read="thinking"), state,
                                                caches_prefixes=True)
        self.assertEqual(len(prefills), 1)
        self.assertIn(state, prefills[0])
        self.assertTrue(all(p.startswith(prefills[0]) for p in firsts + thoughts))
        _, thoughts, prefills = self.calls(Controls(think_mode="always"), state, caches_prefixes=True)
        self.assertEqual(len(prefills), 1)
        self.assertFalse(any(t.startswith(prefills[0][:len("<|im_start|>system\n") + 1]) for t in thoughts))

    def test_knock_out_rounds_take_the_same_form(self):
        crit = {f"opt{i}": f"option number {i}" for i in range(60)}
        wide = {"q": {"type": "choice", "instructions": "Which?", "criteria": crit}}
        firsts, thoughts, _ = self.calls(Controls(think_mode="always", first_read="thinking"), questions=wide)
        self.assertEqual((len(firsts), len(thoughts)), (4, 1))         # three groups of 20, then the final round
        self.assertTrue(all(NOTE in r and r.endswith(THINK_OFF_SUFFIX) for r in firsts))
        self.assertTrue(thoughts[0].startswith(firsts[-1][:-len("\n</think>\n\n")]))

    def test_a_template_without_the_form_is_refused_at_startup(self):
        class Unopened(FakeTokenizer):          # ends a thinking prompt with "<think>" and no newline
            def apply_chat_template(self, *args, enable_thinking=True, **kwargs):
                text = super().apply_chat_template(*args, enable_thinking=enable_thinking, **kwargs)
                return text if enable_thinking is False else text.removesuffix("\n")
        with self.assertRaisesRegex(ValueError, "opened think block"):
            Decider(AUTO, FakeBackend(tokenizer=Unopened()))
        Decider(INSTANT, FakeBackend(tokenizer=Unopened()))     # an Instant folder renders no thinking prompt


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


if __name__ == "__main__":
    unittest.main()
