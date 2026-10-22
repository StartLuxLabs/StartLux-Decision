"""Golden test: the package's pipeline (Decider + VLLMBackend) against the release proxy's per-question pipeline
(reference_impl.Proxy.decide_one), both talking to the same deterministic fake vLLM, on many random requests and
settings. Required: identical requests to vLLM (bodies, including every thought chunk, seed and read-out prompt) and
identical probabilities, choices, scores and thought token counts, bit for bit. The deliberate differences are checked
on their own: confidence formulas, the score legend order, usage.input_tokens of knock-out lists, mix 0 and
single-option choice lists. The room of a thought near the end of the context differs too; package_room gives the
proxy a shorter context for it, and test_pipeline / test_real_tokenizer check that every call fits the context.
"""
from __future__ import annotations

import asyncio
import collections
import os
import random
import unittest
from types import SimpleNamespace

from startlux_decide.backends.fake import FakeTokenizer
from startlux_decide.backends.vllm import VLLMBackend
from startlux_decide.controls import Controls
from startlux_decide.modelcfg import RELEASE_PREFIX, AutoThink, ModelFolder
from startlux_decide.pipeline import Decider
from startlux_decide.prompt import count_tokens, plain_prompt, score_keys
from startlux_decide.readout import choice_confidence, knockout_groups, score_confidence

from . import reference_impl as ref
from .fake_vllm import FakeVLLM

CASES = 1500
MODES = {"hybrid": "auto", "plain": "never", "think": "always"}
NAMES = ("billing", "shipping", "technical", "refund", "escalate", "ignore", "wait", "approve", "reject", "defer",
         "red", "green", "blue", "north", "south", "east", "west", "buy", "sell", "hold")


def random_text(r: random.Random, n: int) -> str:
    pieces = ["order", "#4411", "charged twice", "naïve", "日本語", "😀", "line\nbreak", "  spaced  ",
              "{\"a\": 1}", "A) trap", "<b>bold</b>", "42", "3.14", "tab\there", "Ünïcödé", "end."]
    return " ".join(r.choice(pieces) for _ in range(n))


def random_state(r: random.Random):
    x = r.random()
    if x < 0.3:
        return random_text(r, r.randint(1, 60))
    if x < 0.4:
        return r.choice([None, {}, [], "", "   "])
    if x < 0.55:
        return [r.randint(0, 99) for _ in range(r.randint(1, 20))]
    return {"ticket": random_text(r, r.randint(1, 30)), "items": [{"sku": f"s{i}", "qty": r.randint(1, 5)}
                                                                 for i in range(r.randint(0, 12))],
            "flags": [r.random() < 0.5 for _ in range(r.randint(0, 9))], "note": None}


def random_ids(r: random.Random, n: int) -> list[str]:
    style = r.random()
    if style < 0.15 and n <= 26:
        return [r.choice(["", "(", ""]) + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i] + r.choice(["", ")", "."]) for i in range(n)]
    if style < 0.25:
        return [str(i + 1) for i in range(n)]
    if style < 0.3:
        return [f"option {i + 1}" for i in range(n)]
    names = list(NAMES)
    r.shuffle(names)
    return [names[i] if i < len(names) else f"{r.choice(NAMES)}_{i}" for i in range(n)]


def random_question(r: random.Random) -> dict:
    t = r.choice(["choice"] * 5 + ["noul"] * 2 + ["score"] * 2 + ["bool"])
    ins = random_text(r, r.randint(1, 8)) + "?"
    if t == "choice":
        n = r.choice([2, 2, 3, 4, 5, 8, 12, 26, 2, 3, 27, 40, 75, 120, 230])
        ids = random_ids(r, n)
        if n <= 26 and r.random() < 0.2:
            return {"type": "choice", "instructions": ins, "criteria": ids}
        desc = lambda i: r.choice([None, random_text(r, r.randint(1, 5)), i, i.upper(), ""])  # noqa: E731
        crit = {i: desc(i) for i in ids}
        if r.random() < 0.3:
            crit = {i: random_text(r, 2) for i in ids}                     # every option described
        return {"type": "choice", "instructions": ins, "criteria": crit}
    if t in ("noul", "bool"):
        q = {"type": t}
        if r.random() < 0.6:
            q["instructions"] = ins
        if r.random() < 0.4:
            q["criteria"] = {"true": random_text(r, 2), "false": r.choice([None, random_text(r, 2)])}
        return q
    n = r.randint(2, 10)
    levels = [random_text(r, r.randint(1, 3)) for _ in range(n)]
    if r.random() < 0.3:
        keys = [str(i) for i in range(n)]
        r.shuffle(keys)
        return {"type": "score", "instructions": ins, "criteria": {k: levels[int(k)] for k in keys}}
    return {"type": "score", "instructions": ins, "criteria": levels}


def random_settings(r: random.Random) -> SimpleNamespace:
    chunk = r.choice([0, 100, 1024, 1024, 1024, 1])
    return SimpleNamespace(
        mode=r.choice(["hybrid"] * 6 + ["plain", "think"]), tau=r.choice([0.5, 0.9, 0.9, 0.99, 1.0]),
        mix=r.choice([0.25, 0.5, 0.75, 0.75, 1.0]), budget=r.choice([1, 33, 64]) if chunk == 1 else r.choice(
            [1, 33, 64, 700, 4096, 4096]), loop_chunk=chunk, tail_chars=r.choice([0, 50, 300, 4000, 4000]),
        prefix=r.choice(["", RELEASE_PREFIX, RELEASE_PREFIX, "Think.\n"]),
        max_len=r.choice([65536, 65536, 4000, 2500, 1200]),
        temp={t: round(r.uniform(0.6, 1.9), 4) for t in ("choice", "noul", "score") if r.random() < 0.9},
        tthink=r.choice([{"choice": 1.0, "noul": 1.0, "score": 1.0}, {"choice": 0.8, "noul": 1.3, "score": 1.1}, {},
                         {"choice": 1.0}]),
        open_think=r.random() < 0.85, drop=r.choice([0.0, 0.0, 0.3]))


def folder_for(s: SimpleNamespace) -> ModelFolder:
    auto = AutoThink(think_mode=MODES[s.mode], gate=s.tau, mix=s.mix, budget=s.budget, prefix=s.prefix,
                     tail_chars=s.tail_chars, chunk_tokens=s.loop_chunk, think_temperature=s.tthink)
    return ModelFolder(path="(test)", prompt_attention="causal", temperature_by_type=s.temp, letter_token_ids=None,
                       wide_group=25, wide_keep=3, wide_residual=1e-3, auto_think=auto, release_date="2026-10-11")


def package_room(proxy: ref.Proxy, tok) -> None:
    """Deliberate difference: the package counts a thought's room from the thought's prompt, the proxy from the
    one-pass prompt. The two differ by the same number of tokens for every question, so the proxy gets the package's
    room with a context that much shorter (max_len enters nothing else in the proxy)."""
    plain, think = proxy.prompts(ref.messages(ref.from_systemone("s", {"type": "noul", "instructions": "q"}))[0])
    proxy.a.max_len -= count_tokens(tok, think) - count_tokens(tok, plain)


async def run_both(s: SimpleNamespace, state, questions: dict, tok=None):
    tok = tok or FakeTokenizer(open_think=s.open_think)
    fake_ref, fake_new = FakeVLLM(s.max_len, drop_rate=s.drop), FakeVLLM(s.max_len, drop_rate=s.drop)
    a = SimpleNamespace(mode=s.mode, tau=s.tau, mix=s.mix, budget=s.budget, prefix=s.prefix, tail_chars=s.tail_chars,
                        rep_pen=1.0, stop_loops=0, loop_chunk=s.loop_chunk, max_len=s.max_len, no_waves=True)
    proxy = ref.Proxy(a, tok, s.temp, s.tthink, ["http://127.0.0.1:8001/v1/completions"], fake_ref.complete)
    package_room(proxy, tok)
    expected = await proxy.handle(state, questions)
    backend = VLLMBackend(["http://127.0.0.1:8001"], tok, waves=False, transport=fake_new)
    await backend.start()
    probe_calls = len(fake_new.log)
    decider = Decider(folder_for(s), backend)
    controls = Controls(think_mode=MODES[s.mode], gate=s.tau, budget=s.budget, mix=s.mix, details=True,
                       thought_text=True)
    got = await decider.decide(state, questions, controls)
    await backend.close()
    return expected, got, fake_ref.log, fake_new.log[probe_calls:], tok


class GoldenTest(unittest.TestCase):
    def test_against_release_proxy(self):
        stats = self.run_cases(CASES, None)
        print("\ngolden test:", dict(sorted(stats.items())))
        for what in ("thought_closed", "thought_loop", "thought_budget", "wide", "wide_over_26_finalists"):
            self.assertGreater(stats[what], 100, what)

    @unittest.skipUnless(os.environ.get("STARTLUX_TEST_MODEL_DIR"), "set STARTLUX_TEST_MODEL_DIR for a real tokenizer")
    def test_against_release_proxy_real_tokenizer(self):
        from startlux_decide.modelcfg import load_tokenizer
        stats = self.run_cases(300, load_tokenizer(os.environ["STARTLUX_TEST_MODEL_DIR"]))
        print("\ngolden test, real tokenizer:", dict(sorted(stats.items())))
        self.assertGreater(stats["thought_closed"] + stats["thought_loop"] + stats["thought_budget"], 100)

    def run_cases(self, n: int, tok) -> collections.Counter:
        r = random.Random(20261011)
        stats = collections.Counter()
        for case in range(n):
            s = random_settings(r)
            state = random_state(r)
            questions = {f"q{i}": random_question(r) for i in range(r.choice([1, 1, 2, 3, 4]))}
            if tok is not None:                     # real token counts are ~4x smaller: tighter contexts
                s.max_len = r.choice([65536, 65536, 1200, 800, 450])
            with self.subTest(case=case):
                expected, got, log_ref, log_new, used_tok = asyncio.run(run_both(s, state, questions, tok))
                a, b = sorted(log_ref), sorted(log_new)
                if a != b:          # report the first difference only (a full diff of the logs is very slow)
                    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                    self.fail(f"requests to vLLM differ: {len(a)} vs {len(b)} calls, first difference at sorted "
                              f"position {first}:\n{a[first][:400] if first < len(a) else None}\n"
                              f"{b[first][:400] if first < len(b) else None}")
                self.compare(expected, got, state, questions, used_tok, stats)
                stats["requests"] += 1
                stats["vllm_calls"] += len(log_new)
        return stats

    def compare(self, expected: dict, got, state, questions: dict, tok, stats: collections.Counter) -> None:
        self.assertEqual(expected["usage"]["output_tokens"], got.usage["output_tokens"])
        group_tokens = 0
        for key, spec in questions.items():
            e, g = expected["answers"][key], got.answers[key]
            dbg, det = e["_debug"], g["details"]
            stats["questions"] += 1
            stats["type_" + g["type"]] += 1
            self.assertEqual(e["type"], g["type"])
            if g["type"] == "choice":
                self.assertEqual(e["choice"], g["choice"])
                self.assertEqual(list(e["probabilities"].items()), list(g["probabilities"].items()))
                self.assertEqual(g["confidence"], choice_confidence(list(g["probabilities"].values())))
            elif g["type"] == "noul":
                self.assertEqual(e["noul"], g["noul"])
            else:
                self.assertEqual(e["score"], g["score"])
                self.assertEqual(e["probabilities"], g["probabilities"])
                self.assertEqual(g["confidence"], score_confidence(list(g["probabilities"].values())))
                crit = spec["criteria"]
                levels = [crit[k] for k in score_keys(crit)] if isinstance(crit, dict) else list(crit)
                self.assertEqual(g["legend"], {str(i): levels[i] for i in range(len(levels))})
            self.assertEqual(det["first_pass"]["top"], max(dbg["p0"]))
            self.assertEqual(det["thought"]["used"], dbg["p1"] is not None)
            self.assertEqual(det["thought"]["tokens"], dbg["n_out"])
            crit = spec.get("criteria")
            wide = isinstance(crit, dict) and len(crit) > 26 and spec.get("type") == "choice"
            if wide:
                stats["wide"] += 1
                groups = knockout_groups(list(spec["criteria"]), 25)
                stats["wide_over_26_finalists"] += 3 * len(groups) > 26
                group_tokens += sum(count_tokens(tok, plain_prompt(tok, ref.messages(ref.from_systemone(
                    state, dict(spec, criteria={c: spec["criteria"][c] for c in grp})))[0])) for grp in groups)
            else:
                self.assertEqual(list(det["first_pass"]["probabilities"].values()), dbg["p0"])
                if dbg["p1"] is not None:
                    self.assertEqual(list(det["after_thought"]["probabilities"].values()), dbg["p1"])
            if det["thought"]["used"]:
                stats["thought_" + det["thought"]["ended"]] += 1
            else:
                self.assertIsNone(det["after_thought"])
        self.assertEqual(got.usage["input_tokens"], expected["usage"]["input_tokens"] + group_tokens)


class DeliberateDifferencesTest(unittest.TestCase):
    """What the package does differently from the release proxy, on purpose."""

    def setUp(self):
        self.s = SimpleNamespace(mode="think", tau=0.9, mix=0.0, budget=64, loop_chunk=1024, tail_chars=4000,
                                 prefix=RELEASE_PREFIX, max_len=65536, temp={"choice": 1.1, "noul": 1.3, "score": 1.0},
                                 tthink={"choice": 1.0, "noul": 1.0, "score": 1.0}, open_think=True, drop=0.0)

    def test_mix_zero_keeps_the_first_readout(self):
        q = {"q": {"type": "noul", "instructions": "Is it late?"}}
        expected, got, _, _, _ = asyncio.run(run_both(self.s, "the parcel came a week late", q))
        dbg, det = expected["answers"]["q"]["_debug"], got.answers["q"]["details"]
        self.assertIsNotNone(dbg["p1"])
        self.assertEqual(expected["answers"]["q"]["noul"], dbg["p1"][0])      # the proxy read 0 as "after only"
        self.assertEqual(got.answers["q"]["noul"], dbg["p0"][0])              # the package: weight 0 on the thought
        self.assertTrue(det["thought"]["used"])

    def test_single_option_choice_is_answered_without_the_model(self):
        self.s.mix = 0.75
        q = {"q": {"type": "choice", "instructions": "Which?", "criteria": {"only": "the one"}}}
        with self.assertRaises(ValueError):                                   # the proxy refused it
            asyncio.run(run_both(self.s, "x", q))
        tok = FakeTokenizer()

        async def go():
            backend = VLLMBackend(["8001"], tok, waves=False, transport=FakeVLLM())
            await backend.start()
            return await Decider(folder_for(self.s), backend).decide("x", q)
        out = asyncio.run(go())
        self.assertEqual(out.answers["q"], {"type": "choice", "choice": "only", "confidence": 1.0,
                                            "probabilities": {"only": 1.0}})
        self.assertEqual(out.usage, {"input_tokens": 0, "output_tokens": 0})


if __name__ == "__main__":
    unittest.main()
