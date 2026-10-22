"""Checks against the tokenizer of a real model folder (only its tokenizer files are read). Set
STARTLUX_TEST_MODEL_DIR to a StartLux-Decision folder to run them; they are skipped otherwise.

* the letter ids match the folder's letter_token_ids, and the one-pass / thinking prompts end as trained;
* a thought continues the token ids of the empty-thought read-out but its last three (first_read "thinking");
* with first_read "thinking", every engine call of a request that thinks fits the context up to its end;
* the vLLM plugin's prompt-end rule (on token ids) gives the same prompt end as the transformers reference rule
  (the last "<think>" followed by a newline, or the whole input for a one-pass prompt) on many prompts;
* the pipeline runs on the real chat template.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import unittest

from startlux_decide.backends.fake import FakeBackend
from startlux_decide.controls import Controls
from startlux_decide.modelcfg import RELEASE_PREFIX, load_model_folder, load_tokenizer
from startlux_decide.pipeline import CONTEXT_MARGIN, MIN_ROOM, Decider
from startlux_decide.prompt import (check_tokenizer, count_tokens, empty_thought_prompt, from_systemone, messages,
                                    plain_prompt, think_prompt)
from startlux_decide.vllm_plugin import PromptEnd

from .test_golden import random_question, random_state

MODEL_DIR = os.environ.get("STARTLUX_TEST_MODEL_DIR")


def reference_prompt_end(ids, tok):
    """The transformers reference rule (the training code's): the whole input when it ends with the empty think
    block, else right after the "<think>\\n" of the last "<think>"."""
    think = tok.convert_tokens_to_ids("<think>")
    j = [i for i, t in enumerate(ids) if t == think][-1]
    if list(ids[j:]) == tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False):
        return len(ids)
    assert tok.decode(list(ids[j:j + 2])) == "<think>\n"
    return j + 2


@unittest.skipUnless(MODEL_DIR, "set STARTLUX_TEST_MODEL_DIR to a model folder")
class RealTokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = load_tokenizer(MODEL_DIR)
        with open(os.path.join(MODEL_DIR, "decision_config.json")) as f:
            cls.cfg = json.load(f)

    def test_letters_and_prompts(self):
        self.assertEqual(check_tokenizer(self.tok), self.cfg["letter_token_ids"])
        msgs = messages(from_systemone({"ticket": "charged twice"}, {"type": "noul", "instructions": "Refund?"}))
        self.assertTrue(plain_prompt(self.tok, msgs).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertTrue(think_prompt(self.tok, msgs).endswith("<|im_start|>assistant\n<think>\n"))
        self.assertTrue(plain_prompt(self.tok, msgs).startswith("<|im_start|>system\nApply the criterion"))

    def test_a_thought_continues_the_empty_thought_read_out(self):
        msgs = messages(from_systemone({"ticket": "charged twice"}, {"type": "noul", "instructions": "Refund?"}))
        read = self.tok.encode(empty_thought_prompt(self.tok, msgs), add_special_tokens=False)
        thought = self.tok.encode(think_prompt(self.tok, msgs) + RELEASE_PREFIX, add_special_tokens=False)
        self.assertEqual(read[:-3], thought[:len(read) - 3])
        self.assertEqual(self.tok.decode(read[-3:]), "\n\n</think>\n\n")

    def test_first_read_thinking_fits_the_context_to_its_end(self):
        # A thought cut by the room the context leaves it gives the longest read-out after a thought.
        words = [w for w in (" river", " stone", " cloud", " field", " paper", " market", " window", " garden",
                             " silver", " orange", " planet", " signal", " bridge", " castle", " forest", " island")
                 if len(self.tok.encode(w, add_special_tokens=False)) == 1]
        self.assertGreater(len(words), 8)

        def cut(prompt, seed, max_tokens):              # one token per word, never closed, never a loop
            r = random.Random(seed)
            return [r.choice(words) for _ in range(max_tokens)], False

        folder, context, spec = load_model_folder(MODEL_DIR), 1024, {"type": "noul", "instructions": "Refund?"}
        if folder.auto_think is None:
            self.skipTest("this folder does not think")
        ctl = Controls(think_mode="always", first_read="thinking", details=True)
        rooms, cut_at_room = [], 0
        for n in range(700, 860):
            state = "Ticket: charged twice." + " the" * n
            room = context - count_tokens(self.tok, empty_thought_prompt(
                self.tok, messages(from_systemone(state, spec)))) - CONTEXT_MARGIN
            rooms.append(room)
            engine = FakeBackend(tokenizer=self.tok, script=cut)
            out = asyncio.run(Decider(folder, engine, context_length=context).decide(state, {"q": spec}, ctl))
            thought = out.answers["q"]["details"]["thought"]
            self.assertEqual(thought["used"], room > MIN_ROOM, room)
            cut_at_room += thought["used"] and thought["tokens"] == room
            for call in engine.calls:
                generated = call[2] if call[0] == "generate" else 1
                self.assertLessEqual(count_tokens(self.tok, call[1]) + generated, context, (room, call[0]))
        self.assertTrue(0 <= min(rooms) <= MIN_ROOM and max(rooms) > 100, rooms)
        self.assertGreater(cut_at_room, 50)

    def test_prompt_end_rule(self):
        pe, r = PromptEnd(self.tok), random.Random(3)
        prefix = "Let me work through this carefully, step by step.\n"
        for _ in range(300):
            spec = random_question(r)
            if isinstance(spec.get("criteria"), dict) and len(spec["criteria"]) > 26:
                continue
            msgs = messages(from_systemone(random_state(r), spec))
            thought = r.choice(["", "The refund rule applies.\nB fits.", "A stray <think>\n tag.", "x <think> y"])
            base = len(self.tok.encode(think_prompt(self.tok, msgs), add_special_tokens=False))
            plain = plain_prompt(self.tok, msgs)
            for text in (think_prompt(self.tok, msgs) + prefix + thought,
                         think_prompt(self.tok, msgs) + prefix + thought + "\n</think>\n\n"):
                ids = self.tok.encode(text, add_special_tokens=False)
                self.assertEqual(pe(ids)[0], base)
                if "<think>" not in thought:            # the reference takes the LAST <think>; a stray one in the
                    self.assertEqual(base, reference_prompt_end(ids, self.tok))   # thought would move it
            ids = self.tok.encode(plain, add_special_tokens=False)
            self.assertEqual(pe(ids), (len(ids), "one_pass"))
            self.assertEqual(reference_prompt_end(ids, self.tok), len(ids))

    def test_pipeline_on_the_real_template(self):
        folder = load_model_folder(MODEL_DIR)
        engine = FakeBackend(tokenizer=self.tok)
        d = Decider(folder, engine)
        out = asyncio.run(d.decide({"ticket": "charged twice"}, {
            "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "shipping": None}}},
            Controls(think_mode="never")))
        self.assertEqual(out.answers["team"]["type"], "choice")
        self.assertTrue(engine.calls[0][1].endswith("B) shipping<|im_end|>\n<|im_start|>assistant\n"
                                                   "<think>\n\n</think>\n\n"))
        self.assertGreater(out.usage["input_tokens"], 40)


if __name__ == "__main__":
    unittest.main()
