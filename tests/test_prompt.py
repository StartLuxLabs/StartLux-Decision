"""The prompt format: exact text, the reference renderer on random questions, validation and the tokenizer check."""
from __future__ import annotations

import random
import unittest

from startlux_decide.backends.fake import FakeTokenizer
from startlux_decide.prompt import (SYSTEM, THINK_OFF_SUFFIX, check_tokenizer, count_tokens, empty_thought_prompt,
                                    from_systemone, messages, plain_prompt, score_keys, state_text, think_prompt)

from . import reference_impl as ref
from .test_golden import random_question, random_state


class ExactTextTest(unittest.TestCase):
    def test_choice_with_descriptions(self):
        row = from_systemone({"ticket": "charged twice"}, {
            "type": "choice", "instructions": " Which team? ",
            "criteria": {"billing": "Payments and refunds", "shipping": None, "tech": "TECH"}})
        self.assertEqual(messages(row), [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": 'Evidence:\n{"ticket": "charged twice"}\n\nQuestion: Which team?\nOptions:\n'
                                        "A) billing: Payments and refunds\nB) shipping\nC) tech"}])

    def test_yes_no_defaults_and_empty_state(self):
        row = from_systemone(None, {"type": "bool"})
        self.assertEqual(messages(row)[1]["content"],
                         "Evidence:\n(none)\n\nQuestion: Which answer fits the evidence?\nOptions:\nA) yes\nB) no")

    def test_bare_ids_are_hidden_when_every_option_is_described(self):
        row = from_systemone("x", {"instructions": "Capital?", "criteria": {"A": "Paris", "B": "Rome"}})
        self.assertEqual(messages(row)[1]["content"].split("Options:\n")[1], "A) Paris\nB) Rome")
        row = from_systemone("x", {"instructions": "Capital?", "criteria": {"A": "Paris", "B": None}})
        self.assertEqual(messages(row)[1]["content"].split("Options:\n")[1], "A) A: Paris\nB) B")

    def test_long_arrays_carry_their_index(self):
        self.assertEqual(state_text({"v": list(range(8))}),
                         '{"v": [{"_index": 0, "value": 0}, {"_index": 1, "value": 1}, {"_index": 2, "value": 2}, '
                         '{"_index": 3, "value": 3}, {"_index": 4, "value": 4}, {"_index": 5, "value": 5}, '
                         '{"_index": 6, "value": 6}, {"_index": 7, "value": 7}]}')
        self.assertEqual(state_text([1, 2]), "[1, 2]")
        self.assertEqual(state_text("naïve 日本"), "naïve 日本")

    def test_score_levels_lowest_first(self):
        self.assertEqual(score_keys({"10": "a", "2": "b", "0": "c"}), ["0", "2", "10"])
        self.assertEqual(score_keys({"low": 1, "high": 2}), ["low", "high"])
        row = from_systemone("s", {"type": "score", "instructions": "How bad?", "criteria": {"2": "high", "0": "low",
                                                                                            "1": "mid"}})
        self.assertEqual([o["criterion"] for o in row["options"]], ["low", "mid", "high"])

    def test_prompts(self):
        tok = FakeTokenizer()
        msgs = messages(from_systemone("s", {"type": "noul", "instructions": "q?"}))
        head = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{msgs[1]['content']}<|im_end|>\n"
        self.assertEqual(plain_prompt(tok, msgs), head + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        self.assertEqual(think_prompt(tok, msgs), head + "<|im_start|>assistant\n<think>\n")
        self.assertEqual(think_prompt(FakeTokenizer(open_think=False), msgs), head + "<|im_start|>assistant\n<think>\n")
        self.assertEqual(count_tokens(tok, "<think>ab"), 3)

    def test_empty_thought_prompt(self):
        note = "Reasoning effort is set to high.\n\n"
        tok = FakeTokenizer(think_note=note)
        msgs = messages(from_systemone("s", {"type": "noul", "instructions": "q?"}))
        read = empty_thought_prompt(tok, msgs)
        self.assertEqual(read, think_prompt(tok, msgs) + "\n</think>\n\n")      # a thought continues the read-out
        self.assertTrue(read.endswith(THINK_OFF_SUFFIX))
        self.assertEqual(read.replace(note, "", 1), plain_prompt(tok, msgs))   # the note is all that differs
        unopened = FakeTokenizer(open_think=False)                             # think_prompt opens the block itself
        self.assertEqual(empty_thought_prompt(unopened, msgs), plain_prompt(unopened, msgs))

        class Unopened(FakeTokenizer):              # ends the thinking prompt with "<think>" and no newline
            def apply_chat_template(self, *args, **kwargs):
                return super().apply_chat_template(*args, **kwargs).removesuffix("\n")
        with self.assertRaisesRegex(ValueError, "opened think block"):
            empty_thought_prompt(Unopened(), msgs)


class ReferenceRendererTest(unittest.TestCase):
    def test_random_questions_render_as_the_training_renderer(self):
        r = random.Random(7)
        for _ in range(3000):
            state, spec = random_state(r), random_question(r)
            crit = spec.get("criteria")
            if isinstance(crit, dict) and len(crit) > 26:
                spec = dict(spec, criteria=dict(list(crit.items())[:26]))
            want = ref.messages(ref.from_systemone(state, spec))[0]
            self.assertEqual(messages(from_systemone(state, spec)), want)


class ValidationTest(unittest.TestCase):
    def bad(self, state, spec, fragment):
        with self.assertRaises(ValueError) as cm:
            messages(from_systemone(state, spec))
        self.assertIn(fragment, str(cm.exception))

    def test_errors(self):
        self.bad("s", {"type": "rank", "criteria": ["a", "b"]}, "unknown question type")
        self.bad("s", {"type": "choice", "criteria": ["a", "b"]}, "instructions are required")
        self.bad("s", {"type": "choice", "instructions": "q"}, "needs criteria")
        self.bad("s", {"type": "choice", "instructions": "q", "criteria": "a,b"}, "needs criteria")
        self.bad("s", {"type": "choice", "instructions": "q", "criteria": [str(i) for i in range(27)]}, "2 to 26")
        self.bad("s", {"type": "choice", "instructions": "q", "criteria": ["a"]}, "2 to 26")
        self.bad("s", {"type": "choice", "instructions": "q", "criteria": {"a\nb": None, "c": None}}, "bad option id")
        self.bad("s", {"type": "choice", "instructions": "q", "criteria": [1, "1"]}, "2 to 26")   # one id after str()
        self.bad("s", {"type": "score", "instructions": "q", "criteria": [str(i) for i in range(11)]}, "2 to 10 levels")
        self.bad("s", {"type": "score", "instructions": "q", "criteria": 5}, "list of levels")


class TokenizerCheckTest(unittest.TestCase):
    def test_letters(self):
        self.assertEqual(check_tokenizer(FakeTokenizer()), list(range(65, 91)))

    def test_template_without_thinking_off_prefix(self):
        class Plain(FakeTokenizer):
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
                return "".join(m["content"] for m in messages) + "\nassistant:"
        with self.assertRaises(ValueError):
            check_tokenizer(Plain())


if __name__ == "__main__":
    unittest.main()
