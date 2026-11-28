"""A deterministic stand-in for a model, for tests and for trying clients without a GPU.

FakeTokenizer renders the same chat layout as the StartLux-Decision tokenizers (special tokens, the empty think block
of one-pass prompts, the opened think block of thinking prompts) and reads every other character as one token.
FakeBackend answers read-outs with log-probabilities derived from a hash of the prompt and writes thoughts from a
hash of the prompt and the seed; an optional `script` decides how a thought behaves. With ``streams=True`` it also
streams thoughts, `burst` tokens per event (several tokens per step, as with speculative decoding), and counts the
streams closed before their end in `aborted`. With ``reads_images=True`` its calls take media (the images' bytes
then enter the read-out hash), and with ``caches_prefixes=True`` it records prefill calls.
"""
from __future__ import annotations

import asyncio
import hashlib
import random
import re
from typing import Callable, Sequence

from .base import Backend, Generation, Sampling, StreamEvent

SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<|vision_start|>",
                  "<|image_pad|>", "<|vision_end|>")
_SPECIAL_BASE = 2_000_000           # above every Unicode code point
_SPLIT = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")


class FakeTokenizer:
    """Chat template and tokenizer with the layout of the real ones; one token per character outside special tokens.
    open_think=False imitates a template that does not open the think block when thinking is on. `think_note` starts
    the system message of a thinking prompt, as the release template's reasoning-effort line does."""

    eos_token_id = _SPECIAL_BASE + SPECIAL_TOKENS.index("<|im_end|>")
    pad_token_id = _SPECIAL_BASE + SPECIAL_TOKENS.index("<|endoftext|>")

    def __init__(self, open_think: bool = True, think_note: str = ""):
        self.open_think, self.think_note = open_think, think_note

    def apply_chat_template(self, messages: Sequence[dict], tokenize: bool = False, add_generation_prompt: bool = True,
                            enable_thinking: bool = True) -> str:
        note = self.think_note if enable_thinking is not False else ""
        text = "".join(f"<|im_start|>{m['role']}\n{note if m['role'] == 'system' else ''}{_content(m['content'])}"
                       "<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
            if enable_thinking is False:
                text += "<think>\n\n</think>\n\n"
            elif self.open_think:
                text += "<think>\n"
        return self.encode(text) if tokenize else text

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for part in _SPLIT.split(text):
            if part in SPECIAL_TOKENS:
                ids.append(_SPECIAL_BASE + SPECIAL_TOKENS.index(part))
            else:
                ids.extend(ord(c) for c in part)
        return ids

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": self.encode(text, add_special_tokens)}

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        out = []
        for t in ids:
            if t >= _SPECIAL_BASE:
                if not skip_special_tokens:
                    out.append(SPECIAL_TOKENS[t - _SPECIAL_BASE])
            else:
                out.append(chr(t))
        return "".join(out)

    def convert_tokens_to_ids(self, token: str) -> int:
        if token in SPECIAL_TOKENS:
            return _SPECIAL_BASE + SPECIAL_TOKENS.index(token)
        return ord(token) if len(token) == 1 else -1


def _content(content) -> str:
    """A message content as the models' chat template writes it: text parts as they are, an image as its
    placeholder, the whole trimmed."""
    if isinstance(content, str):
        return content
    return "".join(p["text"] if p.get("type") == "text" else "<|vision_start|><|image_pad|><|vision_end|>"
                   for p in content).strip()


def _rng(*parts: object) -> random.Random:
    return random.Random(hashlib.sha256("\x00".join(map(str, parts)).encode("utf-8", "surrogatepass")).digest())


def default_thought(prompt: str, seed: int, max_tokens: int) -> tuple[list[str], bool]:
    """A thought of words from a hash of the prompt and the seed; (tokens, ends with the end-of-sequence token)."""
    r = _rng("thought", prompt, seed)
    n = r.randint(0, max_tokens + 50)
    words = [r.choice(("the", "option", "fits", "because", "evidence", "so", "B", "is", "likely", "not", "A."))
             for _ in range(min(n, max_tokens))]
    return [w + " " for w in words], n <= max_tokens and r.random() < 0.5


class FakeBackend(Backend):
    """Deterministic read-outs and thoughts (see the module docstring). `script(prompt, seed, max_tokens)` may return
    (tokens, eos) to control a thought; `delay` seconds are spent in every call; every call is recorded in `calls`."""

    name = "fake"

    def __init__(self, tokenizer: FakeTokenizer | None = None,
                 script: Callable[[str, int, int], tuple] | None = None, delay: float = 0.0,
                 context_length: int | None = None, streams: bool = False, burst: int = 1,
                 reads_images: bool = False, caches_prefixes: bool = False):
        self.tokenizer = tokenizer or FakeTokenizer()
        self.script = script or default_thought
        self.delay = delay
        self.context_length = context_length
        self.streams = streams
        self.reads_images, self.caches_prefixes = reads_images, caches_prefixes
        self.burst = max(1, burst)
        self.aborted = 0
        self.calls: list[tuple] = []

    async def readout(self, prompt: str, letter_ids: Sequence[int], media=None) -> list[float]:
        self.calls.append(("readout", prompt, tuple(letter_ids)) + ((len(media),) if media is not None else ()))
        if self.delay:
            await asyncio.sleep(self.delay)
        r = _rng("readout", prompt, *(media.data_uris() if media is not None else ()))
        return [r.uniform(-8.0, 0.0) for _ in letter_ids]

    async def prefill(self, prompt: str, media=None) -> None:
        self.calls.append(("prefill", prompt) + ((len(media),) if media is not None else ()))

    def _outcome(self, prompt: str, max_tokens: int, stop: Sequence[str], seed: int) -> tuple[list[str], Generation]:
        """The script's tokens and what a generation of them returns."""
        tokens, eos = self.script(prompt, seed, max_tokens)
        tokens = tokens[:max_tokens]
        text = "".join(tokens)
        for s in stop:
            if s in text:
                cut = text.index(s)
                used = next(i for i in range(len(tokens) + 1) if len("".join(tokens[:i])) >= cut + len(s))
                return tokens, Generation(text[:cut], used, "stop")
        if eos and len(tokens) < max_tokens:
            return tokens, Generation(text, len(tokens) + 1, "stop")
        return tokens, Generation(text, len(tokens), "length")

    async def generate(self, prompt: str, max_tokens: int, sampling: Sampling, stop: Sequence[str],
                       seed: int, media=None) -> Generation:
        self.calls.append(("generate", prompt, max_tokens, sampling, tuple(stop), seed)
                          + ((len(media),) if media is not None else ()))
        if self.delay:
            await asyncio.sleep(self.delay)
        return self._outcome(prompt, max_tokens, stop, seed)[1]

    async def generate_stream(self, prompt: str, max_tokens: int, sampling: Sampling, stop: Sequence[str],
                              seed: int, media=None):
        """The generation of `generate` as events of `burst` tokens; the last event has the finish reason."""
        self.calls.append(("generate_stream", prompt, max_tokens, sampling, tuple(stop), seed)
                          + ((len(media),) if media is not None else ()))
        if not self.streams:
            raise NotImplementedError("this fake engine was made without streams=True")
        if self.delay:
            await asyncio.sleep(self.delay)
        tokens, final = self._outcome(prompt, max_tokens, stop, seed)
        ends = [0]
        for t in tokens:
            ends.append(ends[-1] + len(t))
        done, pos, k = False, 0, self.burst
        try:
            while k < final.tokens:
                end = min(ends[min(k, len(tokens))], len(final.text))
                yield StreamEvent(final.text[pos:end], k)
                pos, k = end, k + self.burst
            done = True
            yield StreamEvent(final.text[pos:], final.tokens, final.finish_reason)
        finally:
            if not done:
                self.aborted += 1
