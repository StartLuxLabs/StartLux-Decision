"""The decision pipeline. Every question of a request runs as its own task, so no question waits for another one's
thought, and many requests can be in flight at once.

One question:

1. Choice lists over 26 options are first knocked out: near-equal groups of at most 25 options are read in one pass
   each, the 3 best of every group go to the final round (the 26 best overall when more qualify). This holds for
   every form of a choice list: {option: description}, a list of options, or either under "options".
2. First read-out: the one-pass prompt (thinking off), letter probabilities at the type's temperature. With first_read
   "thinking", a request that may think renders its one-pass prompts as the thinking prompt with an empty thought
   instead (prompt.empty_thought_prompt), and a thought continues their beginning.
3. Gate (AutoThink): think when think_mode is "always", or "auto" and the top probability is below the gate, and the
   context has room beyond the thought's prompt. The thought starts with the forced prefix and is generated in one of two ways
   (thought_generation): "stream", one streamed request, where the loop detector looks at the thought each time
   another `chunk_tokens` tokens have arrived and a looping thought is cut there and its request ended; or "chunked",
   `chunk_tokens` tokens per request, each request re-sending the prompt and the thought so far, with the loop detector
   between them. Both use the same prompt, seed, sampling, stop string and budget, and both run the loop detector once
   more when the budget ends the thought. "auto" (the default) streams on engines that stream thoughts safely (vLLM
   0.31 or later) and uses chunks elsewhere. A thought that is neither closed by the model nor ended at a loop is cut
   with the line CUT.
4. Second read-out after the thought (its last `tail_chars` characters, marked "[...]"), re-tempered with the think
   temperature and mixed with the first: (1 - mix) * first + mix * after.
5. Knocked-out options get a small residual share of probability; the answer is built in the /v1/systemone format.

allow_none (choice questions, a control of the request that a question can override): the question gets one extra last
option, "None of the above", that every read-out and the thought see like any other option. A choice list that does not
leave a letter free for it (26 options) is knocked out first, and the knock-out's final round, not its group rounds,
holds the extra option. The answer's probabilities are the listed options' renormalised to sum to 1, "none_probability"
is the extra option's probability before that, and "choice" is null when the extra option is the most likely one.

Images (images.py): a request may carry images as part of its evidence; they take their places in the state, every
prompt of the request holds them, and every engine call of the request sends them along (engines with reads_images).

Long states: on an engine with a prefix cache (vLLM 0.31 or later) and a causal folder, the questions of a request
whose one-pass prompts share a beginning of SHARE_MIN tokens or more run on one server, after that beginning has been
read into the cache once; each question then reads only its own part. A thought reads only its own part too with
first_read "thinking". With "plain" it does not: the release template's thinking prompt differs from the one-pass
prompt from the start of its system message on, so the first thought of a request reads the whole state again (the
others share its beginning). Instant folders read each prompt whole, every time: their prompt attention is
bidirectional, so no prefix can be reused.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Sequence

from .backends.base import Backend, BackendRequestError
from .controls import STREAM_REFUSED, Controls, RequestError, parse_controls
from .images import IMAGE_PLACEHOLDER, MAX_IMAGES, MAX_PIXELS, MIN_PIXELS, Media, load_media, place_images
from .loops import loop_cut, looping
from .modelcfg import AutoThink, ModelFolder
from .prompt import (MAX_OPTIONS, NONE_OPTION, Row, check_thinking_prompt, check_tokenizer, count_tokens,
                     empty_thought_prompt, from_systemone, messages, plain_prompt, state_text, think_prompt)
from .readout import (answer, choice_answer, knockout_combine, knockout_finalists, knockout_groups, mixture,
                      option_probabilities, probabilities, retemper, split_none)

CUT = "\n\nI have to decide now with what I have worked out so far.\n</think>\n\n"
CLOSE = "\n</think>\n\n"
STOP = ("</think>",)
DEFAULT_CONTEXT = 65536     # vLLM --max-model-len of the release runs
CONTEXT_MARGIN = 64         # beyond a thought's prompt and budget: the line ending it and the read-out's letter
MIN_ROOM = 32               # no thought when fewer tokens than this are left
SHARE_MIN = 4096            # a shared prompt beginning this long is read into the prefix cache once per request


def default_controls(folder: ModelFolder, **overrides: Any) -> Controls:
    """Server defaults: the folder's AutoThink settings (Instant folders never think), then the overrides that are
    not None (think_mode, gate, budget, mix, thought_generation, none_check, allow_none)."""
    auto = folder.auto_think or AutoThink(think_mode="never")
    base = Controls(think_mode=auto.think_mode, gate=auto.gate, budget=auto.budget, mix=auto.mix)
    return replace(base, **{k: v for k, v in overrides.items() if v is not None})


def choice_criteria(spec: dict) -> Any:
    """The options of a choice question in the form the knock-out works on, {id: description}: "criteria" (or
    "options"), a list of options read as from_systemone reads it ({str(option): None}). Anything else as it is."""
    crit = spec.get("criteria", spec.get("options"))
    if isinstance(crit, (list, tuple)):
        return {str(c): None for c in crit}
    return crit


@dataclass
class _Question:
    """A validated question, ready to run."""
    key: str
    spec: dict
    row: Row | None = None                  # ordinary question
    msgs: list | None = None
    only: str | None = None                 # choice with a single option: answered without the model
    groups: list | None = None              # knock-out: option groups and their one-pass prompts
    group_prompts: list = field(default_factory=list)
    group_tokens: int = 0
    none: bool = False                      # choice with the extra last option "None of the above"
    null_ok: bool = False                   # allow_none: "choice" may be null (with none_check alone it may not)


@dataclass
class _Thought:
    text: str               # the whole thought (forced prefix first, think tags removed)
    read_prompt: str        # the prompt of the read-out after the thought
    tokens: int
    ended: str              # closed | loop | budget
    start: float
    end: float
    generation: str = "chunked"     # stream | chunked


@dataclass
class _Request:
    """What the questions of one request share: its images, the form of its one-pass prompts, and the server when they
    share a long state."""
    media: Media | None = None
    extra_tokens: int = 0               # image tokens beyond the one placeholder token each, in every prompt
    lane: Backend | None = None
    shared_tokens: int = 0              # tokens of the shared beginning read into the cache (0: none)
    empty_thought: bool = False         # one-pass prompts are empty_thought_prompt (first_read "thinking")

    def kw(self) -> dict:
        """The keyword arguments of an engine call."""
        return {"media": self.media} if self.media is not None else {}


@dataclass
class _Outcome:
    answer: dict
    input_tokens: int
    output_tokens: int
    first_ms: float = 0.0
    after_ms: float = 0.0
    thought: tuple | None = None            # (start, end) on the perf_counter clock


@dataclass
class Decision:
    """The result of one request; timing_ms: wall, first_pass_sum, thought_span, after_thought_sum (milliseconds)."""
    answers: dict
    usage: dict
    timing_ms: dict


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


async def _gather(aws: Sequence[Awaitable]) -> list:
    """asyncio.gather that cancels the other tasks when one fails."""
    tasks = [asyncio.ensure_future(a) for a in aws]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class Decider:
    """Runs /v1/systemone requests on one backend for one model folder."""

    def __init__(self, folder: ModelFolder, backend: Backend, defaults: Controls | None = None,
                 max_budget: int | None = None, context_length: int | None = None, max_images: int = MAX_IMAGES,
                 max_pixels: int = MAX_PIXELS, min_pixels: int = MIN_PIXELS):
        self.folder, self.backend = folder, backend
        self.max_images, self.max_pixels, self.min_pixels = max_images, max_pixels, min_pixels
        self.tok = backend.tokenizer
        self.letter_ids = check_tokenizer(self.tok)
        if folder.letter_token_ids and tuple(self.letter_ids) != folder.letter_token_ids:
            raise ValueError("the tokenizer's letter ids differ from letter_token_ids in decision_config.json")
        self.auto = folder.auto_think
        if self.auto is not None:
            check_thinking_prompt(self.tok)     # any request may ask for first_read "thinking"
        self.defaults = defaults or default_controls(folder)
        if self.defaults.thought_generation == "stream" and self.auto is not None and not backend.streams:
            raise ValueError("the default thought generation is \"stream\", but this engine does not stream thoughts "
                             "(vLLM 0.31 or later does)")
        self.max_budget = max_budget or self.defaults.budget
        self.context_length = context_length or backend.context_length or DEFAULT_CONTEXT

    @property
    def can_think(self) -> bool:
        return self.auto is not None

    @property
    def reads_images(self) -> bool:
        """The folder has a vision tower and the engine reads images."""
        return self.folder.vision and self.backend.reads_images and self.max_images > 0

    def controls(self, body: dict) -> Controls:
        """The controls of a request body (see controls.py); raises RequestError."""
        return parse_controls(body, self.defaults, self.max_budget, self.can_think, self.backend.streams)

    def generation(self, ctl: Controls) -> str:
        """How a thought of this request is generated: "stream" or "chunked" ("auto" follows the engine)."""
        if ctl.thought_generation == "auto":
            return "stream" if self.backend.streams else "chunked"
        return ctl.thought_generation

    @staticmethod
    def _request(ctl: Controls) -> _Request:
        """A new request: its one-pass prompts take the thinking form when first_read asks for it and the request may
        think (a request that never thinks has no thought to continue them)."""
        return _Request(empty_thought=ctl.first_read == "thinking" and ctl.think_mode != "never")

    def _one_pass(self, msgs: list, req: _Request) -> str:
        """The one-pass prompt of a question of this request."""
        return (empty_thought_prompt if req.empty_thought else plain_prompt)(self.tok, msgs)

    # ---------------------------------------------------------------- validation
    def prepare(self, state: Any, questions: Any, req: _Request | None = None,
                allow_none: bool = False, none_check: bool = False) -> list[_Question]:
        """Check every question and render what can be rendered before any model call; raises RequestError.
        `allow_none`: the request's setting, which a choice question's own "allow_none" overrides. `none_check`: choice
        questions get the extra option too, but their choice stays a listed option unless allow_none is set."""
        if not isinstance(questions, dict) or not questions:
            raise RequestError("questions must be a non-empty object", ("body", "questions"))
        out = []
        for key, spec in questions.items():
            loc = ("body", "questions", key)
            if not isinstance(spec, dict):
                raise RequestError("a question must be an object", loc)
            none = allow_none
            if spec.get("type", "choice") == "choice" and spec.get("allow_none") is not None:
                none = spec["allow_none"]
                if not isinstance(none, bool):
                    raise RequestError(f"allow_none must be true or false, not {none!r}", loc + ("allow_none",))
            try:
                q = self._prepare_one(state, key, spec, req or _Request(), none or none_check)
                q.null_ok = none and q.none
                out.append(q)
            except ValueError as e:
                raise RequestError(str(e), loc) from None
        return out

    def _prepare_one(self, state: Any, key: str, spec: dict, req: _Request, allow_none: bool = False) -> _Question:
        none = allow_none and spec.get("type", "choice") == "choice"
        crit = choice_criteria(spec) if spec.get("type", "choice") == "choice" else None
        if isinstance(crit, dict):
            if crit is not spec.get("criteria"):        # a list of options, or the "options" key: read as a dict
                spec = dict(spec, criteria=crit)
            if len(crit) == 1 and not none:
                return _Question(key, spec, only=next(iter(crit)))
            if len(crit) > MAX_OPTIONS - none:      # the extra option needs a letter of the final round
                q = _Question(key, spec, groups=knockout_groups(list(crit), self.folder.wide_group), none=none)
                for g in q.groups:
                    row = from_systemone(state, dict(spec, criteria={c: crit[c] for c in g}))
                    q.group_prompts.append(self._one_pass(messages(row), req))
                    q.group_tokens += count_tokens(self.tok, q.group_prompts[-1]) + req.extra_tokens
                return q
        row = from_systemone(state, spec, none)
        return _Question(key, spec, row=row, msgs=messages(row), none=none)

    # ---------------------------------------------------------------- running
    async def decide(self, state: Any, questions: Any, controls: Controls | None = None,
                     images: Any = None) -> Decision:
        """Answer one request: {key: answer} (with details when controls.details is set), usage and timings.
        `images`: the request's images (base64 strings or data URIs), placed in the state (images.place_images)."""
        t0 = time.perf_counter()
        ctl = controls or self.defaults
        if ctl.think_mode != "never" and not self.can_think:
            raise RequestError("this is an Instant model: it does not think", ("body", "think_mode"))
        if ctl.think_mode != "never" and self.generation(ctl) == "stream" and not self.backend.streams:
            raise RequestError(STREAM_REFUSED, ("body", "thought_generation"))
        req = self._request(ctl)
        if images is not None and images != []:
            state, req.media = self._images(state, images)
            req.extra_tokens = sum(t - 1 for t in req.media.tokens)
        items = self.prepare(state, questions, req, ctl.allow_none, ctl.none_check)
        await self._share(items, req)
        outcomes = await _gather([self._question(state, q, ctl, req) for q in items])
        spans = [o.thought for o in outcomes if o.thought]
        return Decision(answers={q.key: o.answer for q, o in zip(items, outcomes)},
                        usage={"input_tokens": sum(o.input_tokens for o in outcomes),
                               "output_tokens": sum(o.output_tokens for o in outcomes)},
                        timing_ms={"wall": _ms(time.perf_counter() - t0),
                                   "first_pass_sum": round(sum(o.first_ms for o in outcomes), 3),
                                   "thought_span": _ms(max(e for _, e in spans) - min(s for s, _ in spans))
                                   if spans else 0.0,
                                   "after_thought_sum": round(sum(o.after_ms for o in outcomes), 3)})

    def _images(self, state: Any, images: Any) -> tuple[str, Media]:
        """The state with the places of the request's images, and the decoded images; raises RequestError."""
        if not self.backend.reads_images or self.max_images <= 0:
            raise RequestError(f"the {self.backend.name} engine of this server reads text only: send the evidence "
                               "without images", ("body", "images"))
        if not self.folder.vision:
            raise RequestError("this model folder has no vision tower: it reads text only", ("body", "images"))
        try:
            media = load_media(images, self.min_pixels, self.max_pixels, self.max_images, self.folder.image_factor)
        except ValueError as e:
            raise RequestError(str(e), ("body", "images")) from None
        try:
            return place_images(state_text(state), len(media)), media
        except ValueError as e:
            raise RequestError(str(e), ("body", "state")) from None

    async def _share(self, items: list[_Question], req: _Request) -> None:
        """A long state shared by several questions is read once (module docstring): the shared beginning of their
        one-pass prompts goes into the engine's prefix cache, and the questions run on that server."""
        if not self.backend.caches_prefixes or self.folder.bidirectional:
            return
        prompts = [p for q in items for p in q.group_prompts]
        prompts += [self._one_pass(q.msgs, req) for q in items if q.msgs is not None]
        if len(prompts) < 2:
            return
        prefix = os.path.commonprefix(prompts)
        if req.media is not None and prefix.count(IMAGE_PLACEHOLDER) != len(req.media):
            return
        n = count_tokens(self.tok, prefix) + req.extra_tokens
        if n < SHARE_MIN:
            return
        req.lane = self.backend.lane()
        await req.lane.prefill(prefix, **req.kw())
        req.shared_tokens = n

    async def _read(self, lane: Backend, prompt: str, k: int, question_type: str, req: _Request) -> list[float]:
        z = await lane.readout(prompt, self.letter_ids[:k], **req.kw())
        return probabilities(z, self.folder.temperature(question_type))

    async def _question(self, state: Any, q: _Question, ctl: Controls, req: _Request) -> _Outcome:
        try:
            return await self._run(state, q, ctl, req)
        except BackendRequestError as e:
            raise RequestError(f"the engine refused this question: {e}", ("body", "questions", q.key)) from None

    async def _run(self, state: Any, q: _Question, ctl: Controls, req: _Request) -> _Outcome:
        """One question (see the module docstring). Timings wrap the engine calls only."""
        t_start = time.perf_counter()
        if q.only is not None:
            only = {"type": "choice", "choice": q.only, "confidence": 1.0, "probabilities": {q.only: 1.0}}
            out = _Outcome(only, 0, 0)
            if ctl.details:
                out.answer["details"] = self._details(ctl, t_start, first={q.only: 1.0}, top=1.0, first_ms=0.0)
            return out
        lane = req.lane or self.backend.lane()
        row, msgs, first, first_ms = q.row, q.msgs, None, 0.0
        if q.groups:
            t0 = time.perf_counter()
            reads = await _gather([self._read(lane, p, len(g), "choice", req)
                                   for p, g in zip(q.group_prompts, q.groups)])
            first_ms = _ms(time.perf_counter() - t0)
            first, finalists = knockout_finalists(q.groups, reads, self.folder.wide_keep, MAX_OPTIONS - q.none)
            row = from_systemone(state, dict(q.spec, criteria={c: q.spec["criteria"][c] for c in finalists}), q.none)
            msgs = messages(row)
        one_pass = self._one_pass(msgs, req)
        k, typ = len(row["options"]), row["type"]
        n_in = count_tokens(self.tok, one_pass) + req.extra_tokens
        t0 = time.perf_counter()
        p0 = await self._read(lane, one_pass, k, typ, req)
        first_ms += _ms(time.perf_counter() - t0)
        p, p1, thought, after_ms = p0, None, None, 0.0
        think = ctl.think_mode == "always" or (ctl.think_mode == "auto" and max(p0) < ctl.gate)
        if think and (room := self._room(msgs, req)) > MIN_ROOM:
            thought = await self._think(lane, msgs, min(ctl.budget, room), self.generation(ctl), req)
            t0 = time.perf_counter()
            p1 = await self._read(lane, thought.read_prompt, k, typ, req)
            after_ms = _ms(time.perf_counter() - t0)
            if typ in self.auto.think_temperature:
                p1 = retemper(p1, self.folder.temperature(typ), self.auto.think_temperature[typ])
            p = mixture(p0, p1, ctl.mix)
        if q.none:
            ans = self._none_answer(row, first, p, q.null_ok)
        else:
            ans = choice_answer(self._keyed(row, first, p)) if first is not None else answer(row, p, q.spec)
        out = _Outcome(ans, q.group_tokens + n_in, thought.tokens if thought else 0, first_ms, after_ms,
                       (thought.start, thought.end) if thought else None)
        if ctl.details:
            keyed = self._keyed_none if q.none else self._keyed
            ans["details"] = self._details(
                ctl, t_start, first=keyed(row, first, p0), top=max(p0), first_ms=first_ms, thought=thought,
                after=keyed(row, first, p1) if p1 is not None else None, after_ms=after_ms,
                gate=ctl.gate if ctl.think_mode == "auto" else None)
        return out

    def _keyed(self, row: Row, first: dict | None, p: Sequence[float]) -> dict:
        """Probabilities keyed as in the answer; after a knock-out, over every option (finalists and residual)."""
        if first is None:
            return option_probabilities(row, p)
        ids = [o["id"] for o in row["options"]][:len(p)]
        return knockout_combine(dict(zip(ids, p)), first, self.folder.wide_residual)

    def _none_answer(self, row: Row, first: dict | None, p: Sequence[float], null_ok: bool = True) -> dict:
        """The answer of a choice with the extra last option: probabilities over the listed options renormalised,
        the extra option's probability as none_probability, and with `null_ok` (allow_none) choice null when the extra
        option is the most likely; without it (none_check alone) the most likely listed option."""
        listed, none, abstain = split_none(p)
        return choice_answer(self._keyed(row, first, listed), none, abstain and null_ok)

    def _keyed_none(self, row: Row, first: dict | None, p: Sequence[float]) -> dict:
        """Details probabilities of a read-out with the extra last option: the read-out itself, keyed by option, with
        the extra option under NONE_OPTION (after a knock-out: the listed options over every option, scaled to the
        share the final round gave them)."""
        if first is None:
            return option_probabilities(row, p)
        listed, none, _ = split_none(p)
        out = {c: v * (1 - none) for c, v in self._keyed(row, first, listed).items()}
        out[NONE_OPTION] = none
        return out

    @staticmethod
    def _details(ctl: Controls, t_start: float, first: dict, top: float, first_ms: float,
                 thought: _Thought | None = None, after: dict | None = None, after_ms: float = 0.0,
                 gate: float | None = None) -> dict:
        """The details of one answer (docs/api.md)."""
        info = {"used": thought is not None, "tokens": thought.tokens if thought else 0,
                "ended": thought.ended if thought else "skipped",
                "ms": _ms(thought.end - thought.start) if thought else 0.0}
        if thought:
            info["generation"] = thought.generation
        if ctl.thought_text and thought:
            info["text"] = thought.text
        return {"first_pass": {"probabilities": first, "top": top, "ms": first_ms}, "thought": info,
                "after_thought": {"probabilities": after, "ms": after_ms} if thought else None,
                "gate": gate, "mix": ctl.mix if thought else None, "total_ms": _ms(time.perf_counter() - t_start)}

    def _room(self, msgs: list, req: _Request) -> int:
        """Tokens a thought of the question may take: the context less the thought's prompt (the thinking prompt and
        the forced prefix) and CONTEXT_MARGIN. Not counted from the one-pass prompt: the release template's thinking
        prompt is longer, so the read-out after a thought cut at that room would not fit the context."""
        prompt = think_prompt(self.tok, msgs) + self.auto.prefix
        return self.context_length - count_tokens(self.tok, prompt) - req.extra_tokens - CONTEXT_MARGIN

    async def _think(self, lane: Backend, msgs: list, budget: int, generation: str = "chunked",
                     req: _Request | None = None) -> _Thought:
        """Generate the thought (streamed or in chunks) and build the prompt of the read-out after it."""
        auto = self.auto
        base = think_prompt(self.tok, msgs)
        prompt = base + auto.prefix
        seed = int(hashlib.sha1(prompt.encode("utf-8", "surrogatepass")).hexdigest()[:8], 16)
        run = self._stream if generation == "stream" else self._chunks
        text, tokens, ended, start, end = await run(lane, prompt, budget, seed, (req or _Request()).kw())
        thought = (auto.prefix + text).replace("<think>", "", 1).replace("</think>", "").strip()
        seen = thought
        if auto.tail_chars and len(seen) > auto.tail_chars:
            seen = "[...]\n" + seen[-auto.tail_chars:]
        read_prompt = base + seen + (CUT if ended == "budget" else CLOSE)
        return _Thought(thought, read_prompt, tokens, ended, start, end, generation)

    async def _chunks(self, lane: Backend, prompt: str, budget: int, seed: int, kw: dict) -> tuple:
        """The thought in requests of chunk_tokens tokens (0: one request), the loop detector after each.
        -> (text, tokens, ended, start, end)."""
        auto = self.auto
        text, tokens, ended, rnd = "", 0, "budget", 0
        start = end = time.perf_counter()
        while tokens < budget:
            step = min(auto.chunk_tokens, budget - tokens) if auto.chunk_tokens else budget
            g = await lane.generate(prompt + text, step, auto.sampling, STOP, (seed + 7919 * rnd) % (2 ** 31), **kw)
            end = time.perf_counter()
            text += g.text
            tokens += g.tokens
            if g.finish_reason == "stop":
                ended = "closed"
                break
            if not auto.chunk_tokens or g.tokens == 0:
                break
            if looping(text):
                text = text[:loop_cut(text)]        # keep the reasoning and the start of the loop
                ended = "loop"
                break
            rnd += 1
        return text, tokens, ended, start, end

    async def _stream(self, lane: Backend, prompt: str, budget: int, seed: int, kw: dict) -> tuple:
        """The thought as one streamed request with the seed of the first chunk. The loop detector runs where the
        chunked path runs it: each time another chunk_tokens tokens have arrived, and once more when the budget ends
        the thought (0: never). A looping thought is cut as in the chunked path and the stream is closed, which ends
        the request in the engine. -> (text, tokens, ended, start, end)."""
        auto = self.auto
        every = auto.chunk_tokens
        text, tokens, ended, check = "", 0, "budget", every
        start = time.perf_counter()
        events = lane.generate_stream(prompt, budget, auto.sampling, STOP, seed % (2 ** 31), **kw)
        try:
            async for ev in events:
                text += ev.text
                tokens = ev.tokens
                if ev.finish_reason == "stop":
                    ended = "closed"
                elif every and tokens > 0 and (ev.finish_reason is not None or tokens >= check):
                    while check <= tokens:
                        check += every
                    if looping(text):
                        text = text[:loop_cut(text)]    # keep the reasoning and the start of the loop
                        ended = "loop"
                        break
                if ev.finish_reason is not None:
                    break
        finally:
            close = getattr(events, "aclose", None)
            if close is not None:
                await close()                       # ends the request in the engine when it is still running
        return text, tokens, ended, start, time.perf_counter()
