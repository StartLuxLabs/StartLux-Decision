"""Per-request controls of /v1/systemone: optional top-level fields next to ``state`` and ``questions``.

    think_mode       "auto" | "never" | "always" ("hybrid", "plain", "think" are accepted as older names)
    gate             in [0, 1]: think when the first read-out's top probability is below it
    think_budget     integer >= 1: most thought tokens (capped by the server's maximum)
    mix              in [0, 1]: final = (1 - mix) * first read-out + mix * read-out after the thought
    allow_none       bool: choice questions get the extra last option "None of the above", and "choice" is null when
                     it is the most likely one (a question's own "allow_none" overrides it; other types ignore it)
    none_check       bool: choice questions get the extra last option internally and report "none_probability", but
                     "choice" stays a listed option unless allow_none is set (serve turns it on by default)
    return_details   bool: add per-answer details and per-request timings
    return_thought   bool: also return the thought text (implies return_details)
    thought_generation  "auto" | "stream" | "chunked": how a thought is generated (pipeline.py); "auto" streams on
                     engines that stream thoughts safely (vLLM 0.31 or later), "stream" is refused on the others
    first_read       "plain" | "thinking": the prompt of the one-pass read-outs of a request that may think; "plain" is
                     the trained one-pass prompt, "thinking" the thinking prompt with an empty thought, whose beginning
                     a thought reuses from an engine's prefix cache (prompt.empty_thought_prompt). A think_mode "never"
                     request reads "plain"

A missing field takes the server default. Instant models never think: a request that asks them to (think_mode
"auto" or "always") is refused.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from .modelcfg import think_mode

THOUGHT_GENERATIONS = ("auto", "stream", "chunked")
FIRST_READS = ("plain", "thinking")
STREAM_REFUSED = ("this engine does not stream thoughts (vLLM 0.31 or later does): leave thought_generation out or "
                  "set it to \"chunked\"")


class RequestError(ValueError):
    """A request the server refuses (422), with the location of the offending field."""

    def __init__(self, message: str, loc: tuple = ("body",)):
        super().__init__(message)
        self.loc = list(loc)


@dataclass(frozen=True)
class Controls:
    """The effective controls of one request."""
    think_mode: str = "auto"
    gate: float = 0.6
    budget: int = 4096
    mix: float = 0.75
    details: bool = False
    thought_text: bool = False
    thought_generation: str = "auto"
    allow_none: bool = False
    none_check: bool = False
    first_read: str = "plain"


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (math.isfinite(value) and 0 <= value <= 1):
        raise RequestError(f"{name} must be a number from 0 to 1, not {value!r}", ("body", name))
    return float(value)


def _flag(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RequestError(f"{name} must be true or false, not {value!r}", ("body", name))
    return value


def parse_controls(body: dict, defaults: Controls, max_budget: int, can_think: bool,
                   can_stream: bool = False) -> Controls:
    """The controls of a request body on top of the server defaults; raises RequestError for bad values.
    `can_stream`: the engine streams thoughts (else thought_generation "stream" is refused)."""
    out = defaults
    if body.get("think_mode") is not None:
        try:
            out = replace(out, think_mode=think_mode(body["think_mode"]))
        except ValueError as e:
            raise RequestError(str(e), ("body", "think_mode")) from None
        if not can_think and out.think_mode != "never":
            raise RequestError("this is an Instant model: it answers in one pass and does not think; leave think_mode "
                               "out or set it to \"never\" (thinking needs an AutoThink model)",
                               ("body", "think_mode"))
    if body.get("gate") is not None:
        out = replace(out, gate=_fraction(body["gate"], "gate"))
    if body.get("mix") is not None:
        out = replace(out, mix=_fraction(body["mix"], "mix"))
    if body.get("think_budget") is not None:
        b = body["think_budget"]
        if isinstance(b, bool) or not isinstance(b, int) or b < 1:
            raise RequestError(f"think_budget must be an integer of at least 1, not {b!r}", ("body", "think_budget"))
        out = replace(out, budget=min(b, max_budget))
    if body.get("allow_none") is not None:
        out = replace(out, allow_none=_flag(body["allow_none"], "allow_none"))
    if body.get("none_check") is not None:
        out = replace(out, none_check=_flag(body["none_check"], "none_check"))
    if body.get("return_details") is not None:
        out = replace(out, details=_flag(body["return_details"], "return_details"))
    if body.get("return_thought") is not None:
        out = replace(out, thought_text=_flag(body["return_thought"], "return_thought"))
        out = replace(out, details=out.details or out.thought_text)
    if body.get("thought_generation") is not None:
        g = body["thought_generation"]
        if g not in THOUGHT_GENERATIONS:
            raise RequestError(f"thought_generation must be one of {', '.join(THOUGHT_GENERATIONS)}, not {g!r}",
                               ("body", "thought_generation"))
        if g == "stream" and not can_stream:
            raise RequestError(STREAM_REFUSED, ("body", "thought_generation"))
        out = replace(out, thought_generation=g)
    if body.get("first_read") is not None:
        f = body["first_read"]
        if f not in FIRST_READS:
            raise RequestError(f"first_read must be one of {', '.join(FIRST_READS)}, not {f!r}", ("body", "first_read"))
        out = replace(out, first_read=f)
    return out
