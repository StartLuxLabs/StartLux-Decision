"""What a model folder says about how to run it.

A StartLux-Decision folder holds the weights, the tokenizer, ``config.json`` and ``decision_config.json``:

* ``temperature_by_type``: read-out temperature per question type (choice, noul, score), fitted on held-out data;
* ``letter_token_ids``: the token ids of A to Z (checked against the tokenizer);
* ``letter_bias`` (optional): 26 additive logit biases, in A to Z order, applied before the read-out temperature;
* ``max_options_per_pass``: 26, one letter per option;
* ``wide_choice``: {group, keep, residual} for choice lists over 26 options;
* ``prompt_attention``: "bidirectional" in Instant folders (their ``config.json`` carries the same value under
  ``jev_prompt_attention``, the key the training pipeline writes and the vLLM plugin reads);
* ``auto_think`` (optional, causal folders only): overrides of the built-in AutoThink defaults, see AutoThink;
* ``format``: "StartLux-Decision-v1" in the folders of the first release (v1).

From ``config.json`` the package also reads whether the folder has a vision tower (``vision_config``; images in the
evidence), its patch and merge sizes (``patch_size`` x ``spatial_merge_size`` pixels per image token side), and its
native context (``max_position_embeddings``, 262,144 tokens in these folders).

The kinds of models:

* Instant (prompt-bidirectional): the prompt is read with bidirectional attention in the full-attention layers, and
  the answer comes from one read-out. These models do not think.
* AutoThink (causal): one read-out first; when its top probability is below the gate, the model writes a thought
  and the decision is read again after it, and the two read-outs are mixed.
* v1 (causal; ``format`` "StartLux-Decision-v1" and no ``auto_think`` block): the first release, trained for one
  read-out. It answers in one pass by default (think_mode "never"); a request may still ask it to think, with the
  AutoThink settings, though v1 was not trained for that. Its prompts are those of v1.1.
"""
from __future__ import annotations

import datetime
import json
import math
import os
from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from .backends.base import Sampling
from .prompt import MAX_OPTIONS, TYPES

PROMPT_ATTENTION_KEY = "jev_prompt_attention"     # config.json key of the prompt attention mode (kept for old folders)
PROMPT_ATTENTION_MODES = ("causal", "bidirectional")
THINK_MODES = ("auto", "never", "always")
THINK_MODE_ALIASES = {"hybrid": "auto", "plain": "never", "think": "always"}
RELEASE_PREFIX = "Let me work through this carefully, step by step.\n"
V1_FORMAT = "StartLux-Decision-v1"                # decision_config.json "format" of the first release


def think_mode(value: Any) -> str:
    """"auto", "never" or "always" (the older names "hybrid", "plain" and "think" are accepted)."""
    if isinstance(value, str):
        value = THINK_MODE_ALIASES.get(value, value)
        if value in THINK_MODES:
            return value
    raise ValueError(f"think_mode must be one of {', '.join(THINK_MODES)} (or hybrid, plain, think), not {value!r}")


@dataclass(frozen=True)
class AutoThink:
    """Settings of the AutoThink path. The defaults are the release settings that produced the published numbers,
    except the gate: the published numbers used 0.9, the default is 0.6, which answers more requests in one pass.
    A folder may override any of them with an ``auto_think`` block in decision_config.json (same names)."""
    think_mode: str = "auto"            # auto: think when the first read-out is unsure; never; always
    gate: float = 0.6                   # think when the first read-out's top probability is below this
    mix: float = 0.75                   # final = (1 - mix) * first read-out + mix * read-out after the thought
    budget: int = 4096                  # most thought tokens
    prefix: str = RELEASE_PREFIX        # forced start of every thought
    tail_chars: int = 4000              # the read-out after the thought sees its last N characters; 0 = all
    chunk_tokens: int = 1024            # thought generated N tokens at a time, loop check in between; 0 = one call
    think_temperature: Mapping[str, float] = field(default_factory=lambda: {t: 1.0 for t in TYPES})
    sampling: Sampling = Sampling(temperature=0.6, top_p=0.95, top_k=20)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _number(block: dict, key: str, lo: float, hi: float) -> float:
    v = block[key]
    if not _is_number(v) or not lo <= v <= hi:
        raise ValueError(f"auto_think.{key} must be a number in [{lo}, {hi}], not {v!r}")
    return float(v)


def _integer(block: dict, key: str, lo: int) -> int:
    v = block[key]
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise ValueError(f"auto_think.{key} must be an integer >= {lo}, not {v!r}")
    return v


def parse_auto_think(block: dict | None) -> AutoThink:
    """The ``auto_think`` block of a decision_config.json (None or {} gives the defaults)."""
    if not block:
        return AutoThink()
    if not isinstance(block, dict):
        raise ValueError("auto_think must be an object")
    known = {f.name for f in fields(AutoThink)}
    unknown = sorted(set(block) - known)
    if unknown:
        raise ValueError(f"unknown auto_think settings {unknown}; known: {sorted(known)}")
    kw: dict = {}
    if "think_mode" in block:
        kw["think_mode"] = think_mode(block["think_mode"])
    for key in ("gate", "mix"):
        if key in block:
            kw[key] = _number(block, key, 0.0, 1.0)
    if "budget" in block:
        kw["budget"] = _integer(block, "budget", 1)
    for key in ("tail_chars", "chunk_tokens"):
        if key in block:
            kw[key] = _integer(block, key, 0)
    if "prefix" in block:
        if not isinstance(block["prefix"], str):
            raise ValueError("auto_think.prefix must be a string")
        kw["prefix"] = block["prefix"]
    if "think_temperature" in block:
        tt = block["think_temperature"]
        if not isinstance(tt, dict) or set(tt) - set(TYPES) or not all(_is_number(v) and v > 0 for v in tt.values()):
            raise ValueError(f"auto_think.think_temperature must map question types {TYPES} to positive numbers")
        kw["think_temperature"] = {k: float(v) for k, v in tt.items()}
    if "sampling" in block:
        s = block["sampling"]
        if not isinstance(s, dict) or set(s) - {"temperature", "top_p", "top_k"}:
            raise ValueError("auto_think.sampling must be an object with temperature, top_p and top_k")
        merged = {**AutoThink().sampling.__dict__, **s}
        t, top_p, top_k = merged["temperature"], merged["top_p"], merged["top_k"]
        if not (_is_number(t) and t > 0 and _is_number(top_p) and 0 < top_p <= 1 and isinstance(top_k, int)
                and not isinstance(top_k, bool) and top_k >= 0):
            raise ValueError("auto_think.sampling needs temperature > 0, 0 < top_p <= 1 and an integer top_k >= 0")
        kw["sampling"] = Sampling(float(t), float(top_p), top_k)
    return AutoThink(**kw)


@dataclass(frozen=True)
class ModelFolder:
    """The run settings of one model folder (see the module docstring)."""
    path: str
    prompt_attention: str
    temperature_by_type: Mapping[str, float]
    letter_token_ids: tuple | None
    wide_group: int
    wide_keep: int
    wide_residual: float
    auto_think: AutoThink | None        # None for Instant folders
    release_date: str
    format: str | None = None           # decision_config.json "format", if any
    v1: bool = False                    # a v1 folder: one pass by default (see the module docstring)
    vision: bool = False                # config.json has a vision tower: images can be part of the evidence
    image_factor: int = 32              # pixels per image token side (patch size x spatial merge size)
    native_context: int | None = None   # max_position_embeddings
    letter_bias: tuple[float, ...] = ()  # optional trained A-to-Z read-out offsets

    @property
    def kind(self) -> str:
        """"instant", "auto_think" or "v1"."""
        if self.prompt_attention == "bidirectional":
            return "instant"
        return "v1" if self.v1 else "auto_think"

    @property
    def bidirectional(self) -> bool:
        return self.prompt_attention == "bidirectional"

    def temperature(self, question_type: str) -> float:
        return self.temperature_by_type.get(question_type, 1.0)


def read_config(directory: str) -> dict:
    """DIRECTORY/config.json ({} when absent)."""
    path = os.path.join(str(directory), "config.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def vision_settings(config: dict) -> tuple[bool, int]:
    """(the folder has a vision tower, pixels per image token side) from config.json."""
    vision = config.get("vision_config")
    if not isinstance(vision, dict):
        return False, 32
    return True, int(vision.get("patch_size", 16)) * int(vision.get("spatial_merge_size", 2))


def native_context(config: dict) -> int | None:
    """max_position_embeddings of the text model (top level or text_config)."""
    n = (config.get("text_config") or {}).get("max_position_embeddings", config.get("max_position_embeddings"))
    return n if isinstance(n, int) and not isinstance(n, bool) and n > 0 else None


def read_prompt_attention(directory: str) -> str:
    """The prompt attention mode in DIRECTORY/config.json ("causal" when the key or the file is absent)."""
    path = os.path.join(str(directory), "config.json")
    if not os.path.isfile(path):
        return "causal"
    with open(path) as f:
        mode = json.load(f).get(PROMPT_ATTENTION_KEY, "causal")
    if mode not in PROMPT_ATTENTION_MODES:
        raise ValueError(f"{path}: {PROMPT_ATTENTION_KEY} must be one of {PROMPT_ATTENTION_MODES}, not {mode!r}")
    return mode


def load_model_folder(path: str) -> ModelFolder:
    """Read and check config.json and decision_config.json of a model folder."""
    cfg_path = os.path.join(path, "decision_config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"{path}: no decision_config.json; expected a local StartLux-Decision model folder")
    with open(cfg_path) as f:
        cfg = json.load(f)
    mode = read_prompt_attention(path)
    declared = cfg.get("prompt_attention", mode)
    if declared != mode:
        raise ValueError(f"{path}: prompt attention is {mode!r} in config.json but {declared!r} in "
                         "decision_config.json")
    temps = cfg.get("temperature_by_type")
    if not isinstance(temps, dict) or not all(k in TYPES and _is_number(v) and v > 0 for k, v in temps.items()):
        raise ValueError(f"{cfg_path}: temperature_by_type must map question types {TYPES} to positive numbers")
    if cfg.get("max_options_per_pass", MAX_OPTIONS) != MAX_OPTIONS:
        raise ValueError(f"{cfg_path}: max_options_per_pass must be {MAX_OPTIONS} (one letter per option)")
    letters = cfg.get("letter_token_ids")
    if letters is not None and (not isinstance(letters, list) or len(letters) != MAX_OPTIONS
                                or not all(isinstance(t, int) for t in letters)):
        raise ValueError(f"{cfg_path}: letter_token_ids must be a list of {MAX_OPTIONS} token ids")
    bias = cfg.get("letter_bias", [0.0] * MAX_OPTIONS)
    if (not isinstance(bias, list) or len(bias) != MAX_OPTIONS
            or not all(_is_number(v) and math.isfinite(v) for v in bias)):
        raise ValueError(f"{cfg_path}: letter_bias must be a list of {MAX_OPTIONS} finite numbers (A to Z)")
    wide = cfg.get("wide_choice") or {}
    group, keep, residual = wide.get("group", 25), wide.get("keep", 3), wide.get("residual", 1e-3)
    if not (isinstance(group, int) and 2 <= group <= MAX_OPTIONS and isinstance(keep, int) and 1 <= keep <= group
            and isinstance(residual, (int, float)) and 0 <= residual < 1):
        raise ValueError(f"{cfg_path}: wide_choice needs 2 <= group <= {MAX_OPTIONS}, 1 <= keep <= group and "
                         "0 <= residual < 1")
    if mode == "bidirectional":
        if "auto_think" in cfg:
            raise ValueError(f"{cfg_path}: an Instant (prompt-bidirectional) folder does not think; remove auto_think")
        auto = None
    else:
        try:
            auto = parse_auto_think(cfg.get("auto_think"))
        except ValueError as e:
            raise ValueError(f"{cfg_path}: {e}") from None
    fmt = cfg.get("format")
    if fmt is not None and not isinstance(fmt, str):
        raise ValueError(f"{cfg_path}: format must be a string")
    v1 = mode == "causal" and fmt == V1_FORMAT and "auto_think" not in cfg
    if v1:
        auto = AutoThink(think_mode="never")    # trained for one pass; thinking only when a request asks for it
    released = cfg.get("release_date")
    if not isinstance(released, str):
        released = datetime.date.fromtimestamp(os.path.getmtime(cfg_path)).isoformat()
    model_config = read_config(path)
    vision, factor = vision_settings(model_config)
    return ModelFolder(path=path, prompt_attention=mode, temperature_by_type={k: float(v) for k, v in temps.items()},
                       letter_token_ids=tuple(letters) if letters else None, wide_group=group, wide_keep=keep,
                       wide_residual=float(residual), auto_think=auto, release_date=released, format=fmt, v1=v1,
                       vision=vision, image_factor=factor, native_context=native_context(model_config),
                       letter_bias=tuple(float(v) for v in bias) if any(bias) else ())


def load_tokenizer(path: str):
    """The folder's Hugging Face tokenizer (needs transformers)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)
