"""The prompt format the StartLux-Decision models were trained on.

A question about a state becomes one chat prompt: a fixed system line, then an Evidence / Question / Options block with
lettered options. The answer is read from the next-token probabilities of the option letters right after the prompt, so
nothing has to be generated for a one-pass answer. Every string in this module is part of the trained format: changing
any of them changes the answers.

Rules worth knowing when you build requests by hand:

* an option without a description is shown by its id alone;
* yes/no questions (``noul``) are shown as the options "yes" and "no";
* choice ids that are bare letters or numbers ("A", "2", "option 3") are hidden when every option has a description,
  because a line such as "A) C: Paris" would make the answer letter ambiguous;
* options are shown in the order the request lists them; score levels are always lowest first;
* a choice question with ``allow_none`` gets one extra last option, "None of the above" (``NONE_OPTION``), shown like
  any other option; the bare-id rule above looks at the listed options only.
"""
from __future__ import annotations

import json
import re
import string
from typing import Any

SYSTEM = "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only."
LETTERS = string.ascii_uppercase
TYPES = ("choice", "score", "noul")
MAX_OPTIONS = 26          # one letter per option, A to Z
MAX_LEVELS = 10           # score levels
ANNOTATE_MIN = 8          # JSON arrays at least this long carry an "_index" on every element
NOUL_INSTRUCTIONS = "Which answer fits the evidence?"
NONE_OPTION = "None of the above"   # the extra last option of a choice question with allow_none
THINK_OPEN = "<think>"
THINK_OFF_SUFFIX = "<think>\n\n</think>\n\n"   # the end of every one-pass prompt (thinking switched off)
_BARE = re.compile(r"^(?:\(?[A-Za-z][\).]?|\(?\d{1,3}[\).]?|option[_ ]?\d{1,3}|opt[_ ]?\d{1,3})$", re.I)

Row = dict   # a rendered question: {"type", "state", "instructions", "options": [{"id", "criterion"}]}, plus
             # "none_option": True when its last option is the extra NONE_OPTION


def annotate_indices(x: Any, min_len: int = ANNOTATE_MIN) -> Any:
    """Long arrays carry their element index, so that the model can refer to items by position."""
    if isinstance(x, list):
        if len(x) >= min_len:
            return [({"_index": i, **annotate_indices(v, min_len)} if isinstance(v, dict)
                     else {"_index": i, "value": annotate_indices(v, min_len)}) for i, v in enumerate(x)]
        return [annotate_indices(v, min_len) for v in x]
    if isinstance(x, dict):
        return {k: annotate_indices(v, min_len) for k, v in x.items()}
    return x


def state_text(state: Any) -> str:
    """The evidence as text: a string as it is, any other JSON value as compact JSON with array indices."""
    if isinstance(state, str):
        return state
    if state is None or state == {} or state == []:
        return ""
    return json.dumps(annotate_indices(state), ensure_ascii=False)


def value_text(v: Any) -> str | None:
    """An option description or instruction as text (None when empty)."""
    if v is None:
        return None
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = s.strip()
    return s or None


def _norm(s: Any) -> str:
    return re.sub(r"[\s_\-]+", " ", str(s)).strip().lower()


def validate_row(row: Row) -> list[str]:
    """Check a rendered question and return its option ids; raises ValueError with what is wrong."""
    t = row.get("type")
    if t not in TYPES:
        raise ValueError(f"unknown question type {t!r}")
    if not isinstance(row.get("state"), str):
        raise ValueError("the state must be a string after rendering")
    if not isinstance(row.get("instructions"), str) or not row["instructions"].strip():
        raise ValueError("instructions are required for choice and score questions")
    opts = row.get("options")
    if not isinstance(opts, list) or not 2 <= len(opts) <= MAX_OPTIONS:
        got = len(opts) if isinstance(opts, list) else opts
        raise ValueError(f"a question needs 2 to {MAX_OPTIONS} options in one pass, got {got!r}")
    ids = []
    for o in opts:
        i = o.get("id")
        if not isinstance(i, str) or not i.strip() or "\n" in i or "\r" in i or len(i) > 200:
            raise ValueError(f"bad option id {i!r}: ids must be non-empty single-line strings of at most 200 "
                             "characters")
        c = o.get("criterion")
        if c is not None and not isinstance(c, str):
            raise ValueError("an option description must be a string or null")
        ids.append(i)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate option id")
    if t == "noul" and set(ids) != {"true", "false"}:
        raise ValueError("yes/no options must be 'true' and 'false'")
    if t == "score" and not 2 <= len(ids) <= MAX_LEVELS:
        raise ValueError(f"a score question needs 2 to {MAX_LEVELS} levels, got {len(ids)}")
    return ids


def option_lines(row: Row) -> list[str]:
    """The lettered option lines of a rendered question, in the order of its options."""
    order = [o["id"] for o in row["options"]]
    crit = {o["id"]: o.get("criterion") for o in row["options"]}
    t = row["type"]
    extra = len(order) - 1 if row.get("none_option") else None      # position of the extra option
    listed = order if extra is None else order[:extra]
    hide = t == "choice" and all(_BARE.match(i) for i in listed) and all(crit[i] for i in listed)
    lines = []
    for k, i in enumerate(order):
        name = ("yes" if i == "true" else "no") if t == "noul" else i
        c = crit[i]
        if hide and k != extra:
            body = c
        elif c is None or not c.strip() or _norm(c) == _norm(name):
            body = name
        else:
            body = f"{name}: {c}"
        lines.append(f"{LETTERS[k]}) {body}")
    return lines


def messages(row: Row) -> list[dict]:
    """The chat messages (system and user turn) of a rendered question."""
    validate_row(row)
    state = row["state"] if row["state"].strip() else "(none)"
    content = ("Evidence:\n" + state + "\n\nQuestion: " + row["instructions"].strip() + "\nOptions:\n"
               + "\n".join(option_lines(row)))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


_PROBE = {"type": "noul", "state": "s", "instructions": "q",
          "options": [{"id": "true", "criterion": None}, {"id": "false", "criterion": None}]}


def check_tokenizer(tokenizer: Any) -> list[int]:
    """Check that the tokenizer renders the thinking-off prompt and reads every letter as one token right after it.
    Returns the 26 letter token ids, A to Z."""
    text = tokenizer.apply_chat_template(messages(_PROBE), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    if not text.endswith(THINK_OFF_SUFFIX):
        raise ValueError("the tokenizer's chat template lacks the thinking-off assistant prefix "
                         f"{THINK_OFF_SUFFIX!r}; this is not a StartLux-Decision tokenizer")
    ids = tokenizer.encode(text, add_special_tokens=False)
    letters = []
    for letter in LETTERS:
        t = tokenizer.encode(letter, add_special_tokens=False)
        if len(t) != 1 or tokenizer.encode(text + letter, add_special_tokens=False) != ids + t:
            raise ValueError(f"the letter {letter} is not a separate single token after the prompt")
        letters.append(t[0])
    if len(set(letters)) != len(LETTERS):
        raise ValueError("the letter token ids are not distinct")
    return letters


# ---------------------------------------------------------------- /v1/systemone questions

def score_keys(criteria: dict) -> list:
    """Level order of a score question given as {key: description}: numeric keys ascending, otherwise as given."""
    try:
        return sorted(criteria, key=lambda k: float(k))
    except (TypeError, ValueError):
        return list(criteria)


def from_systemone(state: Any, spec: dict, allow_none: bool = False) -> Row:
    """One /v1/systemone question -> the rendered question. Score level ids are "0" to "n-1", lowest first.
    `allow_none`: a choice question gets the extra last option NONE_OPTION (other types ignore it)."""
    t = spec.get("type", "choice")
    t = "noul" if t == "bool" else t
    crit = spec.get("criteria", spec.get("options"))
    ins = value_text(spec.get("instructions", spec.get("question"))) or ""
    if t == "choice":
        if isinstance(crit, (list, tuple)):
            crit = {str(c): None for c in crit}
        if not isinstance(crit, dict):
            raise ValueError("a choice question needs criteria: {option: description or null} or a list of options")
        opts = [{"id": str(k), "criterion": value_text(v)} for k, v in crit.items()]
        if allow_none:
            if any(_norm(o["id"]) == _norm(NONE_OPTION) for o in opts):
                raise ValueError(f'allow_none adds the option "{NONE_OPTION}" itself: remove the option of that '
                                 "name or leave allow_none out")
            opts.append({"id": NONE_OPTION, "criterion": None})
    elif t == "score":
        if isinstance(crit, dict):
            crit = [crit[k] for k in score_keys(crit)]
        if not isinstance(crit, (list, tuple)):
            raise ValueError("a score question needs criteria: a list of levels, lowest first")
        opts = [{"id": str(i), "criterion": value_text(c)} for i, c in enumerate(crit)]
    elif t == "noul":
        c = crit if isinstance(crit, dict) else {}
        tr, fa = c.get("true", c.get(True)), c.get("false", c.get(False))
        opts = [{"id": "true", "criterion": value_text(tr)}, {"id": "false", "criterion": value_text(fa)}]
        if not ins:
            ins = NOUL_INSTRUCTIONS
    else:
        raise ValueError(f"unknown question type {t!r}: use choice, noul or score")
    row = {"type": t, "state": state_text(state), "instructions": ins, "options": opts}
    if t == "choice" and allow_none:
        row["none_option"] = True
    return row


# ---------------------------------------------------------------- prompts

def _render(tokenizer: Any, msgs: list[dict], thinking: bool) -> str:
    """The chat template over the messages. A question with images (images.SLOT in its state) is rendered with
    content parts, so the template writes its image placeholder at each image's place, as v1 does."""
    from .images import IMAGE_PLACEHOLDER, SLOT, multimodal_messages
    slots = sum(m["content"].count(SLOT) for m in msgs if isinstance(m.get("content"), str))
    chat = multimodal_messages(msgs) if slots else msgs
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    if slots and text.count(IMAGE_PLACEHOLDER) != slots:
        raise ValueError(f"the chat template wrote {text.count(IMAGE_PLACEHOLDER)} image placeholders for {slots} "
                         "images; this folder's template does not render images as the package expects")
    return text


def plain_prompt(tokenizer: Any, msgs: list[dict]) -> str:
    """The one-pass prompt: chat template with thinking off, ending with the empty think block."""
    return _render(tokenizer, msgs, False)


def think_prompt(tokenizer: Any, msgs: list[dict]) -> str:
    """The thinking prompt without the forced prefix: chat template with thinking on, ending with the opened think
    block. The thought (forced prefix first) is generated after it, and the read-out after the thought starts with
    it."""
    text = _render(tokenizer, msgs, True)
    if not text.rstrip().endswith(THINK_OPEN):
        text += THINK_OPEN + "\n"
    return text


def empty_thought_prompt(tokenizer: Any, msgs: list[dict]) -> str:
    """A one-pass prompt that a thought continues: the thinking prompt with an empty thought, ending with the empty
    think block like plain_prompt. The release chat template opens the system message of a thinking prompt with a
    reasoning-effort line that plain_prompt lacks, so a thought shares only a few characters with plain_prompt and an
    engine reads its whole state again; it shares this prompt up to the think block. Not the trained one-pass prompt:
    see the first_read control (controls.py)."""
    text = think_prompt(tokenizer, msgs) + "\n</think>\n\n"
    if not text.endswith(THINK_OFF_SUFFIX):
        raise ValueError("the chat template's thinking prompt does not end with the opened think block "
                         f"{THINK_OPEN!r} and a newline, which first_read \"thinking\" closes")
    return text


def check_thinking_prompt(tokenizer: Any) -> None:
    """Check that the tokenizer's thinking prompt takes the empty thought of empty_thought_prompt; raises ValueError.
    Like the thinking-off prompt of check_tokenizer, its end does not depend on the question, so a folder that thinks
    checks it once, before any request."""
    empty_thought_prompt(tokenizer, messages(_PROBE))


def count_tokens(tokenizer: Any, text: str) -> int:
    """Number of tokens of a prompt, as the backends see it."""
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])
