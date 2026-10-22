# API

The server speaks the TypeSafe `/v1/systemone` format, so TypeSafe clients work against it. `POST /v1/decisions` is the
same API under OpenRouter's name. Every question of a request runs as its own task: a question that thinks does not hold
up the others, and many requests can be in flight. Serving publicly (API keys, HTTPS, early 429s, the OpenRouter models
document, the access log): [serving_publicly.md](serving_publicly.md).

## POST /v1/systemone (and /v1/decisions)

### Request

```json
{
  "state": "any text, or any JSON value",
  "questions": {
    "team":     {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "shipping": null}},
    "urgent":   {"type": "noul", "instructions": "Answer today?"},
    "severity": {"type": "score", "instructions": "How bad?", "criteria": ["minor", "major", "critical"]}
  },
  "images": ["data:image/jpeg;base64,/9j/4AAQ..."],
  "think_mode": "auto", "gate": 0.6, "think_budget": 4096, "mix": 0.75, "thought_generation": "auto",
  "first_read": "plain", "allow_none": false, "none_check": true, "return_details": false, "return_thought": false
}
```

| question type | criteria | answer |
|---|---|---|
| `choice` | `{option id: description or null}`, or a list of option ids | `choice`, `confidence`, `probabilities` over every option |
| `noul` (yes/no; `bool` is accepted) | optional `{"true": description, "false": description}` | `noul`: the probability of yes |
| `score` | a list of 2 to 10 levels, lowest first, or `{key: level}` with numeric keys (sorted ascending) | `score` (probability-weighted mean level), `confidence`, `legend`, `probabilities` keyed `"0"` to `"n-1"` |

`instructions` is required for choice and score questions; a yes/no question without it asks "Which answer fits the
evidence?". A JSON state is rendered as compact JSON; arrays of 8 or more elements carry an `_index` on each element.

Images (optional, as in v1): `images` is a list of base64 strings or data URIs. In the state, `<image>` marks where
each image goes, one mark per image, in order (a JSON state may hold the marks in its strings). Without marks the
images come first (`Image 1:`, `Image 2:` ... on their own lines when there are several), then a blank line and the
state. Each image is resized to at most 1,048,576 pixels and at least 65,536, keeping its aspect ratio, with both
sides multiples of 32 pixels, and becomes one token per 32 × 32 pixels (a 1024 × 1024 photo is 1,024 tokens). Every
question of the request reads every image, and `usage.input_tokens` counts the image tokens. At most 8 images per
request by default (`serve --max-images`, `--max-pixels`; `--no-images` serves text only). Images need a folder with
a vision tower and an engine that reads them: vLLM 0.31 or later, transformers, llama.cpp when llama-server has an
mmproj file ([engines.md](engines.md#llamacpp)), or MLX with an mlx-vlm folder. Any other engine answers 422.

Long states: a prompt can be as long as the engine's context. On vLLM 0.31 that is 262,144 tokens (the models' native
context) for AutoThink and v1 folders and 131,072 tokens for Instant folders, unless the server was started with a
lower `--max-model-len`. A longer prompt gets a 422 with the engine's reason. AutoThink and v1 folders read a long
state shared by the questions of a request once; an Instant folder reads it again for every question, because it
reads each prompt bidirectionally.

Choice lists: one pass reads up to 26 options (letters A to Z). A list of more options is knocked out: near-equal
groups of at most 25 are read in one pass each, the 3 best of every group go to a final round (the 26 best overall when
more than 26 qualify), and the options that miss the final share 0.001 of the probability in proportion to their group
round. The final round is an ordinary question (it may think). A choice with a single option is answered without the
model (probability 1).

None of the above (`choice` questions only). `serve` checks every choice question for a missing answer by default
(`none_check`): the question gets one extra last option, "None of the above", the answer carries `none_probability`,
and `choice` stays the most likely listed option. On 51,506 benchmark questions this left accuracy unchanged (84.66 with
and without it, thinking on). `"none_check": false` in a request, or `serve --no-none-check`, turns it off. With
`"allow_none": true` (opt-in) a choice question also gets one extra last
option, "None of the above", shown to the model like any other option (and visible to the thought, which can choose
it); the read-out covers n + 1 options. The answer then has `choice` null when that option is the most likely one (a tie
goes to the listed option), `probabilities` over your options only, renormalised to sum to 1 (a client that ignores the
new field sees an ordinary distribution), and `none_probability`, the probability of the extra option before the
renormalisation. `confidence` is computed as usual from the renormalised probabilities, so it describes the spread
over your options, and `none_probability` is the signal to act on. A choice with a single option is read by the model
when `allow_none` is on (two options), not answered at once. In a knocked-out list the extra option is added in the
final round only, and a list of 26 options is knocked out too (the 26th letter is the extra option). An option of your
own called "None of the above" gets a 422. With `allow_none` off, which is the default unless the server runs with `--allow-none`, nothing changes: the prompts,
answers and scores are exactly those without the control. `allow_none` can also be set on one question
(`{"type": "choice", ..., "allow_none": true}`), which overrides the request; `null` means not given. Other question
types ignore it.

Reading `none_probability`: it is an alarm score, not a calibrated probability, and 0.2 is the threshold we suggest.
On 51,506 choice questions of DecisionPivot and Decision Index 0.3 (27B AutoThink, thinking on), each asked as
published and once more with its right option removed, a threshold of 0.2 caught 66.3% of the questions without a
right option and flagged 2.2% of the published ones (AUROC 0.920). [none_probability.md](none_probability.md) has the
other thresholds, why 0.2, the tasks where it is not reliable and how it interacts with thinking. `confidence` alone
separates the two kinds of question much worse, because hard questions with a right option also read low. The extra
option also gives a thought a way out: without it, a question that has no fitting option often makes the thought go
round in circles until the budget, since the model must still pick a listed option.

`confidence` follows TypeSafe: for a choice (p_max − 1/n) / (1 − 1/n), clamped to [0, 1]; for a score 1 minus the
probability-weighted distance in levels from the most likely level, divided by the same distance for an even spread
measured from the middle level, floored at 0. Yes/no answers have no confidence; |2p − 1| is the equivalent.

### Controls

| field | values | effect |
|---|---|---|
| `think_mode` | `auto` / `never` / `always` (also `hybrid` / `plain` / `think`) | `auto`: think when the first read-out's top probability is below `gate` |
| `gate` | number in [0, 1] | the threshold of `auto` |
| `think_budget` | integer ≥ 1 | most thought tokens; larger values are capped by the server's maximum (`serve --max-think-budget`, default: the default budget) |
| `mix` | number in [0, 1] | final = (1 − mix) · first read-out + mix · read-out after the thought; 0 ignores the thought, 1 uses only the read-out after it |
| `thought_generation` | `auto` / `stream` / `chunked` | how a thought is generated. `stream`: one streamed request; the loop detector looks at the thought each time another 1,024 tokens have arrived and ends the request when it loops. `chunked`: requests of 1,024 tokens, each re-sending the prompt and the thought so far, with the loop detector between them. `auto`: `stream` when the engine streams thoughts (vLLM 0.31 or later), else `chunked`. `stream` on another engine gets a 422 |
| `first_read` | `plain` / `thinking`, default `plain` (`serve --first-read`) | the prompt of the one-pass read-outs of a request that may think. `plain`: the trained one-pass prompt. `thinking`: the thinking prompt with an empty thought; a thought then continues the read-out's prompt, so an engine with a prefix cache reads only the thought's own part, where after `plain` it reads the whole state again (the release template starts a thinking prompt's system message with a reasoning-effort line). It is not the trained prompt and moves answers that are not thought about. A `think_mode` `never` request reads `plain` |
| `allow_none` | bool, default false (true with `serve --allow-none`) | choice questions get the extra last option "None of the above" (see above); a question's own `allow_none` overrides it; ignored by `noul` and `score` questions; a non-boolean gets a 422 |
| `none_check` | bool, default true in `serve` | choice questions get the extra option internally and report `none_probability`; `choice` stays a listed option unless `allow_none` is set; a non-boolean gets a 422 |
| `return_details` | bool | adds `details` to every answer and `timing_ms` to the response |
| `return_thought` | bool | also returns the thought text; implies `return_details` |

A missing or null field takes the server default. The server defaults are the model folder's (built-in AutoThink
defaults, or an `auto_think` block in its `decision_config.json`; v1 folders answer in one pass, `never`), then the
`serve` options. Instant models never think:
`think_mode` `auto` or `always` gets a 422; `gate`, `mix` and `think_budget` are accepted and have no effect. A thought
also needs room in the engine's context: the budget is cut to the context length minus the prompt minus 64 tokens, and
when that leaves 32 tokens or fewer the question is answered from the first read-out.

Long thoughts: a thought that runs to the budget is the slowest part of a request. One thought is generated at about
27 tokens/s on an A800 in BF16, so 4,096 tokens take about 150 s. `details.thought.ended` is `budget` for such a thought.
On the 10,000 questions above, 4% of the questions ran to the budget and used over a third of all thought tokens. Most
were hard questions, where the long thought still pays (38% answered right against 22% from the first read-out
alone). Questions without a fitting option run to the budget more often (5.9% against 3.9% with `allow_none` on), and
there the long thought gains nothing. Three things keep thoughts short on such questions. Ask
only the questions that apply to the current step (an agent that has decided to scroll should not also ask which link
to click). Add options such as "not visible" or "other" where the case may fall outside the list. Turn on `allow_none`
where the list may be incomplete. Lowering `think_budget` makes every hard question pay: on DecideBench the 27B
AutoThink model scores 88.6 with 4,096 tokens, 80.4 with 512 and 68.4 with 256, against about 70 when it answers in one
pass (`never`).

### Response

```json
{
  "answers": {
    "team": {"type": "choice", "choice": "billing", "confidence": 0.81, "probabilities": {"billing": 0.905, "shipping": 0.095}},
    "urgent": {"type": "noul", "noul": 0.64},
    "severity": {"type": "score", "score": 1.3, "confidence": 0.19, "legend": {"0": "minor", "1": "major", "2": "critical"},
                 "probabilities": {"0": 0.12, "1": 0.46, "2": 0.42}}
  },
  "usage": {"input_tokens": 290, "output_tokens": 0},
  "model": "StartLux-Decision-4B"
}
```

With `allow_none` a choice answer also carries `none_probability` and may have `"choice": null`:

```json
"team": {"type": "choice", "choice": null, "confidence": 0.33, "probabilities": {"billing": 0.667, "shipping": 0.333},
         "none_probability": 0.7}
```

`usage.input_tokens` counts the tokens of every one-pass prompt read (all rounds of a knocked-out list);
`usage.output_tokens` counts thought tokens.

### Details and timings

With `return_details`, every answer carries:

```json
"details": {
  "first_pass":    {"probabilities": {"billing": 0.55, "shipping": 0.45}, "top": 0.55, "ms": 21.4},
  "thought":       {"used": true, "tokens": 812, "ended": "closed", "ms": 2140.7, "generation": "stream",
                    "text": "Let me work through ..."},
  "after_thought": {"probabilities": {"billing": 0.97, "shipping": 0.03}, "ms": 24.9},
  "gate": 0.6,
  "mix": 0.75,
  "total_ms": 2188.3
}
```

| field | meaning |
|---|---|
| `first_pass.probabilities` | the first read-out, keyed as the answer (after a knock-out: over every option); with `allow_none` it also has the key `None of the above` and sums to 1 over all of them, as read (not renormalised) |
| `first_pass.top` | its top probability over the options read in the final pass: the value compared with the gate |
| `first_pass.ms` | time of the engine calls of the first pass (the knock-out rounds and the final read-out) |
| `thought.used`, `thought.tokens` | whether the question thought, and the thought tokens generated |
| `thought.ended` | `closed` (the model closed the thought), `loop` (the loop detector ended it), `budget` (cut: budget or context reached; the read-out then sees the line "I have to decide now with what I have worked out so far."), `skipped` (no thought) |
| `thought.ms` | from sending the thought request (the first chunk) to receiving its end (the last chunk; wave waits included) |
| `thought.generation` | `stream` or `chunked` (see the controls); only when the question thought |
| `thought.text` | with `return_thought`: the whole thought, forced start included; the read-out after it sees its last 4,000 characters |
| `after_thought` | the read-out after the thought, re-tempered, and the time of its engine call; null without a thought |
| `gate` | the gate the first read-out was compared with; null when it was not consulted (`never`, `always`, Instant, single-option lists) |
| `mix` | the mixture weight applied; null without a thought |
| `total_ms` | the question's whole pipeline |

The response also gets `timing_ms`: `wall` (the request inside the server, from validation to the answers),
`first_pass_sum` and `after_thought_sum` (sums over the questions of their read-out times; questions run concurrently,
so these can exceed `wall`), and `thought_span` (from the first thought's first chunk to the last thought's last
chunk; 0 without a thought).

### Errors

Every response carries `x-request-id` (the client's own when it sent a valid one, up to 128 characters of
`A-Z a-z 0-9 . _ : -`) and the same value in `x-typesafe-request-id`. Errors are JSON:

```json
{"error": "gate must be a number from 0 to 1, not 2", "detail": [{"loc": ["body", "gate"], "msg": "...", "type": "value_error"}]}
```

| status | when |
|---|---|
| 401 | the server has API keys and the request has no `Authorization: Bearer <key>` with one of them (every route but `/health`) |
| 429 | the server is at `--max-inflight` decision requests; answered at once with `Retry-After: 1` |
| 422 | the body is not JSON or not an object; a control has a bad value (also `allow_none` that is not true or false, on the request or on a choice question); a question cannot be rendered (no instructions, 11 score levels, duplicate ids, ...); an Instant model is asked to think; `images` that cannot be read, more images than allowed, `<image>` marks that do not match the images, or images on an engine or folder that reads text only; the engine refuses a question (prompt longer than its context) |
| 404, 405 | unknown path or method |
| 502 | the engine failed or could not be reached (four tries per call) |

A 401 or 422 is final; a 429 or 502 may be retried. A `model` field in the request is accepted and ignored; the
response's `model` is the server's name.

## Model folders

A model folder holds the weights, the tokenizer, `config.json` and `decision_config.json`. The package reads:

| `decision_config.json` key | meaning |
|---|---|
| `temperature_by_type` | read-out temperature per question type (`choice`, `noul`, `score`; a missing type uses 1.0) |
| `letter_token_ids` | token ids of A to Z, checked against the tokenizer |
| `max_options_per_pass` | 26 (anything else is refused) |
| `wide_choice` | `{"group": 25, "keep": 3, "residual": 0.001}`: the knock-out of choice lists over 26 options |
| `prompt_attention` | `"bidirectional"` in Instant folders; must agree with `jev_prompt_attention` in `config.json` |
| `release_date` | reported by `/v1/models` (optional) |
| `auto_think` | optional, causal folders only: overrides of the AutoThink defaults below |

The folder's kind follows from `config.json`: `"jev_prompt_attention": "bidirectional"` is an Instant folder, anything
else is served as AutoThink. A causal folder trained for one pass only can carry `"auto_think": {"think_mode":
"never"}` (or be served with `--think-mode never`).

| `auto_think` key | default | meaning |
|---|---|---|
| `think_mode` | `"auto"` | default of the request field |
| `gate` | 0.6 | default of the request field (the published numbers used 0.9) |
| `mix` | 0.75 | default of the request field |
| `budget` | 4096 | default `think_budget` |
| `prefix` | `"Let me work through this carefully, step by step.\n"` | forced start of every thought |
| `tail_chars` | 4000 | the read-out after the thought sees its last N characters, marked `[...]`; 0 = all |
| `chunk_tokens` | 1024 | the thought is generated N tokens at a time with the loop check in between; 0 = one call, no loop check |
| `think_temperature` | `{"choice": 1.0, "noul": 1.0, "score": 1.0}` | the read-out after the thought is re-tempered from the type temperature to this one; a type left out is not re-tempered |
| `sampling` | `{"temperature": 0.6, "top_p": 0.95, "top_k": 20}` | sampling of the thought |

Unknown keys and out-of-range values are refused when the server starts.

## GET /v1/models

`{"models": [{"name", "description", "release_date"}]}`. `release_date` comes from `decision_config.json`
(`release_date`), else the date of that file.

## GET /health

Status, model name and kind (`instant`, `auto_think` or `v1`), whether the server reads images, prompt attention, the
engine and its settings, the default controls, the largest think budget a request may ask for, and the context length
used for the room check. A server with
API keys returns only the status, model name and kind to a request without a valid key, so that the addresses and
settings of its engines stay private.

## Python

```python
import asyncio
from startlux_decide.backends.vllm import VLLMBackend
from startlux_decide.modelcfg import load_model_folder, load_tokenizer
from startlux_decide.pipeline import Decider

async def main():
    folder = load_model_folder("/models/StartLux-Decision-4B")
    engine = VLLMBackend(["http://127.0.0.1:8001"], load_tokenizer(folder.path))
    await engine.start()
    decider = Decider(folder, engine)
    result = await decider.decide(state, questions, decider.controls({"return_details": True}))
    print(result.answers, result.usage, result.timing_ms)
    await engine.close()

asyncio.run(main())
```
