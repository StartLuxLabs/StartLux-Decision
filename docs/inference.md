# Inference

## Install

```bash
hf download startlux-models/StartLux-Decision-4B --local-dir StartLux-Decision-4B      # or any size, see below
# or, from ModelScope: modelscope download StartLuxAI/StartLux-Decision-4B --local-dir StartLux-Decision-4B
# AMD ROCm: follow the dedicated section below instead of this generic install command.
pip install -r requirements.txt
```

The six models are in the [StartLux-Decision collection](https://huggingface.co/collections/startlux-models/startlux-decision-6abba92b301b573fa154d493) on Hugging Face: StartLux-Decision-0.8B, 2B, 4B, 9B, 27B and 35B-A3B, under
`startlux-models/`. The same repositories are on ModelScope under [`StartLuxAI/`](https://modelscope.cn/organization/StartLuxAI). The examples below use a
local folder called `StartLux-Decision-4B`.

`requirements.txt` includes `flash-linear-attention` and `causal-conv1d`. They matter more than anything else on this
page. The models use linear-attention layers, and without these two packages transformers quietly falls back to a
plain PyTorch implementation that is more than ten times slower. Nothing errors; it is just slow. `causal-conv1d`
builds against your CUDA or ROCm runtime and PyTorch, and if pip ends up compiling it, add `--no-build-isolation`. On
Apple Silicon both are skipped and mlx-lm is installed instead; see [Apple Silicon (MLX)](#apple-silicon-mlx).

Check that the fast path is really on:

```bash
python -m startlux_decision.check StartLux-Decision-4B        # fast kernels: active
```

On a machine with a GPU it must say `active`. On NVIDIA CUDA or AMD ROCm, `StartLuxDecision(...)` refuses to start when
the kernels are not active; set `STARTLUX_ALLOW_SLOW=1` if you really want to run without them. On a CPU-only machine
the check is skipped and everything runs, slowly.

### AMD ROCm

The torch backend supports AMD GPUs through ROCm. PyTorch deliberately uses the `cuda` device name and
`torch.cuda` API on ROCm too, so keep `--device cuda`; do not pass `rocm`.

Install a supported ROCm PyTorch build first, following AMD's
[current PyTorch installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html).
Then verify the runtime and a real operation before installing the fused kernels:

```bash
python - <<'PY'
import torch
assert torch.version.hip, f"expected ROCm torch, got {torch.__version__}"
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print(torch.__version__, "HIP", torch.version.hip, torch.cuda.get_device_name(), torch.cuda.get_arch_list())
x = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
assert torch.isfinite(y).all()
PY
```

Official wheel-packaged `rocm/pytorch` images keep the development SDK inside site-packages. Set its path before building
the extension; a native Core SDK install normally discovers `/opt/rocm` and does not need this override.

```bash
export ROCM_HOME="$(python -c 'from importlib.metadata import distribution; print(distribution("rocm-sdk-devel").locate_file("_rocm_sdk_devel"))')"
```

Then install the remaining requirements:

```bash
# causal-conv1d builds a HIP extension against the installed torch when a matching wheel is unavailable.
python -m pip install --no-build-isolation -r requirements.txt
python -m startlux_decision.check StartLux-Decision-4B
```

The check must report `accelerator: rocm` and `fast kernels: active`. The ROCm 10.1 / PyTorch 2.12 official image is a
known working combination. The requirements retain Transformers 5.8.1 because the inference code uses that release's
linear-attention cache and fast-kernel interfaces. `causal-conv1d` 1.7.0 does not currently build against PyTorch 2.14,
whose extensions require C++20, so use a compatible PyTorch release until that package updates. Set `HIP_ARCHITECTURES`
to the target gfx name when building for deployment on a different GPU.

For a multi-GPU host, expose one device to each server with `ROCR_VISIBLE_DEVICES`. First use can spend minutes compiling
and autotuning Triton kernels; later starts reuse its cache.

```bash
ROCR_VISIBLE_DEVICES=0 python -m startlux_decision.server --model StartLux-Decision-4B --port 8090
```

PyTorch implements HIP graphs through the same `torch.cuda.CUDAGraph` API used here. Set `STARTLUX_GRAPHS=0` to separate
graph capture from fused-kernel problems while diagnosing a new GPU. An `invalid device function` error means the
installed extension lacks code for that gfx target; rebuild the extension for the reported architecture rather than
enabling `STARTLUX_ALLOW_SLOW`.

## Serving

```bash
python -m startlux_decision.server --model StartLux-Decision-4B --port 8090
curl -s localhost:8090/health        # {"status": "ok", "model": "StartLux-Decision-4B", "fast_kernels": true}
```

The server speaks the TypeSafe `/v1/systemone` format and serves one request at a time per GPU. Run one server per GPU
and put a load balancer in front if you need more throughput. The official TypeSafe SDK works against it:

```bash
export TYPESAFE_BASE_URL=http://127.0.0.1:8090
export TYPESAFE_API_KEY=unused
```

## Request and response

A request has a `state` (a string or any JSON value) and `questions`, each with a `type`:

<div align="center">

| type | criteria | answer |
|:---:|:---:|:---:|
| `choice` | `{option: description or null}` | `choice`, `confidence`, `probabilities` over the options |
| `noul` (yes/no) | optional `{"true": ..., "false": ...}` | `noul` = probability of yes |
| `score` | a list of levels, lowest first | `score` (probability-weighted mean level), `confidence`, `legend`, `probabilities` keyed `"0".."n-1"` |

</div>

For the support-ticket request in the README, StartLux-Decision-4B answers (numbers rounded here):

```json
{"answers": {
   "team": {"type": "choice", "choice": "billing", "confidence": 0.816,
            "probabilities": {"billing": 0.877, "shipping": 0.005, "technical": 0.118}},
   "urgent": {"type": "noul", "noul": 0.635},
   "severity": {"type": "score", "score": 1.335, "confidence": 0.002,
                "legend": {"0": "cosmetic", "1": "annoying", "2": "blocks the customer"},
                "probabilities": {"0": 0.116, "1": 0.433, "2": 0.451}}},
 "usage": {"input_tokens": 290, "output_tokens": 0}, "model": "StartLux-Decision-4B", "latency_ms": 15.7}
```

Each question is rendered as its own prompt with the full state, options lettered A, B, C and so on in the order you
give them, and the answer is read from the logits of those letters. A choice question can have up to 26 options in one
pass. Longer lists are split into near-equal groups of up to 25, the top three of each group go to a final round, and
the options that miss the final keep a small share of probability in proportion to their group score, so every option
still gets a non-zero probability.

Temperatures are per question type and live in `decision_config.json`. Changing them never changes which option wins.

`confidence` follows TypeSafe's definitions, so code that thresholds it behaves the same against either service. For a
choice it is (p_max − 1/n) / (1 − 1/n): 0 for an even split over the n options, 1 when one option has all the
probability. For a score it is 1 minus the probability-weighted distance, in levels, from the most likely level,
divided by the same distance for an even spread measured from the middle level, and floored at 0: probability on a
neighbouring level lowers it less than probability at the far end. Yes/no answers have no `confidence`; |2p − 1| is the
equivalent. Until 2026-10-03 the package returned the top probability as `confidence`; it is still in `probabilities`.
Every response carries an `x-typesafe-request-id` header, `GET /v1/models` lists the model with its release date, and
an error comes back as `{"error": message, "detail": [{"loc", "msg", "type"}]}`, with status 400 for a malformed body
and 422 for a request the model cannot answer, such as a score with more than ten levels.

## Python

```python
from startlux_decision import StartLuxDecision

m = StartLuxDecision("StartLux-Decision-4B")                   # device defaults to cuda when available
answers, usage = m.decide(state, questions)
answers_list = m.decide_batch([(state1, questions1), (state2, questions2), ...])
```

`decide` is the latency path. `decide_batch` is the throughput path: every question of every request is sorted by
length and packed into padded forward passes of up to `max_batch_tokens` tokens (65,536 by default). On a random 2%
sample of the Decision Index suite (2,678 requests) StartLux-Decision-4B took 140 s with `decide_batch` against 337 s calling
`decide` once per request, and the chosen options agreed on 99.94% of the 6,897 questions. The differences are bf16
rounding between batch shapes; on 981 JevBench and Typed Decisions questions the largest probability difference was
0.008.

## Long inputs

A prompt can be as long as 262,144 tokens, the native context of all six models (`max_length`, or `--max-length` on the
server). Every question of a request carries the full state, so their prompts begin with the same tokens. When that
shared beginning is longer than 4,096 tokens it runs once: the model reads it in chunks of 32,768 tokens
(`prefill_chunk`) and keeps what each layer carries forward, the keys and values of the attention layers and the
convolution and recurrent state of the linear-attention layers. Then all questions run as one batch on top of it. A
single question over a long document runs the same way, so memory stays close to the keys and values of the prompt
instead of growing with the activations of the whole input. Attention over the stored keys runs as fused cuDNN or flash
attention calls whose outputs are merged by their log-sum-exp, without an attention mask. Prompts that long take seconds
rather than milliseconds, and at 256K tokens the 27B and the 35B-A3B need most of an 80 GB GPU.

## Images

The checkpoints include a vision tower, and the package uses it. Images are part of the evidence of a request:

```python
answers, usage = m.decide("Photo taken at delivery: <image>",
                          {"damaged": {"type": "noul", "instructions": "Is the parcel damaged?"}},
                          images=["parcel.jpg"])
```

```bash
curl -s localhost:8090/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Photo taken at delivery: <image>",
  "questions": {"damaged": {"type": "noul", "instructions": "Is the parcel damaged?"}},
  "images": ["data:image/jpeg;base64,'"$(base64 < parcel.jpg | tr -d '\n')"'"]
}'
```

- In Python an image can be a PIL image, a file path, encoded bytes, a base64 string or a data URI. Over HTTP, `images`
  is a list of base64 strings or data URIs.
- `<image>` in a string state marks where each image goes, one mark per image, in order. Without marks the images come
  first (as `Image 1:`, `Image 2:` ... when there are several), followed by the state.
- Each image is resized to at most 1,048,576 pixels, keeping its aspect ratio (`max_pixels`, `--max-pixels`), and
  becomes one token per 32 × 32 pixels: a 1024 × 1024 photo is 1,024 tokens.
- The vision tower runs once per request, and every question of the request reads the same image tokens. Image
  requests run on the eager path, without the CUDA graphs.
- The vision tower adds 0.2 to 0.9 GB of GPU memory. `StartLuxDecision(..., images=False)` or `--no-images` leaves it
  out.
- The MLX and GGUF backends read text only.

## Why it is fast

Three things, in order of how much they matter.

1. The fast kernels above. Without them everything else is moot.
2. No generation. One forward pass per request, each question one row of the batch, and only 26 rows of the output
   matrix are ever multiplied.
3. GPU graphs. At start-up the model records graphs through `torch.cuda.CUDAGraph` on CUDA or ROCm, in a single shared
   memory pool: one per padded input length
   (128, 192, 256 ... 4096 tokens) for a single question, and one per question count and length for requests with two
   to four questions of up to 1024 tokens (40 graphs in all). The experts of StartLux-Decision-35B-A3B run as grouped
   matrix multiplications, with no synchronisation with the host, so they are recorded as well. A short request is right-padded to the next length and
   all its questions are replayed as one graph, which removes the per-layer kernel launch overhead that dominates small
   inputs. Padding goes after the last prompt token, every layer is causal and the rows never mix, so it never affects
   the position that is read. Recording adds to start-up time; set `STARTLUX_GRAPHS=0` to skip it. Requests with more
   or longer questions run on the eager path, still as one batch, and bulk work belongs in `decide_batch`.
   Eager and batched inputs are padded to a fixed ladder of lengths, because the linear-attention kernels are compiled
   once per sequence length. At start-up the server runs one request through both the graph and the eager path and
   prints the largest probability difference (0.0 for all six models); above 0.02 it drops the graphs.

![Latency by model size on one H200](../media/latency.png)

Measured end to end over HTTP on one H200, bf16, one request at a time (`eval/latency.py`, 20 warm-up and 200 timed
requests). "3 fields" is one choice, one yes/no and one score question on a support ticket; the three prompts total
891 tokens because each question carries the full state, and the three run together in one forward pass, as
Intern-Decision's fields do.

<div align="center">

| Model | 3 fields: mean | P50 | P95 | one yes/no question |
|:---:|:---:|:---:|:---:|:---:|
| StartLux-Decision-0.8B | 12.2 ms | 12.2 ms | 14.2 ms | 8.3 ms |
| StartLux-Decision-2B | 15.5 ms | 15.5 ms | 17.2 ms | 9.6 ms |
| StartLux-Decision-4B | 26.0 ms | 26.0 ms | 27.3 ms | 14.7 ms |
| StartLux-Decision-9B | 35.7 ms | 36.2 ms | 37.4 ms | 17.6 ms |
| StartLux-Decision-27B | 102.3 ms | 102.5 ms | 104.3 ms | 50.7 ms |
| StartLux-Decision-35B-A3B | 52.5 ms | 52.9 ms | 54.5 ms | 36.4 ms |
| StartLux-Decision-4B, graphs off | 90.3 ms | 89.5 ms | 92.5 ms | 87.5 ms |

</div>

For reference, Intern-Decision reports 34.0 ms (0.8B), 33.3 ms (2B) and 44.2 ms (4B) on an RTX 4090 for a request of the
same shape. Different hardware, so read it as a ballpark, not a head-to-head. For Jev 1.13 we sent the same two requests
to the TypeSafe API 100 times each on 2026-09-29. Its gateway reports 64.0 ms of server time for the three fields and
63.5 ms for the single yes/no question (`x-envoy-upstream-service-time`, means; medians 57.5 and 58.0 ms); Jev answers all
fields at once, so the count barely matters. End to end from our cluster the requests took about 330 ms, of which about
260 ms is the network. Intern-Decision's 109.7 ms for the Jev API is also an end-to-end number.

## FP8

The models also run with FP8 weights and activations (torchao, `Float8DynamicActivationFloat8WeightConfig` with per-row
scales on the text backbone; the letter readout stays in bf16). On 2,541 decisions (JevBench public, Typed Decisions
test, ToolACE test) the chosen option matched bf16 in this share of cases:

<div align="center">

| Model | agreement with bf16 | accuracy bf16 -> FP8 |
|:---:|:---:|:---:|
| StartLux-Decision-0.8B | 96.3% | 78.00 -> 77.76 |
| StartLux-Decision-2B | 96.5% | 80.44 -> 80.05 |
| StartLux-Decision-4B | 98.0% | 82.37 -> 81.98 |
| StartLux-Decision-9B | 98.2% | 83.67 -> 83.23 |
| StartLux-Decision-27B | 98.6% | 82.96 -> 82.76 |

</div>

FP8 is a memory option here, not a speed option. Weight memory roughly halves (StartLux-Decision-27B takes 27.5 GiB on the GPU
after quantisation), but in this test FP8 was about 2.7 times slower per decision than bf16, both run eagerly with one
question per forward pass: for inputs this short, quantising activations on the fly costs more than the smaller
matmuls save.

One caveat: with dynamic per-row activation scales, an all-zero padding row gets a zero scale and produces NaN, which
then leaks into real rows through attention. When we let FP8 run on padded batches, agreement with bf16 fell to about
52%. Run FP8 one question per forward pass, without padding.

## GGUF and llama.cpp

GGUF files of every size are on Hugging Face, one repository per size and precision (BF16, Q8_0 and Q4_K_M), named
`startlux-models/StartLux-Decision-<size>-<precision>-GGUF` (on ModelScope: `StartLuxAI/` with the same names). They hold
the text decoder only. The prompt format, the
option-letter readout and the per-type temperatures stay in this package, and `startlux_decision.gguf_server` puts them
in front of llama-server:

```bash
hf download startlux-models/StartLux-Decision-4B-Q8_0-GGUF --local-dir StartLux-Decision-4B-Q8_0-GGUF
cd StartLux-Decision-4B-Q8_0-GGUF
pip install -r requirements.txt                     # transformers and torch; a CPU build of torch is enough
llama-server -m StartLux-Decision-4B-Q8_0.gguf -ngl 99 -c 16384 --parallel 4 --port 8081
python -m startlux_decision.gguf_server --model-dir . --llama http://127.0.0.1:8081 --port 8090
```

The server speaks the same `/v1/systemone` format as `startlux_decision.server`. It asks llama-server for the
next-token log-probabilities at the answer position and applies the same readout, temperatures and wide-choice rounds,
so a GGUF file can be compared with the original weights question by question; the GGUF table in the README does that
on the public JevBench items. Plain chat with a GGUF file does not give these decisions. llama.cpp has to be recent
enough to support this model (build b10454 or newer).

## Apple Silicon (MLX)

On a Mac, `pip install -r requirements.txt` installs [mlx-lm](https://github.com/ml-explore/mlx-lm) instead of the CUDA
kernels, and the same commands run the model with MLX (`--backend auto`, the default; `--backend torch` forces PyTorch):

```bash
hf download startlux-models/StartLux-Decision-4B --local-dir StartLux-Decision-4B
pip install -r requirements.txt
python -m startlux_decision.server --model StartLux-Decision-4B --port 8090           # bf16
python -m startlux_decision.server --model StartLux-Decision-4B --port 8090 --int8    # M5 and later
```

```python
from startlux_decision.mlx_model import MLXDecision
model = MLXDecision("StartLux-Decision-4B")                       # int8=True on M5 and later
answers, usage = model.decide(state, questions)
```

The MLX backend does three things:

- **The forward pass runs in MLX.** `MLXDecision` replaces only the forward pass, as the GGUF server does, and uses
  mlx-lm's implementation of the model with its Metal kernels for the linear-attention layers. PyTorch on MPS has no such
  kernels and runs reference code. Prompts, the option-letter readout, the temperatures and wide choices are the
  package's own code. In bf16 the public JevBench scores of 0.8B, 2B, 4B and 9B are the published ones.
- **Prefixes are reused.** Every prompt starts with the same system text. It runs once when the server starts, and each
  request continues from its cache (attention keys and values, linear-attention conv and recurrent states). The
  questions of a request are rows of one forward pass. When they share a long piece of evidence, the evidence runs once,
  and a later request about the same evidence runs only its questions.
- **`--int8` uses int8 matmuls on M5 and later.** The large projections run as int8 × int8 matmuls on the GPU's neural
  accelerators. Activations are quantized per token and weights per channel. SmoothQuant scales, computed at start-up
  from a few built-in requests, are applied first. On the public JevBench items the decisions match bf16 (the 8-bit
  model for 27B) on 97.4 to 99.6% of the items.

StartLux-Decision-27B in bf16 needs 56 GB for its weights. On a 64 GB Mac, convert it to 8-bit first. The 8-bit model
scores 208 of 231 on public JevBench; bf16 scores 209.

```bash
mlx_lm.convert --hf-path StartLux-Decision-27B --mlx-path StartLux-Decision-27B-MLX-8bit -q --q-bits 8
cp StartLux-Decision-27B/decision_config.json StartLux-Decision-27B-MLX-8bit/
python -m startlux_decision.server --model StartLux-Decision-27B-MLX-8bit --port 8090 --int8
```
