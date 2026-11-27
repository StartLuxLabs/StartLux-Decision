<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner_dark.png">
  <img src="docs/assets/banner_light.png" alt="StartLux-Decision" width="100%">
</picture>

<p align="center">
  <a href="https://huggingface.co/startlux-models">Hugging Face</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2610.17309">Technical report</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://decision.startlux.com/">Blog</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="docs/api.md">API reference</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://decision.startlux.com/portal/login">Free unlimited use</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.17309"><img src="https://img.shields.io/badge/arXiv-2610.17309-8A5CD6" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/code-Apache--2.0-8A5CD6" alt="License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.9%2B-8A5CD6" alt="Python"></a>
  <a href="#engines"><img src="https://img.shields.io/badge/backends-vLLM%20%7C%20llama.cpp%20%7C%20MLX-8A5CD6" alt="Backends"></a>
</p>

> **Service update:** The previous access page and API endpoint have been retired. Please use the new [StartLux API portal](https://decision.startlux.com/portal/login).

StartLux-Decision is a family of open decision models. Given a state (text, JSON or images) and a set of typed
questions, a model returns a probability for every option of every question: a choice among listed options
(`choice`), a yes or no judgement (`noul`), or a level on a scale (`score`). Instant models answer in a single forward
pass with bidirectional attention over the prompt. AutoThink models answer in one pass when the top probability
reaches a gate, 0.6 by default, and otherwise generate a short reasoning trace and read the answer again.

This repository contains the inference engine and HTTP server for both variants. The API is compatible with System
One `/v1/systemone`.

<p align="center"><img src="docs/assets/score_vs_size.png" width="860" alt="Decision Index 0.3 public score against model size"></p>

## Demos

<table align="center">
  <tr>
    <td width="50%" align="center"><a href="v1/media/rescue_harvey_27b.mp4"><img src="v1/media/rescue_harvey_27b.gif" alt="Coast Guard helicopter rescue"></a><br><sub>Helicopter rescue: calls each stage of a Coast Guard hoist rescue in Hurricane Harvey, one frame a second</sub></td>
    <td width="50%" align="center"><a href="v1/media/chess_27b_scotch_vs_jev.mp4"><img src="v1/media/chess_27b_scotch_vs_jev.gif" alt="Chess against Jev 1.13"></a><br><sub>Chess: with White, checkmates Jev 1.13 on move 19 of a Scotch Game, and neither side searches</sub></td>
  </tr>
  <tr>
    <td width="50%" align="center"><a href="v1/media/mario_1-1_27b.mp4"><img src="v1/media/mario_1-1_27b.gif" alt="Super Mario Bros."></a><br><sub>Super Mario Bros.: clears World 1-1</sub></td>
    <td width="50%" align="center"><a href="v1/media/sc2_hard_27b.mp4"><img src="v1/media/sc2_hard_27b.gif" alt="StarCraft II"></a><br><sub>StarCraft II: makes the strategic calls for a Terran bot and beats the built-in Hard AI</sub></td>
  </tr>
  <tr>
    <td width="50%" align="center"><a href="v1/media/launch_artemis_27b.mp4"><img src="v1/media/launch_artemis_27b.gif" alt="Artemis I launch"></a><br><sub>Rocket launch: follows the Artemis I night launch one frame a second, calls liftoff and tells the camera when to tilt up</sub></td>
    <td width="50%" align="center"><a href="v1/media/jevball_27b.mp4"><img src="v1/media/jevball_27b.gif" alt="JevBall"></a><br><sub>JevBall: makes the decisions for a football team in real time</sub></td>
  </tr>
</table>

In the chess and game runs, every step is one request with typed questions, answered by StartLux-Decision-v1-27B, the
first release, in one pass. The rescue and launch videos send one frame a second with nine questions to the 27B
AutoThink model on [decision.startlux.com](https://decision.startlux.com/portal/login), which thinks only when its top probability is below the
0.6 gate. Their footage is from the U.S. Coast Guard and NASA (public domain, no endorsement implied), and the faces of
the people being rescued are blurred. Click a preview for the video. [v1/README.md](v1/README.md#demos) describes each
harness, and the harnesses for computer use and chess are in [v1/demos](v1/demos).

## Release notes

- 2026-10-16: StartLux-Decision v1.1 27B Instant ranks first of 174 entries on the
  [Decision Index 0.3.2](https://huggingface.co/spaces/multimodalart/jev-decision-index) leaderboard on Hugging Face,
  with a Full score of 69.12. It is also first on each of the three parts: the public benchmarks (72.88), the private
  tests of the same skills (64.82) and the private tasks from new domains (61.41).
- 2026-10: the technical report is on arXiv as [2610.17309](https://arxiv.org/abs/2610.17309).

## Models

| StartLux-Decision v1.1 | Decision Index 0.3 | DecisionPivot | Other formats |
|---|---|---|---|
| [27B AutoThink](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-AutoThink) | **73.37** | **86.9** | [GGUF](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-AutoThink-GGUF) · [MLX 8-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-AutoThink-MLX-8bit) · [MLX 4-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-AutoThink-MLX-4bit) |
| [27B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant) | 72.83 | 75.4 | [GGUF](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant-GGUF) · [MLX 8-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant-MLX-8bit) · [MLX 4-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant-MLX-4bit) |
| [9B AutoThink](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-AutoThink) | **70.45** | **59.9** | [GGUF](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-AutoThink-GGUF) · [MLX 8-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-AutoThink-MLX-8bit) · [MLX 4-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-AutoThink-MLX-4bit) |
| [9B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-Instant) | 69.96 | 47.5 | [GGUF](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-Instant-GGUF) · [MLX 8-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-Instant-MLX-8bit) · [MLX 4-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-Instant-MLX-4bit) |
| [4B AutoThink](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-AutoThink) | **68.21** | **59.9** | [GGUF](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-AutoThink-GGUF) · [MLX 8-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-AutoThink-MLX-8bit) · [MLX 4-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-AutoThink-MLX-4bit) |
| [4B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-Instant) | 67.88 | 41.5 | [GGUF](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-Instant-GGUF) · [MLX 8-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-Instant-MLX-8bit) · [MLX 4-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-Instant-MLX-4bit) |
| [2B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-2B-Instant) | 59.14 | 21.1 | [GGUF](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-2B-Instant-GGUF) · [MLX 8-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-2B-Instant-MLX-8bit) · [MLX 4-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-2B-Instant-MLX-4bit) |
| [0.8B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant) | 53.75 | 8.1 | [GGUF](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant-GGUF) · [MLX 8-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant-MLX-8bit) · [MLX 4-bit](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant-MLX-4bit) |

Decision Index 0.3 refers to the public subset, evaluated with the public kit. DecisionPivot reports pair accuracy on
7,800 programmatically generated examples arranged in contrast pairs that differ in one decisive detail. A pair counts
only when both answers are correct. The AutoThink results use a gate of 0.9.

Across sizes, the Instant models exceed the strongest evaluated decision model of comparable size by 7.4 to 16.9 points
on Decision Index 0.3, and AutoThink adds 11.5 to 18.4 points of DecisionPivot pair accuracy over Instant. AutoThink
reasoning is trained with RLCD², reinforcement learning for calibrated deliberative decisions. The technical report
contains the full evaluation. The engine also serves the
[v1 models](https://huggingface.co/collections/startlux-models/startlux-decision-v1-6abba92b301b573fa154d493)
(0.8B to 27B and 35B-A3B), which answer in one pass.

On one H200 with vLLM and one request in flight, Instant models answer a request in 25 to 52 ms at the median, from
0.8B to 27B. Through this server, an AutoThink model needs one read-out of 38 to 59 ms when it is sure. When it is
unsure it writes a thought and reads the answer a second time, and the thought takes longer the more it writes.

## Quick start

To try StartLux-Decision without a GPU, sign in to the [StartLux API portal](https://decision.startlux.com/portal/login) for free, unlimited use.

On NVIDIA GPUs, [StartLux-Prism](https://github.com/StartLuxLabs/StartLux-Prism) is our inference engine written for
these models. It serves v1 and v1.1 AutoThink (4B, 9B and 27B) straight from the released GGUF files, behind the same
`/v1/systemone` API; v1.1 Instant models are not supported yet. Install it on Linux with Python 3.12 and a CUDA 13
driver:

```bash
git clone https://github.com/StartLuxLabs/StartLux-Prism.git
cd StartLux-Prism
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # PyTorch 2.14.1 for CUDA 13, Triton, nvcc 13.0
source scripts/cuda_env.sh          # nvcc from pip as CUDA_HOME (in every new shell)
```

Download a GGUF file with its release's tokenizer and config, and start the server:

```bash
M=~/models/StartLux-Decision-v1.1-27B-AutoThink-GGUF
hf download startlux-models/StartLux-Decision-v1.1-27B-AutoThink-GGUF \
  StartLux-Decision-v1.1-27B-AutoThink-Q8_0.gguf --local-dir $M
hf download startlux-models/StartLux-Decision-v1.1-27B-AutoThink --exclude "*.safetensors" --local-dir $M
python -m startlux_prism.server --gguf $M/StartLux-Decision-v1.1-27B-AutoThink-Q8_0.gguf --model-dir $M --port 8090
```

The server is ready when it prints `Ready`; the first start compiles its CUDA kernels and measures the int8 activation
ranges once. It answers the request below as it is. `allow_none` is on by default for v1.1 models (`--allow-none off`
matches this package), and the server has no API keys or TLS, so keep it behind your own access control.
[StartLux-Prism's README](https://github.com/StartLuxLabs/StartLux-Prism#readme) covers CUDA 12 drivers, several GPUs
and every flag.

To run the models yourself, install the package into the same Python environment as vLLM. It carries the vLLM plugin
that Instant models need.

```bash
git clone https://github.com/StartLuxLabs/StartLux-Decision.git
cd StartLux-Decision
pip install -e '.[vllm]'          # vLLM 0.31.0; the published numbers were measured on 0.19.1, which is also supported
```

Download a model and start one vLLM server per GPU with the decision server in front of them:

```bash
hf download startlux-models/StartLux-Decision-v1.1-4B-AutoThink --local-dir models/StartLux-Decision-v1.1-4B-AutoThink
startlux-decide serve --model models/StartLux-Decision-v1.1-4B-AutoThink --launch-vllm --gpus 0,1,2,3 --port 8090
```

Ask a question:

```bash
curl -s localhost:8090/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "I was charged twice for order #4411 and the app still shows it as unpaid."},
  "questions": {
    "team":     {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Payments and refunds", "shipping": "Delivery", "technical": "App and login"}},
    "urgent":   {"type": "noul", "instructions": "Should this ticket be answered today?"},
    "severity": {"type": "score", "instructions": "How severe is the impact?",
                 "criteria": ["cosmetic", "annoying", "blocks the customer"]}
  }
}'
```

The 27B AutoThink model answers (numbers rounded):

```json
{"answers": {
  "team":     {"type": "choice", "choice": "billing", "confidence": 0.992,
               "probabilities": {"billing": 0.995, "shipping": 0.000, "technical": 0.005},
               "none_probability": 0.002},
  "urgent":   {"type": "noul", "noul": 0.870},
  "severity": {"type": "score", "score": 1.858, "confidence": 0.787,
               "legend": {"0": "cosmetic", "1": "annoying", "2": "blocks the customer"},
               "probabilities": {"0": 0.005, "1": 0.133, "2": 0.863}}},
 "usage": {"input_tokens": 290, "output_tokens": 144}, "model": "startlux-decision-27b-autothink"}
```

`noul` is the probability of yes. `score` is the expected level, with a probability for every level.
`none_probability` is the chance that none of the listed options fits (see Controls).

[examples/](examples) has the same request in Python, a request with an image and a small batch client.

## Controls

A request can steer the reasoning of AutoThink models and ask for more detail. All fields are optional.

| Field | Default | Effect |
|---|---|---|
| `think_mode` | `auto` | `auto` reasons when the first read-out's top probability is below `gate`; `never` answers in one pass; `always` reasons first |
| `gate` | 0.6 | the confidence below which `auto` reasons (the published results use 0.9) |
| `think_budget` | 4096 | the longest thought, in tokens |
| `none_check` | true | every choice answer reports `none_probability`, the chance that no listed option fits; `choice` stays a listed option |
| `allow_none` | false (`serve --allow-none`: true) | lets `choice` be null when "None of the above" is the most likely answer |
| `return_details`, `return_thought` | false | per-answer read-outs and timings, and the thought text |
| `images` | none | base64 strings or data URIs used as evidence |

The default gate of 0.6 answers more requests in one pass. A gate of 0.9 thinks more often and is more accurate on
decisions that need rules or arithmetic. For the 27B AutoThink model on this engine, moving the gate from 0.9 to 0.6
lowers Decision Index 0.3 from 73.22 to 72.85 and DecisionPivot pair accuracy from 88.1% to 68.4%. Instant models
always answer in one pass.

We suggest treating a `none_probability` of 0.2 or more as "no listed option fits". At 0.2 it catches two thirds
(66.3%) of the questions whose listed options are all wrong, and flags 2.2% of the questions that do have a right
answer. Lower thresholds flag many more good questions, higher ones miss more of the bad ones
([why 0.2](docs/none_probability.md#why-02)). [docs/api.md](docs/api.md) describes the full request and response
format, the confidence definitions, the details fields and the errors.

## Engines

| Engine | Instant | AutoThink | Images | Typical use |
|---|---|---|---|---|
| vLLM (default) | yes, with the plugin in this package | yes | vLLM 0.31 | serving, and the published numbers |
| [StartLux-Prism](https://github.com/StartLuxLabs/StartLux-Prism) | not yet | yes, 4B, 9B and 27B | with the `mmproj` file | the GGUF files on NVIDIA GPUs, lowest latency |
| llama.cpp | yes, GGUF builds marked non-causal | yes | with the `mmproj` file | machines without vLLM |
| MLX | yes, mlx-vlm builds | yes | yes | Apple Silicon |
| transformers | yes | yes | yes | reference checks |

FP8 and GGUF Q8_0 largely preserve BF16 decision performance, while Q4_K_M can lose up to 4.3 points (27B AutoThink
on DecisionPivot). [docs/deployment.md](docs/deployment.md)
lists the engine settings to switch on and off and why, and [docs/quantization.md](docs/quantization.md) covers the
quantized builds.

## Serving in production

`startlux-decide serve` can face the internet directly, with API keys, HTTPS, early 429 responses under load, an
access log and an optional request log. [docs/serving_publicly.md](docs/serving_publicly.md) has a complete command
and a checklist. The other commands measure a deployment: `bench` for latency, `eval` for accuracy and calibration on
labelled requests, and `quantize` for FP8, GGUF and MLX builds.

## Documentation

| Document | Contents |
|---|---|
| [docs/api.md](docs/api.md) | request and response format, controls, confidence, `none_probability`, errors |
| [docs/none_probability.md](docs/none_probability.md) | what `none_probability` means, how it was measured, why 0.2 |
| [docs/deployment.md](docs/deployment.md) | engine settings, images and long inputs, compatibility with v1 models |
| [docs/engines.md](docs/engines.md) | how each engine is driven and what was measured |
| [docs/vllm_bidirectional.md](docs/vllm_bidirectional.md) | the vLLM plugin for prompt-bidirectional attention |
| [docs/quantization.md](docs/quantization.md) | FP8, GGUF and MLX builds |
| [docs/serving_publicly.md](docs/serving_publicly.md) | keys, TLS, rate limits, logs |
| [docs/benchmarks.md](docs/benchmarks.md) | how accuracy and latency are measured |

## Tests

```bash
python -m unittest -v
```

The tests need only the standard library and skip the parts that need aiohttp, torch or transformers when those are
missing. Set `STARTLUX_TEST_MODEL_DIR` to a model folder to add checks against its real tokenizer.

## The first release

StartLux-Decision-v1 (0.8B to 27B and a 35B-A3B mixture of experts) answers every question in one pass. Its inference
code, evaluation scripts, results, game logs and demo harnesses are in [v1/](v1), and the repository as it was at that
release is tagged `v1`. The engine in this repository serves the v1 models as well.

## Citation

Xialie Zhuang<sup>1,2</sup>, Zunhai Su<sup>1,2</sup>, Jia Pan<sup>2</sup>, Ngai Wong<sup>2</sup>, Guanhua Chen<sup>3,\*</sup>, Chuan-Wei Kuo<sup>1,\*</sup>

<sup>1</sup>StartLux&nbsp;&nbsp;<sup>2</sup>The University of Hong Kong&nbsp;&nbsp;<sup>3</sup>Southern University of Science and Technology&nbsp;&nbsp;<sup>\*</sup>Corresponding authors

```bibtex
@misc{startlux2026decision,
  title         = {StartLux-Decision: AutoThink Decision Models with Reinforcement Learning for Calibrated Deliberative Decisions},
  author        = {Zhuang, Xialie and Su, Zunhai and Pan, Jia and Wong, Ngai and Chen, Guanhua and Kuo, Chuan-Wei},
  year          = {2026},
  eprint        = {2610.17309},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.17309}
}
```

## License

The code in this repository is released under the [Apache License 2.0](LICENSE). The model weights are released
under CC BY-NC 4.0 for non-commercial use. For commercial licensing, contact StartLux.
