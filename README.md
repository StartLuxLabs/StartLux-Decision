<p align="center"><img src="media/hero.png" alt="StartLux-Decision: a probability for every option" width="100%"></p>

<p align="center">
  <a href="https://huggingface.co/collections/startlux-models/startlux-decision-6abba92b301b573fa154d493">Models on Hugging Face</a> ·
  <a href="https://modelscope.cn/organization/StartLuxAI">Models on ModelScope</a> ·
  <a href="docs/results.md">Results</a> ·
  <a href="docs/inference.md">Inference</a> ·
  <a href="docs/evaluation.md">Evaluation</a> ·
  <a href="docs/finetuning.md">Fine-tuning</a> ·
  <a href="results/">Raw results</a>
</p>

> **New, 2026-10-03: images and 256K-token context.** Every size now reads images as part of a request's evidence and
> prompts up to 262,144 tokens. See [Images and long inputs](#images-and-long-inputs).

StartLux-Decision is a family of typed decision models: five dense sizes from 0.8B to 27B, and a 35B-A3B mixture of
experts. You send a state and a set of
questions: pick one of several options, yes or no, or a rating on a scale. The state can be text, JSON or images, and as
long as 262,144 tokens (256K) at every size. Every question comes back with a probability
for each option. Nothing is generated; the answer is read from the option letters after one forward pass, so a short
question takes a few milliseconds and the probabilities can be used as confidence. Requests and responses use the
TypeSafe `/v1/systemone` format, so clients written for Jev work unchanged. This repository has the inference code,
the evaluation scripts, the results and the raw game logs. The weights are on Hugging Face, in the
[StartLux-Decision collection](https://huggingface.co/collections/startlux-models/startlux-decision-6abba92b301b573fa154d493),
and on ModelScope, under [StartLuxAI](https://modelscope.cn/organization/StartLuxAI).

## Demos

<table align="center">
  <tr>
    <td width="50%" align="center"><a href="media/computer_use_store_27b.mp4"><img src="media/computer_use_store_27b.gif" alt="StartLux-Decision-27B orders batteries in a web store"></a><br><sub>StartLux-Decision-27B finds the cheapest AA 8-pack with free delivery and orders it to the right address</sub></td>
    <td width="50%" align="center"><a href="media/chess_27b_scotch_vs_jev.mp4"><img src="media/chess_27b_scotch_vs_jev.gif" alt="StartLux-Decision-27B plays chess against Jev 1.13"></a><br><sub>StartLux-Decision-27B, with White, checkmates Jev 1.13 on move 19 of a Scotch Game; neither side searches</sub></td>
  </tr>
  <tr>
    <td width="50%" align="center"><a href="media/mario_1-1_27b.mp4"><img src="media/mario_1-1_27b.gif" alt="StartLux-Decision-27B plays Super Mario Bros."></a><br><sub>StartLux-Decision-27B clears World 1-1 of Super Mario Bros.</sub></td>
    <td width="50%" align="center"><a href="media/sc2_hard_27b.mp4"><img src="media/sc2_hard_27b.gif" alt="StartLux-Decision-27B plays StarCraft II"></a><br><sub>StartLux-Decision-27B makes the strategic calls for a Terran bot and beats the built-in Hard AI</sub></td>
  </tr>
  <tr>
    <td width="50%" align="center"><a href="media/doom_deathmatch_27b.mp4"><img src="media/doom_deathmatch_27b.gif" alt="StartLux-Decision-27B plays Doom"></a><br><sub>StartLux-Decision-27B plays a Doom deathmatch against four built-in bots</sub></td>
    <td width="50%" align="center"><a href="media/jevball_27b.mp4"><img src="media/jevball_27b.gif" alt="StartLux-Decision-27B plays JevBall"></a><br><sub>StartLux-Decision-27B makes the decisions for a football team in real time</sub></td>
  </tr>
</table>

All six are real runs of StartLux-Decision-27B; click a preview for the video.

**Computer use** (our own harness). A real Chrome window opens a small mock site, and every step is one request with two
typed questions: which of the controls visible on the page to use next, and whether the task is done. The harness
carries out the chosen action and nothing else. The model never types text; a text box it clicks only opens its
suggestion list. The task was completed, and the harness checked the result against it. The side panel lists
the four likeliest controls and the rest in one row, so that block adds up to 100%; below it, set apart, is the
probability that the task is done. The harness and the two sites are in [demos/computer_use](demos/computer_use).

**Chess** (wondertwins/jev-benchmark). StartLux-Decision-27B plays Jev 1.13 through the harness at its default rich
level: every turn is one request with one choice question over all legal moves, given the board, the move history and
the pieces in text. Both sides get the same request, neither searches, and code never overrides a move; Stockfish only
draws the evaluation bar. This is one of the 256 games of the match under [Games](#games), from the position after the
first four moves of a Scotch Game; over the 256 games StartLux-Decision-27B scores 58.4%. The match tools are in
[demos/chess](demos/chess).

**Super Mario Bros.** (4esv/jev-mario). Every step is one request with one choice question over eleven moves. The
harness plays each move ahead in the emulator and describes the outcome in text, such as progress and whether Mario
survives; the model reads only these descriptions and picks a move. The panel shows the probability it gave each move.

**StarCraft II** (our own bot on burnysc2). Every 12 seconds of game time, one request asks three questions: what to
build next, which units to prioritise and whether to attack, defend or gather. A scripted bot carries out the choices:
workers, placement, production and unit control. The game is sped up; the panel shows each decision with its
probabilities.

**Doom** (our own harness on ViZDoom). Every four tics, one request asks which enemy to target, whether to fire and how
to move. The state is a text description from the game engine. Aiming, pathing and getting unstuck are done in code. The
game waits for each decision.

**JevBall** (atarikcaliskan/jevball). One request covers the players near the play, one choice each over as many as
fourteen actions: pass, shot, dribble, press, run and more. The match never waits for the model: the game handles
movement and physics, and its own policy covers players far from the ball and any late answer. The other team is the
game's built-in policy.

## Decision Index

<p align="center"><img src="media/di_chart.png" alt="Decision Index 0.2.1: StartLux-Decision, Jev and other systems" width="100%"></p>

StartLux-Decision-27B reaches 63.88 on Decision Index 0.2.1, StartLux-Decision-35B-A3B 61.55 and StartLux-Decision-9B 58.63,
scored with the board's own kit on the full suite. The highest entry on the public board (2026-09-28) is Jev 1.13 at 57.91; our runs are not on the board.
StartLux-Decision-27B scores higher than Jev on 31 of the 38 benchmarks in the index. Our training data includes the public
train splits of 14 of them, marked † below; their test items were filtered out of it.

Every cell below is on the index's own scale: the benchmark's metric, corrected for chance, so 0% is random guessing
and 100% is perfect. The index is a weighted mean of these cells, which is why the first row matches the chart.

<div align="center">

| | StartLux-Decision-27B | StartLux-Decision-35B-A3B | StartLux-Decision-9B | StartLux-Decision-4B | Jev 1.13 | Rune 26B-A4B | Decider chat 31B | AutoJev-27B |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Decision Index 0.2.1** | **63.88** | 61.55 | 58.63 | 52.75 | 57.91 | 57.44 | 57.33 | 56.40 |
| **Knowledge & Reasoning** | 44.3 | 42.4 | 38.0 | 32.1 | **51.4** | 43.4 | 44.3 | 40.9 |
| GSM8K † | 95.3 | **95.6** | 94.3 | 87.8 | 75.6 | 75.7 | 78.8 | 61.1 |
| ChessBench | 10.7 | 8.8 | 5.3 | 4.0 | 9.8 | 11.5 | **15.4** | 9.8 |
| MuSR | **48.8** | 36.8 | 34.9 | 30.0 | 46.1 | 44.2 | 43.1 | 39.3 |
| SATA-Bench | 9.3 | 27.5 | 25.3 | 14.0 | 25.4 | **34.0** | 28.4 | 28.9 |
| GPQA Diamond ★ | 34.0 | 34.7 | 33.3 | 22.4 | **71.4** | 29.2 | 32.0 | 32.6 |
| CRUXEval | 61.6 | 54.4 | 36.5 | 27.6 | 57.1 | 59.4 | **67.2** | 60.2 |
| CLadder | 47.3 | 39.1 | 36.1 | 31.8 | 45.3 | 45.5 | **49.2** | 49.0 |
| HLE ★ | 0.0 | 0.0 | 0.0 | 0.0 | **4.7** | 0.0 | 0.0 | 0.0 |
| MMLU-Pro ★ | 66.0 | 63.0 | 56.5 | 51.6 | **80.5** | 63.1 | 65.8 | 60.1 |
| BBH ★ | 71.3 | 65.4 | 58.2 | 52.0 | **89.7** | 72.7 | 65.5 | 68.3 |
| **Language Understanding** | **74.5** | 73.6 | 71.3 | 63.9 | 62.0 | 63.1 | 60.4 | 63.5 |
| ContractNLI † | 80.3 | 78.9 | **80.6** | 76.8 | 59.1 | 66.7 | 61.8 | 68.4 |
| ANLI ★ † | **67.0** | 66.8 | 64.6 | 54.9 | 62.2 | 60.3 | 59.8 | 56.0 |
| WinoGrande ★ | **87.1** | 86.0 | 83.0 | 70.3 | 83.9 | 71.9 | 67.3 | 70.3 |
| HellaSwag ★ | 96.8 | **97.0** | 95.7 | 92.4 | 92.7 | 89.9 | 89.4 | 92.0 |
| ACOS † | 50.4 | **50.5** | 46.2 | 32.2 | 27.3 | 24.4 | 16.2 | 17.8 |
| FinEntity | 89.3 | 83.3 | 82.6 | 86.6 | 80.8 | 83.0 | 86.6 | **89.6** |
| iSarcasmEval † | 52.4 | **58.7** | 51.9 | 37.3 | 36.3 | 49.0 | 37.0 | 49.0 |
| VAST † | **73.4** | 70.3 | 68.7 | 65.0 | 46.9 | 64.9 | 59.8 | 56.2 |
| NLI4CT † | **72.3** | 70.0 | 66.1 | 60.0 | 69.0 | 62.0 | 66.6 | 70.5 |
| RAGTruth † | **70.1** | 68.7 | 67.5 | 58.5 | 51.3 | 51.9 | 52.2 | 59.2 |
| **Retrieval & Classification** | **66.8** | 65.8 | 64.1 | 56.7 | 55.4 | 63.5 | 63.1 | 54.9 |
| BANKING77 ★ † | 90.9 | 91.0 | **91.3** | 85.7 | 79.5 | 83.5 | 78.8 | 78.8 |
| CLINC150 ★ † | 92.9 | **93.7** | 93.1 | 91.5 | 89.2 | 87.3 | 91.1 | 87.7 |
| BRIGHT ★ | **43.5** | 41.7 | 41.3 | 38.2 | 40.6 | 39.3 | 36.2 | 41.9 |
| Amazon ESCI † | **47.9** | 46.3 | 47.4 | 43.2 | 43.8 | 43.9 | 41.2 | 43.7 |
| PhishNChips | 40.9 | 38.2 | 28.5 | 19.8 | 25.1 | 61.8 | **75.0** | 19.9 |
| HoVer † | **79.2** | 78.1 | 76.5 | 52.8 | 45.7 | 61.3 | 52.8 | 48.4 |
| **Tools & Automation** | **82.2** | 76.2 | 73.2 | 72.3 | 75.1 | 71.2 | 75.6 | 79.3 |
| BFCL ★ | 96.7 | 96.7 | 97.0 | 94.7 | 94.3 | 93.0 | **97.3** | 96.8 |
| ToolRet | **64.3** | 64.0 | 61.5 | 62.5 | 59.9 | 58.9 | 58.0 | 62.4 |
| API-Bank ★ | 83.8 | 79.5 | 77.9 | 86.0 | **88.0** | 83.0 | 84.8 | 83.8 |
| Home appliances | **77.3** | 51.1 | 38.6 | 30.7 | 52.3 | 46.6 | 62.5 | 73.9 |
| When2Call | **85.7** | 84.7 | 85.2 | 80.3 | 74.6 | 68.0 | 69.2 | 75.6 |
| **Arts & Human Taste** | **47.9** | 44.7 | 41.8 | 33.7 | 37.7 | 41.9 | 38.3 | 39.4 |
| BPoMP | **90.6** | 85.7 | 82.1 | 66.3 | 81.8 | 79.9 | 81.2 | 87.8 |
| Humicroedit † | 25.8 | 24.7 | 24.3 | 19.6 | 23.7 | 24.4 | **27.6** | 24.8 |
| POP909 | 50.6 | 32.0 | 25.5 | 16.0 | 15.9 | **65.8** | 27.3 | 37.3 |
| cfcolor | **43.6** | 42.5 | 42.4 | 30.5 | 28.8 | 25.1 | 25.2 | 28.8 |
| ForecastBench ★ | **34.3** | 30.6 | 26.6 | 24.6 | 30.6 | 18.9 | 19.2 | 22.1 |
| Habermas | 18.4 | **26.2** | 22.5 | 17.7 | 21.5 | 16.2 | 24.4 | 15.5 |
| New Yorker † | **74.2** | **74.2** | 72.1 | 63.3 | 62.6 | 67.6 | 67.3 | 62.8 |

</div>

★ benchmarks weigh 1.2 in the index. The rows with bold names are the index's own area scores; a bold value marks the
best score in its row. The other systems' values come from the public board. The metric of each benchmark and the raw scores of all six models are in [docs/results.md](docs/results.md)
and [results/decision_index_benchmarks.csv](results/decision_index_benchmarks.csv).

## At every size

<p align="center"><img src="media/by_size.png" alt="Ahead at every size" width="100%"></p>

The public JevBench items are the comparison most decision models of this kind report. The table lists the systems
in the Intern-Decision bundle next to ours; a blank cell means the number is not published.

<div align="center">

| Model | JevBench public, of 231 | Intern avg | DI 0.2 / 0.2.1 | Latency, 3 questions |
|:---:|:---:|:---:|:---:|:---:|
| StartLux-Decision-35B-A3B | **210** | **92.29** | 57.24 / 61.55 | 52.5 ms |
| StartLux-Decision-27B | 208 | 91.82 | **59.54 / 63.88** | 102.3 ms |
| StartLux-Decision-9B | 201 | 91.08 | 54.37 / 58.63 | 35.7 ms |
| StartLux-Decision-4B | 204 | 91.17 | 48.38 / 52.75 | 26.0 ms |
| Intern-Decision-4B | 201 | 90.02 | 35.90 / 37.81 | 44.2 ms ¹ |
| JevK5 | 200 | 85.16 | 36.44 / 38.81 | |
| Jev 1.13 | 199 | 88.74 | 51.67 / 57.91 | 64.0 ms ² |
| StartLux-Decision-2B | 196 | 88.46 | 40.72 / 44.19 | 15.5 ms |
| SemIf | 187 | 84.23 | 25.70 / 25.94 | |
| Intern-Decision-2B | 180 | 84.68 | 19.49 / 19.38 | 33.3 ms ¹ |
| StartLux-Decision-0.8B | 179 | 85.03 | 35.57 / 38.86 | **12.2 ms** |
| Intern-Decision-0.8B | 163 | 79.38 | 11.32 / 11.94 | 34.0 ms ¹ |
| Laya | 130 | 57.77 | 5.51 / 6.04 | |

</div>

JevBench public counts the correct answers on the 231 public items in the Intern-Decision bundle, Intern avg is the
average accuracy over its seven suites, and DI is the Decision Index under both editions (the public board's values for
the other systems). Our latencies are for one H200, with the three questions answered in one forward pass, as
Intern-Decision does. ¹ Intern-Decision's own measurement on an RTX 4090. ² The server time the TypeSafe API gateway
reports for the same request, mean of 100, so the network is left out as in ours; Intern-Decision reports 109.7 ms end
to end.

## Compared with Jev 1.13

<p align="center"><img src="media/vs_jev.png" alt="StartLux-Decision-27B compared with Jev 1.13" width="100%"></p>

## Accuracy suites

<p align="center"><img src="media/mistakes.png" alt="Fewer mistakes" width="100%"></p>

The Intern-Decision bundle has seven suites: the three public JevBench tiers, Typed Decisions, ToolACE, AG News and
WildJailBreak. Its ToolACE items are drawn from the public ToolACE training set, which has no held-out split, so
[docs/results.md](docs/results.md) also gives the average without it (StartLux-Decision-4B: 90.67, Intern-Decision-4B: 88.95).
JevBench public counts the correct answers on the 231 public items in the bundle. It is not the official JevBench
score, which adds a sealed tier, speed and cost and is measured only by the maintainers.

## Speed

<p align="center"><img src="media/latency.png" alt="Latency on one H200" width="100%"></p>

StartLux-Decision-35B-A3B, a mixture of experts with about 3B of its 35B parameters active for each token, answers the
same request in 52.5 ms, half the time of the 27B; its experts run as grouped matrix multiplications, so the CUDA graphs
cover them too. It fits one 80 GB GPU (about 71 GiB at its peak).

The fast linear-attention kernels (`flash-linear-attention`, `causal-conv1d`) are required; the server refuses to
start on a GPU without them. All questions of a request run in one forward pass, and CUDA graphs remove most of the
launch overhead for short requests: StartLux-Decision-4B takes 90.3 ms for the same request without them. Bulk evaluation goes through a batched path instead. Details in
[docs/inference.md](docs/inference.md).

## Images and long inputs

Every size reads images and long documents. A request can carry images as part of its evidence, and a prompt can run
to 262,144 tokens (256K), the models' native context: a long state is read once, in chunks, and every question of the
request branches off it.

```python
answers, usage = m.decide("Photo taken at delivery: <image>",
                          {"damaged": {"type": "noul", "instructions": "Is the parcel damaged?"}},
                          images=["parcel.jpg"])
```

Over HTTP, `"images"` is a list of base64 strings or data URIs. Details are in
[docs/inference.md](docs/inference.md#images); the MLX and GGUF backends read text only.

## GGUF

Every dense size also comes as GGUF files for llama.cpp: BF16, which keeps the weights unchanged, and llama.cpp's standard
Q8_0 and Q4_K_M quantizations of it. Each file has its own Hugging Face repository,
`startlux-models/StartLux-Decision-<size>-<precision>-GGUF`, linked from the table below, and the same repository on
ModelScope under `StartLuxAI/`. The decision procedure is not
in the weights; `python -m startlux_decision.gguf_server` runs it in front of llama-server (see
[docs/inference.md](docs/inference.md#gguf-and-llamacpp)). Against the original weights on the 231 public JevBench items:

<div align="center">

| File | Size | Same decision as the original | JevBench public, of 231 |
|:---:|:---:|:---:|:---:|
| StartLux-Decision-0.8B, original weights | | | 177 (76.6%) |
| [`StartLux-Decision-0.8B-BF16.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-0.8B-BF16-GGUF) | 1.52 GB | 99.6% | 178 (77.1%) |
| [`StartLux-Decision-0.8B-Q8_0.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-0.8B-Q8_0-GGUF) | 0.81 GB | 100.0% | 177 (76.6%) |
| [`StartLux-Decision-0.8B-Q4_K_M.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-0.8B-Q4_K_M-GGUF) | 0.53 GB | 93.1% | 172 (74.5%) |
| StartLux-Decision-2B, original weights | | | 195 (84.4%) |
| [`StartLux-Decision-2B-BF16.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-2B-BF16-GGUF) | 3.78 GB | 100.0% | 195 (84.4%) |
| [`StartLux-Decision-2B-Q8_0.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-2B-Q8_0-GGUF) | 2.01 GB | 99.1% | 193 (83.5%) |
| [`StartLux-Decision-2B-Q4_K_M.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-2B-Q4_K_M-GGUF) | 1.27 GB | 94.4% | 199 (86.1%) |
| StartLux-Decision-4B, original weights | | | 204 (88.3%) |
| [`StartLux-Decision-4B-BF16.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-4B-BF16-GGUF) | 8.42 GB | 100.0% | 204 (88.3%) |
| [`StartLux-Decision-4B-Q8_0.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-4B-Q8_0-GGUF) | 4.48 GB | 100.0% | 204 (88.3%) |
| [`StartLux-Decision-4B-Q4_K_M.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-4B-Q4_K_M-GGUF) | 2.71 GB | 98.3% | 201 (87.0%) |
| StartLux-Decision-9B, original weights | | | 200 (86.6%) |
| [`StartLux-Decision-9B-BF16.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-9B-BF16-GGUF) | 17.92 GB | 99.6% | 201 (87.0%) |
| [`StartLux-Decision-9B-Q8_0.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-9B-Q8_0-GGUF) | 9.53 GB | 100.0% | 200 (86.6%) |
| [`StartLux-Decision-9B-Q4_K_M.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-9B-Q4_K_M-GGUF) | 5.63 GB | 98.3% | 201 (87.0%) |
| StartLux-Decision-27B, original weights | | | 209 (90.5%) |
| [`StartLux-Decision-27B-BF16.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-27B-BF16-GGUF) | 53.81 GB | 100.0% | 209 (90.5%) |
| [`StartLux-Decision-27B-Q8_0.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-27B-Q8_0-GGUF) | 28.60 GB | 100.0% | 209 (90.5%) |
| [`StartLux-Decision-27B-Q4_K_M.gguf`](https://huggingface.co/startlux-models/StartLux-Decision-27B-Q4_K_M-GGUF) | 16.55 GB | 96.5% | 208 (90.0%) |

</div>

BF16 and Q8_0 give the original answer on 99 to 100% of the items. Q4_K_M keeps 96.5 to 98.3% from 4B up; at 0.8B and
2B it changes more answers, so Q8_0 is the better choice there. The original rows are the same weights run through this
repository's package on the same machine; they differ from the tables above by one or two items of bf16 rounding.
StartLux-Decision-35B-A3B is not available as GGUF yet.

To run one, download its repository (the GGUF file plus the small files the decision server needs), then start
llama.cpp and the server (for another precision, replace `Q8_0`):

```bash
hf download startlux-models/StartLux-Decision-4B-Q8_0-GGUF --local-dir StartLux-Decision-4B-Q8_0-GGUF
cd StartLux-Decision-4B-Q8_0-GGUF && pip install -r requirements.txt
llama-server -m StartLux-Decision-4B-Q8_0.gguf -ngl 99 -c 16384 --parallel 4 --port 8081 &
python -m startlux_decision.gguf_server --model-dir . --llama http://127.0.0.1:8081 --port 8090
```

## Games

<p align="center"><img src="media/games.png" alt="StartLux-Decision and Jev in the game harnesses" width="100%"></p>

<div align="center">

| Game | Measure | StartLux-Decision-27B | Jev 1.13 | Other reference |
|:---:|:---:|:---:|:---:|:---:|
| NPC addressee detection, clean text | lines with a wrong answer, of 75 (fewer is better) | 1 | 6 | name matching: 27 |
| NPC addressee detection, misheard names | lines with a wrong answer, of 75 (fewer is better) | 5 | 12 | name matching: 29 |
| NPC addressee detection, clean text | F1 over the yes/no answers | 0.990 | 0.962 | name matching: 0.820 |
| Chess, against each other | points in 256 games at the rich level, no search | 149.5 | 106.5 | 64 wins, 171 draws, 21 losses |
| Mate in one | puzzles solved, of 25 | 10 | 6 | a random legal move: 3% |
| Dino Run | runs that reach the 300-obstacle cap, of 20 | 20 | 20 | StartLux-Decision-4B and 9B: 20 |

</div>

Jev's numbers are the ones its harness authors report, except Dino Run and the chess match, which we ran for Jev
through its API in the same harness as ours. The chess match is 256 games: 128 level openings (within 0.6 pawns by
Stockfish), each played once with each colour, at the harness's default rich level. Games end by the rules or are
adjudicated at ±300 centipawns after 160 plies; since neither side searches, most draws are fivefold repetitions.
StartLux-Decision-27B scores 58.4%, +59 Elo (95% interval +37 to +82), and wins 64 of the 85 decisive games. With the
harness's tactical hints, which give both sides the one-move consequences of every move, the two are even (48.6% and
46.1% over 256 games each); all four conditions are in [docs/results.md](docs/results.md). The misheard-names variant is the same set
of lines as a lower-case speech-to-text transcript in which names are misheard.
The 0.8B and 2B models do much worse on these games. Every game, position and line is in
[results/games](results/games): Elo ladder games with PGN, chess positions and mate-in-one answers, and NPC predictions
per line in all three transcript variants.

## Quick start

The weights are on Hugging Face, as the original checkpoints for this package and as GGUF files for llama.cpp (see [GGUF](#gguf)). Each model folder also carries the inference package from this repository. Every repository is also on ModelScope, with the same name and the same files, under [StartLuxAI](https://modelscope.cn/organization/StartLuxAI): `https://modelscope.cn/models/StartLuxAI/<repository name>`.

<div align="center">

| Model | Original weights | GGUF Q8_0 (recommended) | GGUF Q4_K_M | GGUF BF16 |
|:---:|:---:|:---:|:---:|:---:|
| StartLux-Decision-0.8B | [1.8 GB](https://huggingface.co/startlux-models/StartLux-Decision-0.8B) | [0.81 GB](https://huggingface.co/startlux-models/StartLux-Decision-0.8B-Q8_0-GGUF) | [0.53 GB](https://huggingface.co/startlux-models/StartLux-Decision-0.8B-Q4_K_M-GGUF) | [1.52 GB](https://huggingface.co/startlux-models/StartLux-Decision-0.8B-BF16-GGUF) |
| StartLux-Decision-2B | [4.6 GB](https://huggingface.co/startlux-models/StartLux-Decision-2B) | [2.01 GB](https://huggingface.co/startlux-models/StartLux-Decision-2B-Q8_0-GGUF) | [1.27 GB](https://huggingface.co/startlux-models/StartLux-Decision-2B-Q4_K_M-GGUF) | [3.78 GB](https://huggingface.co/startlux-models/StartLux-Decision-2B-BF16-GGUF) |
| StartLux-Decision-4B | [9.3 GB](https://huggingface.co/startlux-models/StartLux-Decision-4B) | [4.48 GB](https://huggingface.co/startlux-models/StartLux-Decision-4B-Q8_0-GGUF) | [2.71 GB](https://huggingface.co/startlux-models/StartLux-Decision-4B-Q4_K_M-GGUF) | [8.42 GB](https://huggingface.co/startlux-models/StartLux-Decision-4B-BF16-GGUF) |
| StartLux-Decision-9B | [19.4 GB](https://huggingface.co/startlux-models/StartLux-Decision-9B) | [9.53 GB](https://huggingface.co/startlux-models/StartLux-Decision-9B-Q8_0-GGUF) | [5.63 GB](https://huggingface.co/startlux-models/StartLux-Decision-9B-Q4_K_M-GGUF) | [17.92 GB](https://huggingface.co/startlux-models/StartLux-Decision-9B-BF16-GGUF) |
| StartLux-Decision-27B | [55.6 GB](https://huggingface.co/startlux-models/StartLux-Decision-27B) | [28.60 GB](https://huggingface.co/startlux-models/StartLux-Decision-27B-Q8_0-GGUF) | [16.55 GB](https://huggingface.co/startlux-models/StartLux-Decision-27B-Q4_K_M-GGUF) | [53.81 GB](https://huggingface.co/startlux-models/StartLux-Decision-27B-BF16-GGUF) |

</div>

On AMD GPUs, follow [AMD ROCm](docs/inference.md#amd-rocm) to install ROCm PyTorch and build the fused kernels before
continuing. PyTorch continues to use the `cuda` device name on ROCm.

```bash
hf download startlux-models/StartLux-Decision-4B --local-dir StartLux-Decision-4B
# or, from ModelScope: modelscope download StartLuxAI/StartLux-Decision-4B --local-dir StartLux-Decision-4B
pip install -r requirements.txt
python -m startlux_decision.check StartLux-Decision-4B          # must print "fast kernels: active" (on a Mac: MLX)
python -m startlux_decision.server --model StartLux-Decision-4B --port 8090
```

On Apple Silicon the same commands run the model with MLX; add `--int8` on an M5 or later. See
[Apple Silicon (MLX)](docs/inference.md#apple-silicon-mlx).

```bash
curl -s localhost:8090/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "I was charged twice for order #4411 and the app still shows it as unpaid."},
  "questions": {
    "team":   {"type": "choice", "instructions": "Which team should handle this ticket?",
               "criteria": {"billing": "Payments, refunds and invoices",
                            "shipping": "Delivery and tracking",
                            "technical": "App, login and account problems"}},
    "urgent": {"type": "noul", "instructions": "Should this ticket be answered today?"},
    "severity": {"type": "score", "instructions": "How severe is the impact?",
                 "criteria": ["cosmetic", "annoying", "blocks the customer"]}
  }
}'
```

Or in Python:

```python
from startlux_decision import StartLuxDecision

m = StartLuxDecision("StartLux-Decision-4B")
answers, usage = m.decide(state, questions)          # one request
many = m.decide_batch([(state, questions), ...])     # many requests, batched together
```

[docs/inference.md](docs/inference.md) covers the request format, the speed-ups and FP8.
[docs/evaluation.md](docs/evaluation.md) explains how to reproduce every number above, including the batched Decision
Index runner. [docs/finetuning.md](docs/finetuning.md) shows how to adapt a model to your own decisions.

## Layout

```
startlux_decision/       inference: prompt rendering, letter readout, GPU graphs, long inputs, images, MLX for Apple Silicon, HTTP servers (also for GGUF), kernel check
demos/          the computer-use harness and its two mock sites, and the chess match tools
eval/           evaluation: Intern-Decision suites and JevBench public tiers, Typed Decisions, Decision Index, latency
finetune/       LoRA fine-tuning on your own data and temperature calibration
docs/           inference, evaluation, fine-tuning and full results
results/        our numbers as JSON and CSV, per-benchmark Decision Index scores, raw game and computer-use logs
media/          figures and recordings used here
```

## License

The code in this repository is Apache-2.0 ([LICENSE](LICENSE)). The model weights on Hugging Face and ModelScope are released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): free for research and other non-commercial use, with
attribution. Commercial use requires a separate license from StartLux Labs; contact
[contact@startlux.com](mailto:contact@startlux.com). Benchmark data is fetched from its original sources under their own
terms.
