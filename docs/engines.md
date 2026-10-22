# Engines

Every engine implements two primitives (`startlux_decide/backends/base.py`):

* `readout(prompt, letter_ids)`: one forward pass, the log-probabilities of the option letters as the next token;
* `generate(prompt, max_tokens, sampling, stop, seed)`: a sampled continuation, for the thoughts of AutoThink models.

Engines that stream a thought safely (vLLM 0.31 or later) also implement `generate_stream`, the same continuation as
one streamed request; closing the stream early ends the request in the engine.

Prompt rendering, temperatures, the gate, the thought loop, the mixture, knock-out lists and the answer format are
the package's own code and the same for every engine.

## vLLM (default)

vLLM 0.19.1 (the version of the release runs) or 0.31.0, one OpenAI-compatible server per GPU. Both versions are
validated with the Instant plugin ([vllm_bidirectional.md](vllm_bidirectional.md#vllm-versions)).
`startlux-decide launch-vllm` (or `serve --launch-vllm`) reads the vLLM version of the Python that runs vLLM and starts
the servers with these flags:

```
all:      --gpu-memory-utilization 0.88 --logprobs-mode processed_logprobs --max-logprobs 26
          --generation-config vllm --served-model-name m
          --max-model-len: 0.31 or later 262144 (the folder's max_position_embeddings), Instant folders 131072;
                           0.19.1 65536; --max-model-len N sets it
0.19.1, and Instant folders on every version:
          --no-enable-prefix-caching --no-enable-chunked-prefill --max-num-batched-tokens <max-model-len>
AutoThink and v1 folders on 0.31 or later:
          --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192
          --speculative-config '{"method":"mtp","num_speculative_tokens":3}'   (folders with MTP weights; --no-mtp: off)
images, on 0.31 or later for folders with a vision tower (--no-images: text only):
          --limit-mm-per-prompt '{"image":8,"video":0}' --trust-request-chat-template
          --mm-processor-kwargs '{"size":{"longest_edge":1048576,"shortest_edge":65536}}'
          (Instant folders also --skip-mm-profiling)
text only (0.19.1, --no-images, folders without a vision tower):
          --limit-mm-per-prompt '{"image":0,"video":0}'
```

| flag | why |
|---|---|
| `--logprobs-mode processed_logprobs`, `--max-logprobs 26` | a read-out asks for the log-probabilities of up to 26 letters with `allowed_token_ids`; processed log-probabilities are taken after that mask, so exactly the letters come back. With raw log-probabilities the top 26 of the whole vocabulary come back and letters go missing |
| `--generation-config vllm` | otherwise vLLM applies the folder's generation_config (top_k, top_p) to every request, the read-out included, and letters outside the top k get no probability |
| `--no-enable-prefix-caching` (0.19.1, Instant) | on vLLM 0.19.1 prefix caching changes answers of these hybrid models (Gated DeltaNet layers) silently: in our tests it moved 2.6% of ordinary and 16% of knocked-out read-outs by more than 0.05 |
| `--no-enable-chunked-prefill` (0.19.1, Instant) | on 0.19.1 chunked prefill also changed read-outs of long prompts silently, and the Instant plugin needs every prompt prefilled whole |
| `--max-num-batched-tokens` = `--max-model-len` (0.19.1, Instant) | a whole prompt fits in one step |
| `--enable-prefix-caching --enable-chunked-prefill` (AutoThink and v1 on 0.31) | on vLLM 0.31 prefix caching works on these hybrid models only together with chunked prefill. With both on, the 27B AutoThink model gave the same answer as on 0.19.1 for 99.88% of 823 questions (0.12% of read-outs moved by more than 0.05), and a thought request no longer re-reads the prompt it shares with the first read-out |
| `--max-num-batched-tokens 8192` (AutoThink and v1 on 0.31) | the prefill chunk; the setting of the reference deployment |
| `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` (0.31, folders with MTP weights) | multi-token prediction: the MTP layer proposes 3 tokens per step and the model checks them, so a thought is generated faster with the same sampling distribution. A folder has MTP weights when its config declares `mtp_num_hidden_layers` and it holds `mtp.*` tensors; on 4B, 9B and 27B with 3 draft tokens the thoughts kept their length distribution in our check. `--no-mtp` turns it off. The Instant plugin refuses speculative decoding |
| `--limit-mm-per-prompt '{"image":0,"video":0}'` (text only) | no memory for the vision tower's inputs |
| `--limit-mm-per-prompt '{"image":8,"video":0}'`, `--mm-processor-kwargs`, `--trust-request-chat-template` (images) | at most 8 images per prompt (`--max-images`), resized within the bounds of the package (`--max-pixels`); an image call carries the client's pass-through chat template (below) |
| `--max-model-len` | on 0.31 the native context of the folders, 262,144 tokens, and 131,072 for Instant folders (their one-step prefill of a 196,000 to 256,000-token prompt crashed vLLM 0.31, see [vllm_bidirectional.md](vllm_bidirectional.md#known-limits)); a lower serving limit (for example `--max-model-len 65536` on 80 GB GPUs) costs nothing for shorter prompts |

At start the vLLM client reads the served model name and context length from `/v1/models` and sends a probe read-out
over all 26 letters; a server without the read-out flags is refused with the flags to add. It cannot see the caching
flags: servers started by hand must use the flags above.

**Thoughts on vLLM.** At start the client reads every server's version from `/version`. On vLLM 0.31 or later a
thought is one streamed request (`thought_generation` `stream`, the default there). The loop detector looks at the
thought each time another 1,024 tokens have arrived and once more when the budget ends it, the points where the
chunked path runs it; a looping thought is cut as in the chunked path (the reasoning and the start of the loop are
kept) and its request is closed, which ends it in vLLM. With MTP an event of the stream can carry up to 4 tokens, so a
check can run up to 3 tokens past 1,024, 2,048 or 3,072. On older vLLM (or with `thought_generation` `chunked`) a
thought is generated 1,024 tokens at a time; each chunk re-sends the prompt and the thought so far with its own seed,
and the loop detector runs between chunks. Both use the prompt with the forced start, the seed of the first chunk (from
a hash of the thinking prompt; later chunks add 7,919 each), temperature 0.6, top-p 0.95, top-k 20, the stop string
`</think>`, the budget and the CUT line. All calls of one question go to the same server; questions are spread round
robin over the servers.

**Waves (vLLM 0.19.1).** On vLLM 0.19.1 the state of hybrid models gets corrupted when new generation requests join a
decode batch that is already running: in our tests 25 to 35% of thoughts turned into garbage loops. The client
therefore sends the chunked thought requests of each server in waves: everything queued goes out together, and the
next wave starts only when the whole wave has finished. With waves no thought was corrupted (0 of 183). A request waits
up to 0.3 s for its wave to fill (`--wave-gather`) and at most 128 requests go in a wave (`--wave-max`). Read-outs are
prefill only and are sent at any time. vLLM 0.31 fixed the corruption (0 of 183 thoughts unclosed at 64 requests in
flight, without waves), so waves are on only for servers older than 0.31 or that do not report a version;
`--waves` and `--no-waves` decide instead. A request that asks for `stream` on such a server gets a 422.

**Images on vLLM** (0.31 or later). A call whose prompt holds images goes to `/v1/chat/completions` instead of
`/v1/completions`: one user message whose parts are the pieces of the package's rendered prompt with the images (data
URIs) between them, and a chat template that writes the parts back to back and the image placeholder
(`<|vision_start|><|image_pad|><|vision_end|>`) for each image. vLLM therefore reads exactly the package's prompt; its
image processor resizes each image within the server's bounds (`--mm-processor-kwargs`, which the launcher sets to the
package's 65,536 to 1,048,576 pixels; the package counts the tokens with the same rule), puts each image's tokens at its
placeholder and gives them their multimodal positions. The servers need `--trust-request-chat-template` (the template
comes with each call). Read-outs, thoughts (chunked or streamed) and prefix-cache fills all take this path when the
request has images. On Instant servers the plugin reads the image tokens bidirectionally like every other prompt token.

**Long prompts.** The servers take prompts up to `--max-model-len` (on vLLM 0.31: 262,144 tokens for causal folders,
131,072 for Instant folders). A longer prompt is refused by vLLM, and the request gets a 422 with vLLM's reason. Causal folders (AutoThink, v1) read
long prompts in steps of 8,192 tokens (chunked prefill) into the prefix cache; when the questions of a request share a
beginning of 4,096 tokens or more (their state), the package reads that beginning once (a one-token request) and then
sends every question of the request to the same server, so each question reads only its own part, and a thought reads
neither the prompt nor the thought so far again. Instant folders must read each prompt whole in one step: their servers
run with `--max-num-batched-tokens` equal to the context and no prefix cache, so a request with n questions over a long
state reads the state n times, and one step must fit a whole prompt.

Measured on vLLM 0.31, one H200 per server (five servers side by side on one node), a decision whose answer is in the
first lines of a long state, the state not yet in any cache:

| model | 32,000 tokens | 128,000 | 256,000 | two questions over the same state | the same request again |
|---|---|---|---|---|---|
| 4B AutoThink | 0.65 s | 3.8 s | 11.0 s | 0.81 / 4.4 / 12.1 s | 0.14 / 0.45 / 0.96 s |
| 27B AutoThink | | 15.9 s | 42.0 s | 16.4 / 43.3 s | 0.46 / 1.04 s |
| v1 4B | | 4.1 s | | 4.7 s | 0.46 s |
| 4B Instant | 0.78 s | 7.3 s | crashed (see above) | 1.5 / 12.7 s | (no cache) |
| 27B Instant | | 24.7 s | not at 196,000 | | (no cache) |

Every answer was right (the choice with probability 0.997 or more). The second question of a causal request costs
the reading of its own part; at 256,000 tokens the package's own work (rendering and counting the prompts of each
question) is most of the 0.96 s of a request whose state is cached. KV cache at the native context (0.88 of the GPU):
4B AutoThink 3.60 million tokens (13.7 prompts of 262,144), 27B AutoThink 1.09 million (4.2 prompts), v1 4B 3.11
million; an Instant server reserves the activations of a whole-prompt step on top (4B at 262,144: 2.96 million
tokens of cache; 27B at 131,072: 0.79 million).

**One process.** The decision server is one Python process; prompt rendering and token counting run in its event loop
and cost about 0.4 ms per question with a short state and about 10 ms per question of 8,000 prompt tokens (measured
with the models' tokenizer on one CPU core). For high request rates over many GPUs, run several `serve` processes, each
with its share of the vLLM servers, behind a load balancer, as the release runs did (one per GPU).

**Instant folders on vLLM** need the plugin in this package, which makes vLLM read the prompt bidirectionally in the
full-attention layers. It is a vLLM general plugin (entry point `startlux_prefix_bidir`), active when the folder is
prompt-bidirectional and `STARTLUX_PREFIX_BIDIR=1` is set. The launcher sets the variable, checks that vLLM's Python
sees the plugin, and refuses to continue unless every server log shows "prefix-bidirectional prompt attention ON".
Without the plugin vLLM reads such a folder causally, without notice, and answers as a different model. vLLM 0.31
runs a new model runner by default (V2); the plugin patches it as well as the V1 runner, and a vLLM worker whose runner
it did not patch refuses an Instant folder at start. Details and test results:
[vllm_bidirectional.md](vllm_bidirectional.md).

## transformers

In-process reference on PyTorch (`--backend transformers`, extra `[transformers]`, transformers 5.8). One prompt per
forward pass, one call at a time: for checks, not for serving load. Images go through the folder's processor with the
same resize bounds and the model's vision tower; an Instant folder reads the result (image features in place,
multimodal positions) with its bidirectional prompt attention. Instant folders are read with the bidirectional
prompt attention they were trained with (`backends/bidirectional.py`, checked on the model at start: the causal path
equals the stock forward, padding never reaches a token, a late token reaches the first position). Thoughts use
transformers' sampler, so they are not token-identical to vLLM's for the same seed. On CUDA install
flash-linear-attention and causal-conv1d; without them transformers runs the linear-attention layers on a plain
PyTorch path that is more than ten times slower. On a machine without a GPU those two packages must not be installed
(they would be called on CPU tensors).

## llama.cpp

`--backend llamacpp --llamacpp-url URL` puts the package in front of llama-server for GGUF builds; the model folder
still provides the tokenizer and `decision_config.json`. A read-out sends the token ids with `n_predict` 1,
`post_sampling_probs`, a logit bias of +100 on the k option letters and `top_k` = k, so the sampler keeps exactly the
letters and their probabilities are the softmax over the letters alone (the same bias on every letter does not change
their relative probabilities); every other sampler is neutral. Thoughts use the sampling settings of vLLM and its
sampler order (temperature first, then top-k and top-p; llama.cpp's default applies top-p first). Prompt caching is
off. Tested with llama.cpp b11549.

**Images on llama.cpp.** Start llama-server with the model's mmproj file (`--mmproj mmproj-...-F16.gguf`, the F16 file in
the same repository as the GGUF) and `--image-min-tokens 64 --image-max-tokens 1024` (the package's pixel bounds in
tokens). `/props` then reports the vision modality and the server's media marker, and the backend takes images
(`engine.images` in `/health` is true). A call with images sends `"prompt": {"prompt_string": ..., "multimodal_data":
[...]}`: the rendered prompt as text with the media marker at each image's place, and the images as base64 PNGs. llama.cpp
tokenizes the text itself, because an object prompt cannot carry token ids (its count equalled the Hugging Face
tokenizer's on every image request of the smoke test), and adds `<|vision_start|>` and `<|vision_end|>` around each
image. The backend has already resized every image with PIL (bicubic) to the processor size, so the server's own resize
does nothing and the pixels are the ones vLLM reads. After every image call it compares the server's `tokens_evaluated`
with its own count (text tokens, plus 2 and the image's tokens for each image; `/tokenize` tells a difference in the text
from one in the images) and raises when they differ, for example when the server was started with a smaller
`--image-max-tokens` and resized an image. Without `--mmproj`, a request with images gets a 422. Text-only calls are what
they were.

**Instant folders** need non-causal attention over the prompt in the full-attention layers. A released Instant GGUF
carries `qwen35.attention.causal = false` (written by `patches/llama.cpp/0002-...`), and the prompt must be evaluated
alone and in one micro-batch. `STARTLUX_LLAMACPP_BIDIR` says how the server provides this:

| value | server | what the backend does |
|---|---|---|
| `auto` (default) | any | reads `/props` once at start: a server that reports patch 0003 (`mtmd_prompt_whole_when_non_causal`) works as `server`, any other as `request` |
| `request` | llama-server with `patches/llama.cpp/0001-...`, any `--parallel` | sends `prompt_bidirectional` with every read-out and checks the echo; a stock server is an error |
| `server` | stock llama-server with `--parallel 1 --ctx-checkpoints 0 -b 16384 -ub 16384` | relies on the non-causal context of the file |
| `off` | any | reads the prompt causally: a different model, for measuring that cost only |

A stock llama-server reports nothing about its context, so `auto` takes it for a `request` server and the first read-out
of an Instant folder fails with a message that names `server`: set `STARTLUX_LLAMACPP_BIDIR=server` for it. The mode in
use is `engine.bidir_mode` in `/health`.

**Instant folders with images.** The text and the image embeddings of a prompt must be read in one micro-batch with
non-causal attention. Use mode `server` (`auto` picks it for a server with patch 0003; for any other, set
`STARTLUX_LLAMACPP_BIDIR=server`) with a non-causal GGUF and `--parallel 1 --ctx-checkpoints 0 -b N -ub N`, N at least the
longest prompt including its image tokens. The per-request mode (`request`, patch 0001) does not cover images and is
refused with a 422. Stock b11549 splits such a prompt into three micro-batches (text, image, text): on 32 image
questions the 4B Instant Q8_0 read-outs then differ from BF16 by 0.022 on average (largest 0.22), and with
`patches/llama.cpp/0003-server-mtmd-prompt-whole-on-non-causal-context.patch` by 0.0055 (largest 0.06). The patched
server says so in `/props` (`mtmd_prompt_whole_when_non_causal`); the backend warns, and reports
`image_prompt: split (stock server)` in `/health`, when an Instant folder with images is served without it.

Measured on one H200 against vLLM BF16 (823 questions): BF16 GGUF top-1 agreement 99.5 to 100% (two vLLM runs: 99.9%),
Q8_0 99.4 to 99.6%, Q4_K_M 97.0 to 99.2%. On DecideBench the 4B Instant model loses 0.6 points as Q8_0, 3.1 as Q4_K_M,
7.7 on a stock server with default settings and 26 when its prompt is read causally ([benchmarks.md](benchmarks.md)).
llama.cpp answers short Instant requests faster than vLLM and long ones up to three times more slowly, because it
evaluates the bidirectional prompts of a request one at a time.

## MLX

`--backend mlx` (extra `[mlx]`, Apple Silicon) runs the folder's own decoder in mlx-vlm and builds the input itself.
It reads mlx-vlm folders (`mlx_vlm.convert`: the text model 8-bit or 4-bit, the vision tower kept in BF16), the BF16
release folders, and text-only MLX folders (images then get a 422). Instant folders are read with the bidirectional
prompt attention they were trained with (the full-attention layers see every token of the one-pass input, the
linear-attention layers stay causal; a self-check at load time confirms it). AutoThink thoughts are sampled with the
request's settings and seed. Images follow the vLLM flow: the folder's image processor, the vision tower's features at
the image tokens, the processor's three-axis positions. One call runs at a time.

## Parity with the release runs

The golden test (`tests/test_golden.py`) runs this package's pipeline and the per-question pipeline of the runs behind
the published AutoThink numbers side by side on a fake vLLM, over random requests and settings: the requests sent to
vLLM (every read-out prompt, thought chunk, seed and sampling setting) and the probabilities, choices, scores and thought
token counts are identical, bit for bit. Instant read-outs are the one-pass read-out of those runs (the same path with
thinking off). Deliberate differences, none of which changes a probability of an answered question:

| | release runs | this package |
|---|---|---|
| `confidence` of choice and score answers | the top probability | TypeSafe's definitions (above) |
| score `legend` when criteria is `{key: level}` | levels in the order given | levels in the order they are shown to the model (numeric keys ascending) |
| choice with one option | refused (422) | answered with probability 1 |
| `mix` 0 | meant "the read-out after the thought only" | means "the first read-out only" (the formula) |
| `usage.input_tokens` of knocked-out lists | the final round's prompt | every round's prompts |
| request controls | `think_mode` only, unknown values ignored | `think_mode`, `gate`, `think_budget`, `mix`, `thought_generation`, `first_read`, details; bad values give 422 |
| thoughts on vLLM 0.31 | (not run) | one streamed request (the chunked path on request); the golden test runs the chunked path, and `tests/test_stream.py` shows that with one token per stream event the streamed path gives the same thoughts and probabilities as the chunked one on the fake vLLM |
| malformed body, empty `questions` | 400 | 422, FastAPI-style `detail` |
| `cache_mode`, `--rep-pen`, `--stop-loops`, `--loop-version` 1 and 2 | test switches | not carried over (the release used none of them) |
| backend errors | 422 for a refusal, otherwise an unstructured 500 | 422 for a refusal, 502 after four tries |
