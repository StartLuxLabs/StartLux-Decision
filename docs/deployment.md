# Deployment

What the server and the engines do with images, long prompts and the two model releases, and which engine settings to switch on or off. Every setting below was measured; [engines.md](engines.md) and [quantization.md](quantization.md) have the reasons and the numbers.

## Deployment settings

`launch-vllm` reads the vLLM version and picks the flags below; on vLLM 0.31 the settings are those of the reference
deployment (prefix caching and chunked prefill on, 8,192 batched tokens per step).

| engine | setting | on or off | why |
|---|---|---|---|
| vLLM 0.19.1 | `--no-enable-prefix-caching` | prefix caching off | on 0.19.1 it moves 2.6% of the read-outs of these hybrid models by more than 0.05, without an error |
| vLLM 0.19.1 | `--no-enable-chunked-prefill`, `--max-num-batched-tokens` equal to `--max-model-len` | chunked prefill off | on 0.19.1 a prompt split across scheduler steps is misread |
| vLLM 0.31 | `--enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192` | on for AutoThink and v1 | on 0.31 prefix caching works on these models only with chunked prefill, and with both on the 27B AutoThink model gave the same answer as on 0.19.1 for 99.88% of 823 questions; a thought no longer re-reads its prompt |
| vLLM 0.31 | `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` | on when the folder holds MTP weights; `--no-mtp` turns it off | the MTP layer proposes 3 tokens per step, so thoughts are generated faster; the thoughts kept their length distribution in our check at 4B, 9B and 27B |
| vLLM 0.31 | thought as one streamed request (`--thought-generation`, default `auto`) | on | no chunk re-sends the prompt and the thought so far; the loop detector still looks at the thought every 1,024 tokens and ends a looping one there |
| vLLM, Instant | prefix caching and chunked prefill off, no MTP, on every version | required | the Instant plugin needs every prompt prefilled whole in one step |
| vLLM | `--logprobs-mode processed_logprobs --max-logprobs 26` | on | the read-out returns exactly the option letters |
| vLLM | `--generation-config vllm` | on | keeps the folder's top-k and top-p away from the read-out |
| vLLM | prompt-bidirectional plugin (`launch-vllm` sets `STARTLUX_PREFIX_BIDIR=1`), vLLM 0.19.1 or 0.31.0 | on for Instant | Instant models were trained to read the prompt bidirectionally; without it they answer as a different model |
| vLLM | thought waves (`--waves` / `--no-waves`; default: on for servers older than 0.31) | on with vLLM 0.19.1 | on 0.19.1 a thought that joins a running decode batch can turn into a loop; 0.31 fixed this |
| vLLM | `--quantization fp8` | optional, 27B only | within 0.2 points at 27B; costs the 4B Instant model 1.3 DecideBench points and shifts the decision to think |
| vLLM | `--kv-cache-dtype fp8` | off | refused by the plugin, and it shifts the decision to think |
| vLLM 0.31 | `--limit-mm-per-prompt '{"image":8,"video":0}' --mm-processor-kwargs ... --trust-request-chat-template` | on for folders with a vision tower; `--no-images` turns it off | image calls carry the client's pass-through chat template, and the server resizes images within the package's bounds; Instant servers also get `--skip-mm-profiling` (with vLLM's profile of 256 full-size images the 4B Instant server did not start) |
| vLLM | `--max-model-len` | on 0.31 the folder's native 262,144, Instant 131,072; on 0.19.1 65,536 | an Instant prompt is read in one step, and on 0.31 that step crashed vLLM at 196,000 (27B) and 256,000 tokens (4B); prompts beyond the limit get a 422 |
| llama.cpp | GGUF converted with [`patches/llama.cpp/0002`](../patches/llama.cpp) (or `--override-kv qwen35.attention.causal=bool:false`) | on for Instant | marks the prompt as bidirectional; read causally, the 4B Instant model loses 27 DecideBench points |
| llama.cpp | stock llama-server: `--parallel 1 --ctx-checkpoints 0 -b 16384 -ub 16384`; or [`patches/llama.cpp/0001`](../patches/llama.cpp) with any `--parallel` | required for Instant | otherwise prompts are batched or split, which costs the 4B Instant model 7.7 DecideBench points |
| llama.cpp | prompt cache (`cache_prompt`) | off (the backend sends false) | a reused prefix would break the bidirectional reading |
| llama.cpp | GPU sampling (`-bs`) | off | it changed read-outs by up to 0.40 in log-probability |
| llama.cpp | quantization | Q8_0 recommended | Q4_K_M costs 1.6 to 3.8 DecideBench points at 2B to 9B |

## Images and long inputs

A request can carry images as part of its evidence, and a prompt can run to 262,144 tokens (256K), the models' native
context. This is v1's request format.

```bash
curl -s localhost:8090/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Photo taken at delivery: <image>",
  "questions": {"damaged": {"type": "noul", "instructions": "Is the parcel damaged?"}},
  "images": ["data:image/jpeg;base64,'"$(base64 < parcel.jpg | tr -d '\n')"'"]
}'
```

- `images` is a list of base64 strings or data URIs. `<image>` in the state marks where each image goes, one mark per
  image, in order. Without marks the images come first (as `Image 1:`, `Image 2:` ... when there are several),
  followed by the state.
- Each image is resized to at most 1,048,576 pixels, keeping its aspect ratio (`--max-pixels`), and becomes one token
  per 32 × 32 pixels: a 1024 × 1024 photo is 1,024 tokens. Every question of the request reads the same images. At
  most 8 images per request (`--max-images`); `--no-images` serves text only.
- Images work on vLLM 0.31 or later, on the transformers engine, on llama.cpp (llama-server started with the model's
  `mmproj-*-F16.gguf` file; without it a request with images gets a 422) and on MLX (mlx-vlm folders), for v1 and
  v1.1 folders (all of them keep the vision tower).
- Checked on vLLM 0.31 against the transformers engine with the same 12 image questions: the same top option in 59 of
  60 read-outs over 4B and 27B AutoThink, 4B and 27B Instant and v1 4B, largest probability difference 0.04 to 0.09
  per model. The same questions without their images differ by at most 0.012, and both engines' image processors give
  identical pixels: the image differences come from the two engines' vision paths (also in the causal folders, which
  do not use the Instant plugin). Read-outs with images are therefore close to, not equal to, the reference.

Long inputs:

- On vLLM 0.31 the servers accept prompts up to the folder's native 262,144 tokens, and Instant servers up to 131,072
  (`launch-vllm` defaults; on 0.19.1 65,536 as before). `--max-model-len` sets the serving limit (65,536 is a
  good choice on 80 GB GPUs); a longer prompt gets a 422 with the engine's reason.
- AutoThink and v1 folders read a long state once per request: the package reads the beginning that the questions
  share into vLLM's prefix cache, sends every question of the request to that server, and each question then reads
  only its own part. A thought reads only its own part too with `serve --first-read thinking` (docs/api.md); with
  the default `plain`, the first thought of a request reads the whole state again.
- Instant folders read every prompt whole, in one step, because their prompt attention is bidirectional: a request
  with n questions reads its state n times, and one step must hold a whole prompt. On vLLM 0.31 that step crashed the
  server for a 256,000-token prompt (4B) and a 196,000-token prompt (27B), which is why Instant servers stop at
  131,072 tokens.
- What a long prompt costs, one H200 per server, the answer in the first lines of the state, nothing cached before:

  | model | 32,000 tokens | 128,000 | 256,000 | the same state again |
  |---|---|---|---|---|
  | 4B AutoThink | 0.65 s | 3.8 s | 11.0 s | 0.14 / 0.45 / 0.96 s |
  | 27B AutoThink | | 15.9 s | 42.0 s | 0.46 / 1.04 s |
  | 4B Instant | 0.78 s | 7.3 s | | every question reads it again |
  | 27B Instant | | 24.7 s | | every question reads it again |

  Every answer was right. A second question over the same state added 0.2 to 1.3 s to the causal models and nearly
  doubled the time of the 4B Instant model (1.5 s at 32,000 tokens, 12.7 s at 128,000). At the full context the 4B AutoThink server's cache holds 13.7 prompts of 262,144
  tokens, the 27B's 4.2 ([engines.md](engines.md) has the details).

## Compatibility

The engine serves the decisions of both releases, with images and long inputs:

| release | folders | how it answers |
|---|---|---|
| v1.1 | Instant and AutoThink folders (0.8B to 27B) | as above |
| v1 | `decision_config.json` with `"format": "StartLux-Decision-v1"` (0.8B, 2B, 4B, 9B, 27B and 35B-A3B) | one pass by default, with the same prompts as v1.1. A request may still set `think_mode` to `auto` or `always`, but v1 was not trained to think |

Images work for both releases on vLLM 0.31, transformers, llama.cpp (with an mmproj file) and MLX (mlx-vlm folders),
and the 256K context on vLLM 0.31 and transformers. The v1 4B folder, served by this engine, read 12 image questions
with the same top option as the transformers engine and answered a 128,000-token request correctly in 4.1 s.

## Published numbers

The published numbers were measured on vLLM 0.19.1 with the defaults there: the vLLM flags of `launch-vllm`, the
read-out temperatures of each folder, and for AutoThink the gate 0.9, budget 4,096, thought chunks of 1,024 tokens with
the loop detector between them, the last 4,000 characters of the thought read after it, think temperature 1.0 and the
mixture 0.75. On vLLM 0.31 the defaults change how the thought is produced, not the decision rule: the thought is one
streamed request, prefix caching is on and MTP proposes 3 tokens per step (see Deployment settings). The engine's default
gate is now 0.6, so serve with `--gate 0.9` on vLLM and use `startlux-decide eval` on the benchmark's requests;
[benchmarks.md](benchmarks.md) describes the measurement of accuracy and latency, and
[engines.md](engines.md) lists where this package differs from the code of those runs (answers' confidence
field, usage counts and edge cases; never the probabilities).

