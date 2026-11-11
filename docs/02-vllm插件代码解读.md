# StartLux-Decision · vLLM 插件代码解读

> 范围：默认 serving 栈上的 **vLLM general plugin**。  
> 主代码：`startlux_decide/vllm_plugin.py`  
> 文档正本：`docs/vllm_bidirectional.md`、`docs/engines.md`  
> 客户端：`startlux_decide/backends/vllm.py` + `launch.py`  
> 这条路径**使用** vLLM 的调度、FlashAttention、**paged KV**；插件只改 Instant 的 prompt 注意力 mask。

---

## 1. 结论先钉死

1. StartLux **默认推理吃 vLLM**（`startlux-decide serve` / `launch-vllm`），不是「整仓都自研 transformers」。  
2. 插件名：`startlux_prefix_bidir`（entry point `vllm.general_plugins`）。  
3. 插件**不是**重写推理引擎：不实现 paged attention，不换 KV 布局，不动 Gated DeltaNet / 短卷积。  
4. 它只做一件事：让 Instant 文件夹在 **full-attention 层**上按训练时的 **prompt 双向** mask 读 prompt；thought / decode 仍因果。  
5. **Paged attention：走这条路径时就是 vLLM 自带的，支持。** Instant 为了整段 prompt 一步算完，会关掉 chunked prefill 和 prefix caching——关的是这两项，不是 paged attention。

---

## 2. 何时生效

同时满足：

| 条件 | 含义 |
|---|---|
| 环境变量 `STARTLUX_PREFIX_BIDIR=1` | 未设则插件完全空操作 |
| 模型 `config.json` 里 `jev_prompt_attention: "bidirectional"` | Instant；因果文件夹（AutoThink / v1）即使变量开着也跑 stock |
| 包装进跑 vLLM 的同一个 Python | entry point 被 engine-core / worker 加载 |
| 若设了 `VLLM_PLUGINS` | 必须包含 `startlux_prefix_bidir` |

Launcher 会设变量、检查 log 里出现  
`prefix-bidirectional prompt attention ON for <DIR>`，否则拒绝对 Instant 继续。

因果模型 + 插件已加载但未匹配 bidirectional：**输出与 stock vLLM bit-identical**（官方测试覆盖）。

---

## 3. Mask 语义（人话）

只在 **full-attention 层**（约每 4 层一层），同一个 request 内：

```text
query q 可以读 key k  ⟺  k ≤ q   或   (q < p 且 k < p)
```

- `p` = prompt end（双向区域右边界）  
- `q < p` 的 query：可以看到整个 `[0, p)`（双向）  
- `q ≥ p` 的 query：只看因果前缀（thought / `</think>` / 后续 readout）  
- 线性注意力层、KV cache **写入路径**：完全不动  

### `p` 怎么定（`PromptEnd`）

| 规则 | p |
|---|---|
| prompt 以空 think 块结尾 `<think>\n\n</think>\n\n`（one-pass readout） | `L`（整段双向） |
| 否则：第一个 `<\|im_start\|>assistant\n` 后的 `<think>\n` 之后 | 该位置；其后 forced prefix / thought / readout 因果 |
| `SamplingParams.extra_args["startlux_prompt_end"]=p`（或 OpenAI `vllm_xargs`） | 显式 p（0=全因果，L=全双向） |
| 都不匹配 | 报错；或 `STARTLUX_PREFIX_BIDIR_UNMATCHED=causal\|full` |

与 transformers 参考规则在测试集上对齐过（见 `docs/vllm_bidirectional.md`）。

---

## 4. 实现机制（三步）

### 4.1 挂 import hook，不改磁盘上的 vLLM

`install()` 注册 meta path hook：在 vLLM 导入相关模块时 patch 类方法。已验证版本：**0.19.1**、**0.31.0**；其它版本默认拒绝（可 `STARTLUX_PREFIX_BIDIR_ALLOW_VERSION=1` 强开，需重测）。

### 4.2 Runner 侧：为「本步要 prefill 的 prompt」做 Plan

| Runner | 挂点 |
|---|---|
| V1（0.19.1；0.31 可 `VLLM_USE_V2_MODEL_RUNNER=0`） | `GPUModelRunner._build_attention_metadata` |
| V2（0.31 默认） | model state 的 `prepare_attn`；新请求的 prompt ids 来自 `add_requests` |

Plan 内容：batch 里哪些行属于各 request 的 `[0, p)`，以及 varlen 的 `cu_seqlens`。

硬约束（违反直接 raise，避免静默跑成「另一个模型」）：

- **必须关** chunked prefill、prefix caching  
- `max_num_batched_tokens >= max_model_len`（整段 prompt **一步** prefill 完，从 position 0）  
- 注意力后端 **FLASH_ATTN**（TRITON 基本拒绝）  
- 禁 speculative decoding / context parallel / ubatching / FP8 KV 等  
- prefill 步不能走 FULL CUDA graph（fix-up 要在 eager/piecewise 里跑）

每步还有一条人话版硬检查（`_plan`）：对本步还要读 prompt 的请求，必须 **`computed == 0` 且 `scheduled >= p`**，否则报 `prompt would be split across steps`。  
这三个词（`computed` / `scheduled` / `p`）以及「为啥两边都要关」的图解，见 [03 · 为何关掉 chunked prefill / prefix caching](./03-为何关掉chunked-prefill与prefix-caching.md)。

0.31 的 kernel warm-up 用合成 prompt ids，匹配不到规则 → 这些步 **不做 plan**（因果跑完丢弃），避免 warm-up 炸插件。

### 4.3 Attention 侧：先因果，再覆盖 prompt 行

`FlashAttentionImpl.forward` 被包一层：

1. 先跑 **stock 因果** FlashAttention（照常写进 **paged KV cache**）  
2. 若 metadata 上挂了 `startlux_plan`：对每个 request 的 prompt 行 `[start, start+p)`，用 **非因果 varlen FlashAttention**，K/V 就是 stock 刚写入 cache 的那份，结果 **覆盖** 这些行的 output  
3. `q ≥ p` 的行保留因果结果  
4. 之后 decode：**不动**——prompt 的 K/V 已是双向算出来的；后续 query 都晚于所有 prompt key，mask 自然正确；decode 仍可走 CUDA graph  

额外开销：官方测大约 **+2.2%** wall time（多算一次会被扔掉的因果 prompt 注意力）。

长 prompt（一步 ≥16384 prompt rows）走 lean 路径：按 request 做 in-place view，避免 gather 大拷贝把显存打爆。

---

## 5. 和 paged attention / chunked prefill 的关系

| 能力 | Instant + 插件 | AutoThink / v1 因果文件夹（无插件语义） |
|---|---|---|
| vLLM paged KV | **有**（stock） | **有** |
| chunked prefill | **关**（插件要求整段一步） | 0.31 上 **开**（`--max-num-batched-tokens 8192`） |
| prefix caching | **关** | 0.31 上与 chunked prefill **一起开** |
| 插件 fix-up | **有** | 不需要（causal） |

常见误解纠正：

- 「写了插件 = 自己实现了推理 / 自己实现了 paged attention」→ **否**，只改 prompt 区 full-attn mask  
- 「Instant 关了 chunked prefill = 没有 paged attention」→ **否**，分页 KV 仍在  
- 「#122 里 SharedKV / `prefill_chunk` 就是 vLLM 插件」→ **否**，那是 **transformers 自研路径**（见第一篇）

---

## 6. 客户端怎么用这条栈

1. `startlux-decide launch-vllm`：按版本打 flag、设 `STARTLUX_PREFIX_BIDIR=1`、确认插件 ON  
2. Decision server（`serve`）通过 OpenAI-compatible HTTP 调 vLLM：  
   - **readout**：要 processed logprobs + letter `allowed_token_ids`  
   - **thought**（AutoThink）：0.31+ 默认 stream；老版本 chunked；0.19.1 还要 wave 调度防 hybrid 状态腐蚀  
3. Instant 长上下文：一步装下整段 → 0.31 上 launcher 常把 Instant 的 `max_model_len` 收到 **131072**（更长一步 prefill 在 H200 上会 OOM / illegal access）

细节与数字见 `docs/engines.md`、`docs/vllm_bidirectional.md`。

---

## 7. 校验与安全网

| 手段 | 作用 |
|---|---|
| 启动 log `prefix-bidirectional prompt attention ON` | launcher 必查 |
| `vllm_plugin.require_active(llm, DIR)` | 进程内确认每个 worker 已挂上 |
| `STARTLUX_PREFIX_BIDIR_CHECK=1` | 每层用显式 mask 的 fp32 参考核对 fix-up；偏差超过 stock 内核误差太多则 raise |
| 未知 runner 类 | Instant 直接拒绝启动（避免「以为开了其实在跑因果」） |

静默失败模式要避免：`STARTLUX_PREFIX_BIDIR=1` 但包没进 vLLM 环境 → 实际是 stock 因果，**不会报错**。所以必须看 ON 行或 `require_active`。

---

## 8. 和 transformers 自研路径对照

| | transformers 自研（v1 `model.py`） | vLLM + 本插件 |
|---|---|---|
| 调度 | 自己的 Python 循环 | vLLM Scheduler |
| KV | contiguous `SharedKV` / `DynamicCache` | **paged** KV |
| 长 prompt 切块 | 自己的 `prefill_chunk` | Instant：**不许**切；因果模型：scheduler chunked prefill |
| Instant 双向 | serve 参考引擎 `bidirectional.py`；v1 主路径不扛这个 | 本插件 fix-up |
| 用途 | 本地 Decision / 研究 / 图优化 | **默认线上 serve** |

---

## 9. 读代码顺序（建议）

1. `vllm_plugin.py` 模块 docstring（mask + 流程已写在文件头）  
2. `PromptEnd` → `_plan` / `_make_plan` / `_make_plan_v2`  
3. `_patch_flash_attn` → `_fixup`  
4. `_check_runner_config`（引擎约束清单）  
5. 对照 `docs/vllm_bidirectional.md` 的测试表理解「开插件 vs stock」的 Δp
