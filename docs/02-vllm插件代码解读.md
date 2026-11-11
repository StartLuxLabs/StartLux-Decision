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

源码落点（`_req_prompt_end`）：先看请求上缓存的 `_startlux_prompt_end`，再看 `extra_args`，否则用 `PromptEnd` 扫 token ids；结果写回 `state._startlux_prompt_end`，同请求后续层不再重算。

```249:285:StartLux-Decision/startlux_decide/vllm_plugin.py
def _req_prompt_end(st: _RunnerState, state) -> int:
    p = getattr(state, "_startlux_prompt_end", None)
    if p is not None:
        return p
    ids = state.prompt_token_ids
    ...
    ex = extra.get(EXTRA_KEY, extra.get(LEGACY_EXTRA_KEY))
    if ex is not None:
        p, kind = int(ex), "explicit"
        ...
    else:
        p, kind = st.prompt_end()(ids)
        if p is None:
            policy = env("_UNMATCHED", "error")
            ...
    state._startlux_prompt_end = p
    ...
    return p
```

---

## 4. 实现机制（三步总览）

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

### 4.3 Attention 侧：先因果，再覆盖 prompt 行（一句话）

`FlashAttentionImpl.forward` 被包一层：

1. 先跑 **stock 因果** FlashAttention（照常写进 **paged KV cache**）  
2. 若 metadata 上挂了 `startlux_plan`：对每个 request 的 prompt 行 `[start, start+p)`，用 **非因果 varlen FlashAttention**，K/V 就是 stock 刚写入 cache 的那份，结果 **覆盖** 这些行的 output  
3. `q ≥ p` 的行保留因果结果  
4. 之后 decode：**不动**——prompt 的 K/V 已是双向算出来的；后续 query 都晚于所有 prompt key，mask 自然正确；decode 仍可走 CUDA graph  

额外开销：官方测大约 **+2.2%** wall time（多算一次会被扔掉的因果 prompt 注意力）。

长 prompt（一步 ≥16384 prompt rows）走 lean 路径：按 request 做 in-place view，避免 gather 大拷贝把显存打爆。

**下面 §5 把这条链对着源码逐步拆开。**

---

## 5. fix-up 对着代码逐步走（重点）

人话先钉死：

> **fix-up ≠ 改调度 / 改 KV 布局**。  
> 它是：每个 full-attention 层里，**先照常因果算一遍**，再把本步属于 `[0, p)` 的那些 **attention 输出行**，用一次 **非因果** FlashAttention **原地盖掉**。

调用链（源码顺序）：

```text
_req_prompt_end  →  定 p
      ↓
_plan / _make_plan(_v2)  →  本步哪些行要盖（Plan）
      ↓
_attach  →  plan 挂到 FlashAttentionMetadata.startlux_plan
      ↓
_patch_flash_attn · forward  →  stock 因果 → _fixup
      ↓
_fixup  →  flash_attn_varlen_func(causal=False) 盖 output[0:p)
```

### 5.1 `Plan` 里装的是什么

```288:326:StartLux-Decision/startlux_decide/vllm_plugin.py
class Plan:
    """The prefix fix-up of one step: rows of the flattened batch to recompute, per-request prompt lengths."""
    __slots__ = ("idx", "cu", "maxp", "reqs", "ntok", "rows", "single")
    ...

def _plan(st: _RunnerState, items, ntok: int) -> Plan | None:
    ...
    for i, state, c, s, a0, npt in items:
        p = _req_prompt_end(st, state)
        if p <= 1 or c >= p:          # 已等价因果，或 prompt 早算完
            continue
        if c != 0 or s < p:
            raise RuntimeError(... "prompt would be split across steps" ...)
        rows.append(np.arange(a0, a0 + p, dtype=np.int64))
        cu.append(cu[-1] + p)
        reqs.append((i, a0, s, p))
    ...
    plan.idx = torch.from_numpy(np.concatenate(rows))   # 所有要盖的 batch 行下标
    plan.cu = torch.tensor(cu, dtype=torch.int32)       # varlen 的 cu_seqlens
    plan.maxp = max(r[3] for r in reqs)
    plan.reqs = reqs                                    # (batch_i, a0, scheduled, p)
    plan.rows = cu[-1]                                  # 本步一共要盖几行
```

逐步含义：

| 字段 | 形状 / 含义 |
|---|---|
| `plan.idx` | 一维 long：本步 flattened batch 里所有要重算的行号（各请求 `[a0, a0+p)` 拼起来） |
| `plan.cu` | `int32` 前缀和：第 j 个请求占 `cu[j] … cu[j+1)` 这段 gathered 行 |
| `plan.reqs` | 每请求 `(batch下标, a0, scheduled, p)`；`a0` 是该请求在本步 query 张量上的起始行 |
| `plan.rows` / `plan.maxp` | 总行数、最长那段 p（给 varlen kernel 的 `max_seqlen`） |

`rows` / `plan.idx` 就是：**本步 flattened query 张量里，所有要拿双向注意力重算（盖掉 stock 因果结果）的行下标**，按请求顺序拼成一条一维 long。对每个「本步要盖」的请求：`a0` 是该请求在本步 query 上的起始行，`p` 是 prompt 双向区长度，`np.arange(a0, a0 + p)` 就是该请求要盖的行号；`rows` 是这些区间的 list，`np.concatenate(rows)` → `plan.idx`。后面 `_fixup` 用 `query.index_select(0, plan.idx)` 把这些行捞出来做 `causal=False`，再 `index_copy_` 盖回去。

#### 具体例子（单测同款）

本步 3 个请求，`sched=[1,7,6]`，`computed=[9,0,0]`（与 `tests/test_vllm_plugin.py` 的 `PlanTest` 一致）：

| 请求 | 角色 | scheduled | a0（起始行） | p | 是否进 plan |
|---|---|---|---|---|---|
| r0 | decode | 1 | 0 | 8 | 否（`c=9 ≥ p`，prompt 早已算完） |
| r1 | prefill | 7 | 1 | 5 | 是 → `[1,2,3,4,5]` |
| r2 | prefill | 6 | 8 | 6 | 是 → `[8,9,10,11,12,13]` |

本步 flattened 行布局（共 `ntok=14`）：

```text
row:  0 | 1 2 3 4 5 6 7 | 8 9 10 11 12 13
      r0|     r1        |      r2
```

拼出来：

```text
rows = [
  arange(1, 1+5) = [1,2,3,4,5],        # r1 只盖前 p=5 行；行 6、7 是 scheduled 多出来的，仍留因果
  arange(8, 8+6) = [8,9,10,11,12,13],  # r2 整段 p=6 都盖
]
plan.idx = [1,2,3,4,5, 8,9,10,11,12,13]
plan.cu  = [0, 5, 11]   # gathered 里：r1 占 0:5，r2 占 5:11
plan.reqs = [(1, 1, 7, 5), (2, 8, 6, 6)]  # (batch_i, a0, scheduled, p)
plan.rows = 11; plan.maxp = 6; plan.ntok = 14
```

注意：r1 的 `s=7 > p=5`，所以 `plan.idx` **不含** 行 6、7——那两行不在双向 prompt 区，stock 因果结果保留。

硬检查 `c != 0 or s < p`：**整段 `[0,p)` 必须一步、从 0 开始全部出现在本步**。半截时前面那些 query 看不到后半 key，下一截也盖不到上一截的 output——详见 [03](./03-为何关掉chunked-prefill与prefix-caching.md)。

### 5.2 Runner 怎么把 Plan 挂上 metadata

**V1**（patch `_build_attention_metadata`）：

```446:459:StartLux-Decision/startlux_decide/vllm_plugin.py
def _build_attention_metadata(self, *args, **kwargs):
    out = build0(self, *args, **kwargs)
    ...
    if a["for_cudagraph_capture"] or a["num_scheduled_tokens"] is None or a["max_query_len"] <= 1:
        return out             # capture / 纯 decode 步：不做 plan
    plan = _make_plan(self, a["num_reqs"], a["num_scheduled_tokens"])
    if plan is not None:
        _attach(out[0], plan)
    return out
```

**V2**（wrap `prepare_attn`）：

```476:487:StartLux-Decision/startlux_decide/vllm_plugin.py
def prepare_attn(input_batch, cudagraph_mode, *args, **kwargs):
    out = prep0(input_batch, cudagraph_mode, *args, **kwargs)
    if kwargs.get("for_capture") or not input_batch.has_prefill:
        return out
    plan = _make_plan_v2(runner, input_batch)
    if plan is not None:
        if getattr(cudagraph_mode, "name", cudagraph_mode) == "FULL":
            raise RuntimeError(... "fix-up runs only in eager attention ...")
        _attach(out, plan)
    return out
```

`_attach` 对 `FlashAttentionMetadata` 做一件事：`m.startlux_plan = plan`（顺带拒绝 cascade / PrefixLM ranges / 非 FLASH_ATTN）。

### 5.3 挂钩点：`_patch_flash_attn` —— 先 stock，再 `_fixup`

这是「具体怎么 fix up」的入口：

```673:697:StartLux-Decision/startlux_decide/vllm_plugin.py
def _patch_flash_attn(mod) -> None:
    Impl = mod.FlashAttentionImpl
    ...
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None, ...):
        out = fwd0(self, layer, query, key, value, kv_cache, attn_metadata, output=output, ...)
        plan = getattr(attn_metadata, "startlux_plan", None) if attn_metadata is not None else None
        if plan is not None and self.attn_type == AttentionType.DECODER:
            if self.dcp_world_size > 1:
                raise RuntimeError(... "decode context parallelism is not supported")
            chk = env("_CHECK") == "1"
            stock_rows = output.index_select(0, plan.idx).clone() if chk else None
            _fixup(self, plan, query, key, value, output)
            if chk:
                _check(self, plan, query, key, value, output, stock_rows)
        return out

    Impl.forward = forward
```

对着读：

1. **`fwd0(...)`**：原版因果 FlashAttention。KV **已经**按 stock 路径写进 paged cache；`output` 里先躺着因果结果。  
2. **有 `startlux_plan` 且是 DECODER 全注意力层** 才继续；线性注意力层根本不会进这个 Impl。  
3. **`_fixup(..., output)`**：只改 `output` 里 plan 点名的那些行；**不**改 `kv_cache` 张量布局。  
4. 开了 `STARTLUX_PREFIX_BIDIR_CHECK=1` 时，先把因果 prompt 行 clone 一份，fix-up 后再跟 fp32 显式 mask 参考比（`_check`）。

### 5.4 `_fixup` 本体：两种盖法

```581:608:StartLux-Decision/startlux_decide/vllm_plugin.py
def _fixup(impl, plan: Plan, query, key, value, output) -> None:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
    if key is None or value is None:
        raise RuntimeError(... "KV sharing")
    if impl.alibi_slopes is not None or impl.sinks is not None or impl.kv_cache_dtype.startswith("fp8"):
        raise RuntimeError(... "ALiBi / sinks / fp8 KV are not supported")
    kw = dict(softmax_scale=impl.scale, causal=False, window_size=list(impl.sliding_window),
              softcap=impl.logits_soft_cap, fa_version=impl.vllm_flash_attn_version,
              num_splits=1 if impl.batch_invariant_enabled else 0)
    if plan.rows >= int(env("_LEAN_ROWS", str(LEAN_ROWS))) and output.dtype == query.dtype:
        # —— lean：长步，按请求 view，原地写 ——
        if plan.single is None:
            plan.single = [torch.tensor([0, p], dtype=torch.int32, device=query.device)
                           for _, _, _, p in plan.reqs]
        for (_, a0, _, p), cu in zip(plan.reqs, plan.single):
            flash_attn_varlen_func(
                q=query[a0:a0 + p], k=key[a0:a0 + p], v=value[a0:a0 + p],
                max_seqlen_q=p, cu_seqlens_q=cu, max_seqlen_k=p, cu_seqlens_k=cu,
                out=output[a0:a0 + p], **kw)
        ...
        return
    # —— 默认：gather 成一批，再 index_copy_ 回去 ——
    q = query.index_select(0, plan.idx)
    k = key.index_select(0, plan.idx)
    v = value.index_select(0, plan.idx)
    o = flash_attn_varlen_func(
        q=q, k=k, v=v,
        max_seqlen_q=plan.maxp, cu_seqlens_q=plan.cu,
        max_seqlen_k=plan.maxp, cu_seqlens_k=plan.cu, **kw)
    output.index_copy_(0, plan.idx, o.to(output.dtype))
```

#### A. 共同前提

| 点 | 代码在干什么 |
|---|---|
| `causal=False` | 这一次 **不是** 因果；对单个请求的 `[0,p)`，每个 q 都能看该请求自己的全部 prompt k（再叠加滑动窗口等 `kw`） |
| `key` / `value` | 就是 stock forward 刚算出来、喂进本层的那份（与写入 paged KV 的内容一致）；**不是**另开一套 cache |
| 只动 `output` | `index_copy_` / `out=output[a0:a0+p]` 盖的是 **attention 输出行**；Gated DeltaNet / 短卷积 / KV 写入路径都不进这里 |

#### B. 默认路径（短步 / 多短请求）

示意：本步 batch 里两条 Instant one-pass（`p=L`），中间夹着别的行：

```text
flattened query/key/value/output 行：
  [reqA 的 0..pA) | 其它 | reqB 的 0..pB) | ...]
         ↑ plan.idx 只收集这些 ↑

index_select → 拼成连续 q/k/v
flash_attn_varlen_func(..., cu_seqlens=plan.cu, causal=False)
index_copy_ → 写回 output 的原行号
```

`plan.cu` 保证 **请求之间不串注意力**：varlen 把 A、B 当成两条独立序列，不会出现 A 的 q 去看 B 的 k。

#### C. lean 路径（`plan.rows ≥ 16384`）

常量：`LEAN_ROWS = 16384`（可用 `STARTLUX_PREFIX_BIDIR_LEAN_ROWS` 改）。

动机写在模块头：**vLLM 的 profile 跑没有 plan，不会给 fix-up 预留 gather 显存**。对 4B、26 万 token 那种一步，再 `index_select` 整份 QKV/O 能多吃约 5GB/层，直接 OOM。

所以改成：对每个 `(a0, p)` 只切 **view** `query[a0:a0+p]` …，并把 kernel 的 `out=` 指到 `output[a0:a0+p]`——**零拷贝覆盖**。数值与 gather 路径一致（测试钉过）。

#### D. 盖完以后谁还动？

| 位置 | 还要不要 fix-up |
|---|---|
| 本步 `q ≥ p`（thought / `</think>` / readout） | **否**，保留 stock 因果 |
| 后续纯 decode 步 | **否**：plan 为空（`max_query_len<=1` / `has_prefill=False`）；prompt K/V 已是「双向 prompt 条件下」写出的；之后所有 query 都晚于所有 prompt key，因果读 = 目标 mask |
| 被 preempt 后再 prefill | **要**：从 position 0 整段重算，同一套 `_plan` + `_fixup` |

### 5.5 一张图收束

```text
本步 full-attn 层：

  query/key/value ──► stock FlashAttention (causal=True)
                            │
                            ├─► 写 paged KV（stock 路径，未改）
                            └─► output ← 先是因果结果
                                      │
                      metadata.startlux_plan ?
                           │ yes
                           ▼
                      _fixup(..., causal=False)
                           │
              ┌────────────┴────────────┐
              │ rows < LEAN_ROWS        │ rows ≥ LEAN_ROWS
              │ gather → varlen → copy_ │ 每请求 view → out=切片
              └────────────┬────────────┘
                           ▼
                 output[0:p) = 双向结果
                 output[p: ) = 仍是因果
```

---

## 6. 和 paged attention / chunked prefill 的关系

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

## 7. 客户端怎么用这条栈

1. `startlux-decide launch-vllm`：按版本打 flag、设 `STARTLUX_PREFIX_BIDIR=1`、确认插件 ON  
2. Decision server（`serve`）通过 OpenAI-compatible HTTP 调 vLLM：  
   - **readout**：要 processed logprobs + letter `allowed_token_ids`  
   - **thought**（AutoThink）：0.31+ 默认 stream；老版本 chunked；0.19.1 还要 wave 调度防 hybrid 状态腐蚀  
3. Instant 长上下文：一步装下整段 → 0.31 上 launcher 常把 Instant 的 `max_model_len` 收到 **131072**（更长一步 prefill 在 H200 上会 OOM / illegal access）

细节与数字见 `docs/engines.md`、`docs/vllm_bidirectional.md`。

---

## 8. 校验与安全网

| 手段 | 作用 |
|---|---|
| 启动 log `prefix-bidirectional prompt attention ON` | launcher 必查 |
| `vllm_plugin.require_active(llm, DIR)` | 进程内确认每个 worker 已挂上 |
| `STARTLUX_PREFIX_BIDIR_CHECK=1` | 每层用显式 mask 的 fp32 参考核对 fix-up；偏差超过 stock 内核误差太多则 raise |
| 未知 runner 类 | Instant 直接拒绝启动（避免「以为开了其实在跑因果」） |

静默失败模式要避免：`STARTLUX_PREFIX_BIDIR=1` 但包没进 vLLM 环境 → 实际是 stock 因果，**不会报错**。所以必须看 ON 行或 `require_active`。

---

## 9. 和 transformers 自研路径对照

| | transformers 自研（v1 `model.py`） | vLLM + 本插件 |
|---|---|---|
| 调度 | 自己的 Python 循环 | vLLM Scheduler |
| KV | contiguous `SharedKV` / `DynamicCache` | **paged** KV |
| 长 prompt 切块 | 自己的 `prefill_chunk` | Instant：**不许**切；因果模型：scheduler chunked prefill |
| Instant 双向 | serve 参考引擎 `bidirectional.py`；v1 主路径不扛这个 | 本插件 fix-up |
| 用途 | 本地 Decision / 研究 / 图优化 | **默认线上 serve** |

---

## 10. 读代码顺序（建议）

1. `vllm_plugin.py` 模块 docstring（mask + 流程已写在文件头）  
2. `_req_prompt_end` / `PromptEnd` → 定 `p`  
3. `_plan` / `_make_plan` / `_make_plan_v2` → `_attach`  
4. `_patch_flash_attn` → **`_fixup`（本文 §5）**  
5. `_check`（可选，对照 fp32 显式 mask）  
6. `_check_runner_config`（引擎约束清单）  
7. 对照 `docs/vllm_bidirectional.md` 的测试表理解「开插件 vs stock」的 Δp  
8. 关 chunked / prefix 的调度原因 → [03](./03-为何关掉chunked-prefill与prefix-caching.md)
