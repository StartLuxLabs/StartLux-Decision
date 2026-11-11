# StartLux-Decision · transformers 自研推理路径解读

> 范围：本仓库里**不走 vLLM** 的 PyTorch / transformers 路径。  
> 主代码：`v1/startlux_decision/model.py`（Decision 本地运行时）  
> 对照参考：`startlux_decide/backends/transformers.py` + `backends/bidirectional.py`（serve 包里的 transformers 引擎）  
> 和 vLLM 插件无关；paged attention **不在这条路径上**。

---

## 1. 这条路径是干什么的

Decision 不做自回归「写答案」：每道题渲染成一段 prompt，**一次 forward**，在最后一个真 token 上读选项字母 A/B/C… 的 next-token logits，再按题型温度做 softmax。生成（thought）不是这条 v1 本地路径的主业。

`v1/startlux_decision/model.py` 是一套**自研调度 + 自研 cache + 自研 attention 拼装**，底座仍是 Hugging Face transformers 的 hybrid 模型类（full_attention + linear_attention / Gated DeltaNet），但：

- 不启动 vLLM server
- 不用 vLLM 的 page table / block manager
- KV 存在 transformers 的 `DynamicCache` 变体上（连续张量）
- 长 prompt 的「别一次打爆激活」由本文件自己的 `prefill_chunk` 循环负责

一句话：**原生 transformers forward + 自写 SharedKV / CUDA graph / packing**，不是「改了 vLLM 内核」。

---

## 2. 入口与模块地图

| 组件 | 位置 | 职责 |
|---|---|---|
| `StartLuxDecision` | `v1/startlux_decision/model.py` | 加载权重、渲染、批处理、读字母 logits |
| `SharedKV` | 同文件 `_shared_kv_layer()` | 前缀 K/V 缓冲；冻住后只读不写 |
| `_causal_attention` | 同文件 | 右填充因果注意力；支持「历史块 / 共享前缀」+「当前块」两段 LSE merge |
| CUDA graph | `_capture` / `_graph_logits` | 短题（≤4096）按 (行数, 长度) 回放，砍掉逐层 launch |
| `_shared_groups` / `_shared_logits` | 同文件 | 公共前缀 ≥4K 或单题超长 → 分块 prefill + 共享 KV |
| prompt 渲染 | `jevfmt.py` | TypeSafe / SystemOne 题面 → token ids |
| serve 包参考引擎 | `startlux_decide/backends/transformers.py` | 进程内对照；Instant 走 `bidirectional.py` |

常量（`model.py` 顶层）：

- `SHARE_MIN = 4096`：前缀至少这么长才值得走共享路径  
- `SHARE_STEP = 1024`：共享长度对齐到 1024 的倍数  
- `PREFILL_CHUNK = 32768`：分块 prefill 默认块长（压激活峰值用）  
- `GRAPH_LENGTHS` / `GRAPH_ROWS`：CUDA graph 形状表  

---

## 3. 请求怎么跑完（主流程）

```text
decide(state, questions)
  → 每题 jevfmt 渲染成 token 行
  → _logits(rows)
       ├─ 若短且命中 graph 形状 → _graph_logits（整请求一次 replay）
       ├─ 否则 _shared_groups
       │     └─ 命中组 → _shared_logits（分块写 SharedKV，再跑各题尾巴）
       └─ 其余 → 右填充 eager batch 一次 forward，取末位 hidden @ letter_rows
  → 温度 softmax → 选项概率 / confidence
```

没有 scheduler，没有 page table：batch 形状、是否切块、是否共享前缀，全是本文件的 Python 逻辑。

---

## 4. SharedKV + chunked prefill（核心机制）

### 4.1 为什么要有它

一次 transformer forward 里显存大致两块：

| | 随什么涨 | 跨步能否留下 |
|---|---|---|
| **激活**（当前块 hidden / attn / MLP 中间量） | **本步序列长度** | 算完就扔 |
| **KV（+ 线性注意力 conv/recurrent 状态）** | **已处理总 token** | 必须留下 |

「压激活峰值」压的是第一行。128K 一次喂进去，激活尖峰按 128K 计；切成 4×32K，峰值钉在一块 chunk 上，KV 才跟着总长度涨到 128K。

文件头注释写明：chunks bound the memory for activations；单题 128K 也走同一套路，显存大致只要「整段 prompt 的 K/V」，不必再叠一整段 128K 激活。`PREFILL_CHUNK=32768` 旁注：激活压在几 GiB，35B-A3B 才能在一张 80GB 卡上读 128K。

### 4.2 何时进共享路径

`_shared_groups` 逻辑（人话版）：

1. 按前 `SHARE_MIN` 个 token 分桶  
2. 桶内公共前缀长度 `c` 对齐到 `SHARE_STEP`  
3. 进入共享当且仅当：`c >= 4096` **且**（`len(group) > 1` **或** `len(first) > prefill_chunk`）

所以 **`len(group)==1` 也会进**：条件不是「有人来蹭前缀」，而是「这条序列自己已经长过一块 chunk，必须分块写 KV」。多题共享和单题超长分块，共用同一套 `_shared_logits`。

### 4.3 `_shared_logits` 在干什么

```text
ids = [prefix[:c]]                    # batch=1，只跑公共前缀
for start in range(0, c, prefill_chunk):
    _forward(ids[:, start:start+chunk], past_key_values=shared, use_cache=True)
SharedKV.frozen = True
把每层 keys/values 挂到线程本地 _SHARED.kv
再对每道题只跑 t[c:]，attention 读那份共享 KV + 自己尾巴的因果 KV
```

`SharedKV.update`：未冻结时把本块 K/V 写入预分配 buffer；冻结后直接把当前 query 侧的 key/value 原样返回（真正的历史 K/V 由 `_causal_attention` 从 `_SHARED.kv` 读）。

线性注意力层：共享状态算完后，`_rows_cache` 把 conv/recurrent 状态 **按行数 repeat** 给每道题，各题从同一前缀状态分叉。

### 4.4 注意力怎么拼（`_causal_attention`）

右填充、行与行不互看；因果是唯一 mask（不另建大 mask 张量）。

- **第一块**（q 长度 == 当前 k 长度）：普通因果 SDPA  
- **后续块**：对「块前历史 KV」做非因果 attend，对本块做因果 attend，用 log-sum-exp `_merge` 合成——数学上等于对拼接后的 K/V 做正确因果注意力，但激活只摊开当前块  
- **题目读共享前缀**：把多行 query 折成一行去读**一份**共享 K/V（非因果），再与各行自己尾巴的因果部分 `_merge`

这就是「单题分块」和「多题共享」共用一套算子的原因。

---

## 5. CUDA graph 与短请求路径

短请求（无图、长度落在 `GRAPH_LENGTHS`，多题时还受 `MULTI_ROW_MAX_LENGTH=1024` 约束）走录好的 CUDA graph：

- 启动时 `_capture` 按 (batch 行数, pad 长度) 录图，共用一个 memory pool  
- 推理时 `static_ids` / `last` copy_ 后 `graph.replay()`，去掉逐层 kernel launch  
- 注释写明思路来自 JevK5 runtime（Apache-2.0）

更长或形状未录的，退回 eager：右填充到 `PAD_LENGTHS` 之一，整批一次 `_forward`。

---

## 6. 和 serve 包里 transformers 引擎的关系

`startlux_decide/backends/transformers.py` 是**另一条、更薄的参考引擎**（`--backend transformers`）：

- 一次只跑一条 prompt，给核对用，不扛 serving 负载  
- Instant（`jev_prompt_attention=bidirectional`）在 `bidirectional.py` 里：full_attention 层对整段 one-pass prompt **双向**；linear_attention 仍因果  
- `enable()` 会自检：stock 因果 forward 不变、padding 不影响有效 token、末位 token 能影响第 0 位  
- **没有** SharedKV / CUDA graph / packing；长上下文能力远弱于 `v1/.../model.py`

读「自研 Decision 运行时」看 v1；读「Instant 双向在 transformers 里长什么样」看 `bidirectional.py`。

---

## 7. 和 vLLM / paged attention 的边界（本篇结论）

| 问题 | 答案 |
|---|---|
| 这条路径用不用 vLLM？ | **不用** |
| 有没有 paged attention？ | **没有**（contiguous `DynamicCache` / `SharedKV`，无 block table） |
| 谁负责长 prompt 不炸激活？ | 本文件的 `prefill_chunk` 循环 |
| Instant 双向在这里怎么做？ | v1 主路径是因果 Decision；Instant 双向在 serve 包 `bidirectional.py` / vLLM 插件（见第二篇） |

**一句话**：transformers 自研路径 = 自己当调度器 + 自己管 KV；省的是激活峰值和多题前缀算力，不是 vLLM 的分页 KV。

---

## 8. 读代码顺序（建议）

1. `model.py` 模块 docstring（整体意图）  
2. `_logits` → `_shared_groups` → `_shared_logits`  
3. `SharedKV` + `_causal_attention` + `_merge`  
4. `_capture` / `_graph_logits`（短路径）  
5. 需要 Instant 双向语义时再读 `backends/bidirectional.py`
