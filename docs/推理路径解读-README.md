# StartLux-Decision 推理路径解读

1. [01 · transformers 自研推理路径](./01-transformers自研推理路径解读.md)
2. [02 · vLLM 插件](./02-vllm插件代码解读.md)
3. [03 · 为何关掉 chunked prefill / prefix caching](./03-为何关掉chunked-prefill与prefix-caching.md)（`computed` / `scheduled` / `p` 图解）

不要把 SharedKV/`prefill_chunk` 和 `startlux_prefix_bidir` 混成一条栈。
