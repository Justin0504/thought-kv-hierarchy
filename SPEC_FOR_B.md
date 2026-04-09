# 系统侧工作规范（给 B）

## 统一配置

- **模型**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`（Qwen2 架构，28 层，28 heads）
- **精度**: `torch.float16`
- **数据集**: GSM8K, 前 50 samples（`datasets` 库加载 `openai/gsm8k`）
- **Prompt 格式**:
  ```
  Please solve this math problem step by step.

  Question: {question}

  Let me think step by step.
  ```
- **生成参数**: greedy decoding, max_new_tokens=2048

## KV Cache 格式

HuggingFace `DynamicCache` 对象：
- `past_kv.key_cache[layer_idx]` → `[batch=1, heads=28, seq_len, head_dim=128]`，float16
- `past_kv.value_cache[layer_idx]` → 同上
- 共 28 层
- 每个 token 的 KV 大小: 28层 × 28heads × 128dim × 2(K+V) × 2bytes = **401KB/token**

## 重要性分数接口

算法侧输出 per-token importance（float64 array, shape=[seq_len]）：
- 值 = 该位置在所有 decoding step 中累计收到的 attention weight
- 值越高越重要
- 文件: `results/week1_profiling_results.json` 中有 top20_coverage 等统计

## 保护区域（不可驱逐/offload）

- Prompt tokens（前 ~80 tokens）
- Attention sinks: prompt 后的前 4 个 reasoning token
- Recent window: 最近 128 个 token

## B 的三个任务

### B-1: PCIe Offloading 延迟测量（P0）
测量 GPU HBM ↔ CPU DDR 的 KV block 传输延迟：
- 测试不同 block size: 1 token, 16 tokens, 64 tokens, 256 tokens
- 每个 token KV = 401KB, 所以 256 tokens ≈ 100MB
- 用 `torch.cuda.Event` 精确计时
- 测 H2D 和 D2H 双向
- 输出: latency vs block_size 曲线

### B-2: Gradient-based Importance Scoring（P1）
实现替代重要性度量——用 gradient norm:
```python
# 伪代码
logits = model(input_ids, use_cache=True)
loss = logits[:, -1, :].max()  # 或 target token 的 logit
loss.backward()
# 对每个 KV position 计算 gradient norm
for layer in range(28):
    grad_k = past_kv.key_cache[layer].grad  # [1, 28, seq, 128]
    importance[pos] = grad_k[:, :, pos, :].norm()
```
然后和 attention-based importance 计算 Spearman 相关系数。

### B-3: 真实 Offloading 系统原型（P1）
用 `torch.Tensor.to('cpu')` / `.to('cuda')` 实现：
- 当 importance < threshold → 把该 position 的 KV 搬到 CPU
- Attention 计算前 → prefetch 回 GPU
- 测量端到端 tokens/sec

## 已有实验结果（供参考）

| 方法 | 配置 | GSM8K 准确率 |
|------|------|-------------|
| Full cache | 100% HBM | 68% |
| Pure eviction | 50% HBM | 0% |
| **Memory hierarchy** | **50% HBM, 10% evict** | **34%** |
| **Memory hierarchy** | **30% HBM, 10% evict** | **34%** |

关键发现: HBM ratio 不影响准确率（30%=50%），evict_ratio 才是关键。
说明 offloaded 到 DDR 的 token 确实被用到了，hierarchy 有效。
