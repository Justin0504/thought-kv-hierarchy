# 系统侧工作规范（给 B）

## 一、项目背景

我们在写一篇论文：**"Not All Thoughts Need HBM: Semantics-Aware Memory Hierarchy for LLM Reasoning"**

**核心假设**：推理模型（如 DeepSeek-R1）生成的思维链中，不同 token 的重要性差别很大（长尾分布）。低重要性的 KV-cache 不需要放在昂贵的 GPU HBM 里，可以搬到更便宜的 CPU DDR 内存，需要时再取回来。

**我已完成的工作（算法侧）**：
- 验证了 token 重要性确实是长尾分布（top-20% token 贡献 56.5% 的总重要性）
- 验证了 attention-based 重要性打分有效（优于 random）
- 验证了 memory hierarchy（offload 到 DDR）远好于直接 eviction（丢弃）
- 在 50% HBM budget 下：直接丢弃 → 0% 准确率，hierarchy → 34% 准确率

**你需要做的（系统侧）**：证明这个 hierarchy 在实际硬件上可行，测量延迟和吞吐量。

---

## 二、环境搭建

### 2.1 Clone 项目

```bash
git clone https://github.com/Justin0504/thought-kv-hierarchy.git
cd thought-kv-hierarchy
```

### 2.2 安装依赖

```bash
# 方法1：conda
conda env create -f environment.yml
conda activate thought-hbm

# 方法2：pip
pip install torch transformers accelerate datasets numpy matplotlib tqdm
```

### 2.3 模型和数据

**不需要手动下载**，第一次运行脚本时会自动从 HuggingFace 下载：
- 模型：`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`（约 14GB，公开模型）
- 数据：`openai/gsm8k`（自动下载）

如果想手动预下载模型：
```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
"
```

### 2.4 验证环境

```bash
# 跑一个最小测试（1 个样本），确认一切正常
python scripts/run_week1_validation.py --n_samples 1 --device cuda:0
```

---

## 三、统一配置（必须和我保持一致）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 模型 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Qwen2 架构，28 层，28 个 attention head |
| 精度 | `torch.float16` | 半精度推理 |
| 数据集 | GSM8K 前 50 题 | `datasets` 库加载 `openai/gsm8k` |
| 生成方式 | greedy decoding, max_new_tokens=2048 | 不用 sampling |
| sink_size | 4 | 始终保留的前 4 个 reasoning token |
| window_size | 128 | 始终保留的最近 128 个 token |

### Prompt 格式（必须完全一致）

```
Please solve this math problem step by step.

Question: {question}

Let me think step by step.
```

---

## 四、KV Cache 技术细节

### 4.1 数据结构

HuggingFace `DynamicCache` 对象：
```python
past_kv = model(input_ids, use_cache=True).past_key_values

# 访问方式
past_kv.key_cache[layer_idx]   # shape: [batch=1, heads=28, seq_len, head_dim=128], dtype=float16
past_kv.value_cache[layer_idx] # shape: 同上

# 共 28 层
len(past_kv.key_cache)  # 28
```

### 4.2 每个 token 的 KV 大小

```
每个 token = 28层 × 28heads × 128dim × 2(K+V) × 2bytes(fp16)
           = 28 × 28 × 128 × 2 × 2
           = 401,408 bytes ≈ 401 KB/token
```

一条完整的推理链（~1500 tokens）的 KV cache ≈ 600 MB。

### 4.3 保护区域（不可驱逐/offload 的 token）

```
[prompt tokens (~80)] [sink (4)] [...reasoning tokens...] [recent window (128)]
 ^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^                           ^^^^^^^^^^^^^^^^^^
      不可动              不可动                                不可动
```

中间的 reasoning tokens 是分层管理的对象。

---

## 五、你的三个任务

### 任务 1：PCIe Offloading 延迟测量（优先级最高）

**目标**：测量 GPU HBM ↔ CPU DDR 之间搬运 KV 数据的实际延迟。

**为什么重要**：我们的 hierarchy 方法把低重要性 token 搬到 CPU DDR，attention 计算时需要搬回 GPU。如果搬运延迟太高，整个方法就不实用。

**具体做什么**：

```python
import torch
import time

def benchmark_transfer(n_tokens, n_layers=28, n_heads=28, head_dim=128, dtype=torch.float16):
    """测量 n_tokens 个 token 的 KV cache 在 GPU ↔ CPU 之间的传输延迟"""

    # 模拟 KV cache 数据
    k = torch.randn(1, n_heads, n_tokens, head_dim, dtype=dtype, device='cuda')
    v = torch.randn(1, n_heads, n_tokens, head_dim, dtype=dtype, device='cuda')

    # GPU → CPU (offload)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    k_cpu = k.to('cpu', non_blocking=True)
    v_cpu = v.to('cpu', non_blocking=True)
    end.record()
    torch.cuda.synchronize()
    gpu_to_cpu_ms = start.elapsed_time(end)

    # CPU → GPU (prefetch)
    start.record()
    k_gpu = k_cpu.to('cuda', non_blocking=True)
    v_gpu = v_cpu.to('cuda', non_blocking=True)
    end.record()
    torch.cuda.synchronize()
    cpu_to_gpu_ms = start.elapsed_time(end)

    data_size_mb = k.nelement() * k.element_size() * 2 / 1024 / 1024  # K + V
    print(f"  {n_tokens} tokens ({data_size_mb:.1f} MB):")
    print(f"    GPU→CPU: {gpu_to_cpu_ms:.2f} ms")
    print(f"    CPU→GPU: {cpu_to_gpu_ms:.2f} ms")

    return gpu_to_cpu_ms, cpu_to_gpu_ms

# 测试不同大小
for n_tok in [1, 16, 64, 128, 256, 512, 1024]:
    # 测 28 层的总延迟
    total_g2c, total_c2g = 0, 0
    for layer in range(28):
        g2c, c2g = benchmark_transfer(n_tok, n_layers=1)
        total_g2c += g2c
        total_c2g += c2g
    print(f"  TOTAL (28 layers): GPU→CPU={total_g2c:.1f}ms, CPU→GPU={total_c2g:.1f}ms")
```

**输出要求**：
- 一张图：latency (ms) vs block_size (tokens)，包含 GPU→CPU 和 CPU→GPU 两条线
- 带宽利用率：实际 throughput (GB/s) vs 理论 PCIe 带宽
- 结论：offload N 个 token 的延迟是多少，能否隐藏在计算中

### 任务 2：Gradient-based Importance Scoring（优先级中）

**目标**：实现一种替代的重要性打分方法，和我的 attention-based 方法对比。

**为什么重要**：审稿人可能会质疑 attention score 是否是最优的重要性度量。gradient-based 方法从理论上更直接（直接衡量每个 KV 位置对输出的影响）。

**具体做什么**：

```python
def compute_gradient_importance(model, input_ids, past_key_values):
    """计算每个 KV 位置的 gradient norm 作为重要性"""
    # 需要开启 gradient
    for layer in range(len(past_key_values.key_cache)):
        past_key_values.key_cache[layer].requires_grad_(True)
        past_key_values.value_cache[layer].requires_grad_(True)

    outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
    # 对最后一个 token 的 top logit 求梯度
    logits = outputs.logits[:, -1, :]
    target_logit = logits.max()
    target_logit.backward()

    # 汇总每个位置的 gradient norm
    importance = torch.zeros(past_key_values.get_seq_length())
    for layer in range(len(past_key_values.key_cache)):
        k_grad = past_key_values.key_cache[layer].grad  # [1, 28, seq, 128]
        v_grad = past_key_values.value_cache[layer].grad
        # 每个位置的 grad norm
        k_norm = k_grad.norm(dim=(0, 1, 3))  # [seq]
        v_norm = v_grad.norm(dim=(0, 1, 3))
        importance += (k_norm + v_norm).detach().cpu()

    return importance.numpy()
```

**输出要求**：
- 对 50 个 GSM8K 样本，分别计算 attention-based 和 gradient-based 的重要性排序
- 计算两者的 Spearman rank correlation（越高说明两种方法越一致）
- 用 gradient-based importance 替换 attention-based 跑 streaming eviction，对比准确率
- 结论：gradient-based 是否比 attention-based 更好

### 任务 3：端到端 Offloading 系统原型（优先级中）

**目标**：实现真实的 KV cache GPU-CPU 搬运，测量端到端吞吐量。

**为什么重要**：我目前的实验是"模拟"hierarchy（offloaded token 仍在 GPU 上），需要证明真实搬运也可行。

**具体做什么**：
1. 生成过程中，当 cache 超过 HBM budget，把低重要性 token 的 KV 搬到 CPU
2. 每次 attention 计算前，把需要的 KV 从 CPU prefetch 回 GPU
3. 测量实际 tokens/sec

**输出要求**：
- tokens/sec: full attention vs hierarchy offloading
- 延迟分解：计算时间 vs 搬运时间的比例
- 结论：搬运开销是否可接受（<20% 额外延迟为理想）

---

## 六、时间线

| 周 | 你的任务 | 我的任务 |
|----|----------|----------|
| 第 1 周 | 任务 1（PCIe 延迟测量） | 完善 evict_ratio、改进 importance scoring |
| 第 2 周 | 任务 2（gradient importance） | 多 benchmark 验证（MATH-500, AIME） |
| 第 3 周 | 任务 3（端到端原型） | 整合结果、写论文 |
| 第 4 周 | 补充实验 + review | 论文定稿 |

---

## 七、文件约定

你的代码放在：
```
src/system/           # 系统侧代码
scripts/benchmark_*   # 性能测量脚本
results/system/       # 系统侧实验结果
```

提交到同一个 repo 的独立分支或直接 push main 都行。

---

## 八、联系方式

有任何问题随时找我。关键是保持模型、数据、prompt 的一致性，实验才能对齐。
