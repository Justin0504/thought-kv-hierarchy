# 下一步实验规划

## 当前进展
- ✅ Motivation 实验完成（naive streaming eviction，50 samples）
- ✅ 重要性分布分析完成（长尾分布确认，top-20% 覆盖 56.5%）
- ✅ 相关工作调研完成（20 篇论文）

## 核心问题：如何与竞争者区分

现有竞争者都在 **HBM 内** 做压缩/驱逐：
- ThinKV: 按 thought 类型分段压缩（<5% cache，近无损）
- R-KV: redundancy-aware 选择（10% cache，100% 准确率）

**我们的差异点：memory hierarchy（HBM → DDR → 压缩 → 驱逐），不是删掉，而是搬到慢存储按需取回。**

---

## 实验路线图

### 实验 2：Memory Hierarchy 模拟（Justin - 核心贡献）

**目标**：证明 offload + recall 比 pure eviction 好得多

**方案**：
1. 生成时按重要性把 token 分成 4 个 tier：
   - T0 (HBM): prompt + attention sinks + recent window + top-K% 重要 token
   - T1 (DDR 模拟): 中等重要性 token，用原始精度存储，attend 时加延迟
   - T2 (压缩): 低重要性 token，量化到 4-bit/2-bit 后存储
   - T3 (驱逐): 最不重要的 token，直接丢弃
2. 关键区别：T1/T2 的 token 仍然参与 attention 计算（不像 eviction 直接丢掉）
3. 对比：
   - Full (100% HBM) — baseline
   - Pure eviction (只保留 top-K%) — 当前结果
   - **Hierarchy (T0+T1+T2, 只驱逐 T3)** — 我们的方法
   - Hierarchy + quantization (T2 用 4-bit)

**实现要点**：
```python
# 伪代码
for each decoding step:
    compute attention scores using ALL tiers (T0 full precision, T2 quantized)
    only T3 tokens 不参与计算
    update importance scores
    periodically re-classify tokens into tiers
```

**预期结果**：即使只有 30% token 在 HBM (T0)，accuracy 也能保持 >60%，因为 T1/T2 仍可 attend。

**指标**：
- Accuracy vs HBM usage（不是 vs total cache）
- 对比 pure eviction 在相同 HBM budget 下的表现

### 实验 3：改进重要性打分（Justin）

**动机**：VATP (EMNLP 2024) 指出纯 attention score 不够

**方案**：对比 3 种 importance scoring：
1. Attention-only（当前方法）
2. Value-norm-weighted：importance = attention_score × ||v||
3. Hybrid：attention + value_norm + recency decay

在 50 samples GSM8K 上对比三种打分的 eviction 效果。

### 实验 4：更多 Benchmark（Justin）

**动机**：NeurIPS 要求多 benchmark 验证

**方案**：除 GSM8K 外，加：
- MATH-500（更难的数学推理）
- AIME 2024/2025（竞赛级数学，reasoning chain 更长）
- BBH（BIG-Bench Hard，多样化推理）

先在 GSM8K 确定最优配置，再在其他 benchmark 验证。

### 实验 5：更大模型（Justin + B）

**方案**：
- DeepSeek-R1-Distill-Qwen-14B（如果显存够）
- 验证方法在不同模型规模上的泛化性

---

## B 的实验（系统/硬件侧）

### B-实验 1：实际 offloading 延迟测量
- 测量 GPU HBM → CPU DDR 的 PCIe 传输延迟
- 测量不同 KV block 大小的传输效率
- 建立延迟模型：latency = f(block_size, bus_utilization)

### B-实验 2：Gradient-based importance scoring
- 对比 attention-based vs gradient-based 的重要性排序
- 计算两种方法的 Spearman/Kendall 相关系数
- 看是否 gradient-based 能更准确识别关键 token

### B-实验 3：端到端吞吐量测量
- 在实际 hierarchy 下测量 tokens/sec
- 对比 full attention vs eviction vs hierarchy 的吞吐量
- 分析 PCIe 带宽是否成为瓶颈

---

## 优先级排序

| 优先级 | 实验 | 负责人 | 预计时间 |
|--------|------|--------|----------|
| P0 | 实验 2: Memory Hierarchy 模拟 | Justin | 1 周 |
| P0 | B-实验 1: Offloading 延迟测量 | B | 1 周 |
| P1 | 实验 3: 改进重要性打分 | Justin | 3-4 天 |
| P1 | B-实验 2: Gradient importance | B | 1 周 |
| P2 | 实验 4: 多 Benchmark | Justin | 3-4 天 |
| P2 | B-实验 3: 端到端吞吐量 | B | 1 周 |
| P3 | 实验 5: 14B 模型 | 一起 | 2-3 天 |

## 论文 main result 应该展示什么

Table/Figure: 在相同 HBM budget 下，对比：
1. Full attention (100% HBM) — upper bound
2. StreamingLLM (sink + window) — baseline
3. H2O (heavy hitter eviction) — baseline  
4. Pure eviction (our importance scoring) — ablation
5. **Our hierarchy (offload + recall)** — main result

在 GSM8K, MATH-500, AIME 三个 benchmark 上展示。
关键信息：**30-50% HBM budget 下，hierarchy 保持 >90% baseline accuracy，而 eviction 完全崩溃。**
