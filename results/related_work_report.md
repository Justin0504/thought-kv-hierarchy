Now I have comprehensive information. Let me compile the structured report.

Here is my thorough research report on KV-cache eviction and memory management for LLM reasoning/inference, organized by relevance to your paper.

---

## PART 1: Most Relevant Competing/Similar Papers

### A. Reasoning-Specific KV Cache Work (HIGHEST relevance -- direct competitors)

**1. ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models**
- Authors: multiple; Venue: arXiv Oct 2025
- Key idea: Decomposes CoT into reasoning, execution, and transition segments based on attention sparsity patterns. Applies a hybrid quantization-eviction strategy that assigns token precision by thought importance, progressively evicting tokens from less critical thoughts.
- Results: Near-lossless accuracy with <5% of original KV cache; 5.8x throughput over SOTA baselines. Tested on DeepSeek-R1-Distill, GPT-OSS, AceReason.
- **Relation to your work: This is your most direct competitor.** ThinKV also recognizes unequal importance of reasoning tokens and applies differentiated treatment. The key difference is ThinKV does quantization+eviction within HBM, while your paper proposes a memory hierarchy (HBM/DDR/compressed). Your semantics-aware scoring and tiered offloading approach is complementary but distinct. You MUST cite and differentiate from this paper carefully.
- [arXiv link](https://arxiv.org/abs/2510.01290)

**2. R-KV: Redundancy-aware KV Cache Compression for Reasoning Models**
- Authors: Cai, Xiao et al.; Venue: NeurIPS 2025
- Key idea: Ranks tokens on-the-fly for both importance AND non-redundancy, retaining only informative, diverse ones. Traditional compression methods tuned for long prompts fail on generated reasoning traces (~60% accuracy at 10% cache).
- Results: Preserves nearly 100% performance at 10% cache; even achieves 105% at 16% cache. 90% memory savings, 6.6x throughput.
- **Relation to your work: Another direct competitor for reasoning-specific KV compression.** R-KV's finding that traditional methods retain only ~60% accuracy at 10% cache on reasoning models corroborates your finding about naive eviction degradation. The key distinction is R-KV focuses on redundancy-aware selection within a single tier, while you propose cross-tier memory management.
- [arXiv link](https://arxiv.org/abs/2505.24133)

**3. Hold Onto That Thought: Assessing KV Cache Compression On Reasoning**
- Authors: multiple; Venue: NeurIPS 2025
- Key idea: Comprehensive benchmark of SOTA KV cache compression across 8 reasoning benchmarks (including GSM8K). Evaluates on DeepSeek-R1-Distill-Qwen-7B/14B. Finds that heavy-hitter attention-based strategies (H2O, SnapKV-Decoding) significantly outperform other methods for reasoning.
- **Relation to your work: Directly validates your experimental setup and findings.** Uses the same model (DeepSeek-R1-Distill-Qwen-7B) and benchmark (GSM8K). Their finding that attention-based heavy-hitter strategies outperform others aligns with your attention-based importance scoring results. However, they don't propose a memory hierarchy -- they just benchmark existing methods. This is a key reference for positioning your contribution.
- [arXiv link](https://arxiv.org/abs/2512.12008)

**4. TriAttention: Efficient Long Reasoning with Trigonometric KV Compression**
- Authors: researchers from MIT, ZJU, NVIDIA; Venue: arXiv April 2026
- Key idea: Observes that pre-RoPE Q/K vectors are concentrated around fixed non-zero centers. Estimates key importance from stable Q/K centers using trigonometric series, avoiding post-RoPE instability.
- Results: Matches full attention on AIME25 (32K-token generation) with 2.5x throughput and 10.7x KV memory reduction.
- **Relation to your work: Very recent concurrent work on reasoning KV compression. Uses a fundamentally different importance estimation mechanism (trigonometric/geometric rather than attention-score-based). Could be complementary -- their importance scoring could potentially be integrated into your memory hierarchy.**
- [arXiv link](https://arxiv.org/abs/2604.04921)

---

### B. Foundational KV Cache Eviction Papers (cite as baselines)

**5. StreamingLLM: Efficient Streaming Language Models with Attention Sinks**
- Authors: Xiao et al. (MIT Han Lab); Venue: ICLR 2024
- Key idea: Discovered "attention sinks" -- initial tokens receive disproportionately high attention regardless of semantic relevance. Maintains a small prefix of initial tokens + sliding window of recent tokens.
- Results: Stable language modeling up to 4M tokens; 22.2x speedup over sliding window recomputation.
- **Relation to your work: StreamingLLM's naive eviction of middle tokens is essentially what your "streaming eviction" baseline does. Your finding that streaming eviction causes sharp accuracy drops (68% to 56% at 90% budget) directly motivates going beyond StreamingLLM's approach. The attention sink phenomenon should be accounted for in your importance scoring.**
- [arXiv link](https://arxiv.org/abs/2309.17453)

**6. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models**
- Authors: Zhang, Sheng et al.; Venue: NeurIPS 2023
- Key idea: Observes power-law distribution in accumulated attention scores -- a small set of "heavy hitter" tokens are critical. Dynamically retains recent + heavy-hitter tokens. Formulated as dynamic submodular optimization with theoretical guarantees.
- Results: Up to 29x throughput improvement with 20% heavy hitters on OPT-6.7B/30B.
- **Relation to your work: H2O's power-law observation directly supports your "long-tail distribution" claim about reasoning token importance. Your work extends H2O's insight to reasoning models specifically and proposes tiered memory instead of binary eviction. H2O is a critical baseline to compare against.**
- [arXiv link](https://arxiv.org/abs/2306.14048)

**7. ScissorHands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time**
- Authors: multiple; Venue: NeurIPS 2024 (OpenReview)
- Key idea: "Persistence of importance" hypothesis -- pivotal tokens that matter at one step continue to matter in future steps. Maintains KV cache at fixed budget without finetuning.
- Results: Up to 5x memory reduction without quality loss.
- **Relation to your work: The persistence hypothesis is relevant -- if important tokens persist, you can make reliable offloading decisions early. However, this may not hold for reasoning models where importance shifts across CoT phases. Worth discussing.**
- [OpenReview link](https://openreview.net/forum?id=JZfg6wGi6g)

**8. SnapKV: LLM Knows What You are Looking for Before Generation**
- Authors: Li et al.; Venue: NeurIPS 2024
- Key idea: Uses an "observation window" at the end of prompts to identify which KV positions each attention head consistently focuses on. Compresses via clustering of important positions.
- Results: 3.6x generation speed, 8.2x memory efficiency at 16K tokens; 92% compression at 1024 cache size.
- **Relation to your work: SnapKV is prompt-compression focused (prefill phase), not generation-phase compression. "Hold Onto That Thought" extends it to SnapKV-Decoding for reasoning. Your approach handles the generation/reasoning phase where tokens are produced sequentially, making SnapKV's observation window less directly applicable.**
- [arXiv link](https://arxiv.org/abs/2404.14469)

**9. FastGen: Adaptive KV Cache Compression for Long-Context LLM Inference**
- Authors: multiple; Venue: arXiv 2023
- Key idea: Identifies five fundamental attention structures per head (local, punctuation-focused, sparse, broad, etc.) and applies per-head optimal eviction strategies.
- **Relation to your work: FastGen's per-head strategy selection is relevant -- different heads may require different memory tier placements. However, FastGen doesn't consider reasoning-specific patterns or memory hierarchies.**
- [arXiv link](https://arxiv.org/abs/2310.01801)

---

### C. KV Cache Offloading / Memory Hierarchy Papers (closest architectural match)

**10. InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management**
- Authors: Lee et al.; Venue: OSDI 2024
- Key idea: Works with offloading-based systems. Speculates which tokens will be important for the next layer by performing minimal "rehearsal" with current-layer inputs and next-layer weights. Prefetches only important KV entries from CPU to GPU.
- Results: Up to 3.0x improvement over prior KV cache management in offloading systems.
- **Relation to your work: InfiniGen is architecturally the closest to your vision -- it implements a memory hierarchy with selective prefetching. The difference is InfiniGen focuses on long-context prefill, not reasoning-token generation, and uses next-layer speculation rather than attention-based importance scoring.**
- [USENIX link](https://www.usenix.org/conference/osdi24/presentation/lee)

**11. Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference**
- Authors: MIT Han Lab; Venue: ICML 2024
- Key idea: Token criticality depends on the current query. Tracks min/max Key values per KV cache page and estimates page criticality using query vectors. Only loads critical pages for attention.
- Results: 7.03x self-attention speedup, 2.23x latency reduction.
- **Relation to your work: Quest's query-aware importance is directly relevant -- importance should be reassessed per-query, not static. Your memory hierarchy could use Quest-style page-level importance estimation for deciding what to keep in HBM vs. DDR.**
- [arXiv link](https://arxiv.org/abs/2406.10774)

**12. HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading**
- Authors: Sun, Cai et al.; Venue: arXiv Feb 2025
- Key idea: Fine-grained head-wise offloading -- dynamically keeps only selective attention heads' KV cache on GPU while offloading rest to CPU. Asynchronous prefetching via PCIe.
- Results: Reduces KV cache GPU footprint from 128GB to 1GB for Llama-3-8B at 1M tokens (92% reduction). Enables 4M-token inference on a single RTX 4090.
- **Relation to your work: HeadInfer demonstrates that head-level granularity is effective for offloading decisions. Your token-level importance scoring operates at a finer granularity. The approaches could be complementary (head-level + token-level hierarchy).**
- [arXiv link](https://arxiv.org/abs/2502.12574)

**13. ArkVale: Efficient Generative LLM Inference with Recallable Key-Value Eviction**
- Authors: multiple; Venue: NeurIPS 2024
- Key idea: Page-based KV cache manager that can RECALL previously evicted tokens. Asynchronously copies filled pages to CPU memory as backup, creates compact "digests" (bounding volumes of keys) for importance estimation. Recalls important evicted pages before attention.
- Results: Negligible accuracy loss at 2K-4K cache budget; 2.2x decoding latency, 4.6x throughput.
- **Relation to your work: ArkVale is very close to your memory hierarchy concept -- it explicitly uses CPU memory as a secondary tier with recall capability. The key difference is your semantics-aware importance scoring for reasoning tokens vs. ArkVale's geometric bounding-volume approach. Your work should cite ArkVale and distinguish the reasoning-specific contribution.**
- [NeurIPS 2024](https://openreview.net/forum?id=4oAt5L4lYe)

**14. ScoutAttention: Efficient KV Cache Offloading via Layer-Ahead CPU Pre-computation**
- Authors: multiple; Venue: DAC 2026
- Key idea: CPU initiates attention computation one layer in advance. GPU-CPU collaborative block-wise sparse attention retains only critical blocks on GPU.
- Results: <2.1% accuracy degradation; 5.1x decoding throughput vs. full attention; 2.1x vs. existing offloading.
- **Relation to your work: ScoutAttention's GPU-CPU collaboration is an implementation of the memory hierarchy you propose. Concurrent/very recent work. Your distinction is reasoning-specific importance scoring.**
- [arXiv link](https://arxiv.org/abs/2603.27138)

---

### D. KV Cache Quantization/Compression (orthogonal, can combine)

**15. KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache**
- Authors: Liu, Yuan et al.; Venue: ICML 2024
- Key idea: Key cache should be quantized per-channel; value cache per-token. Achieves 2-bit quantization without fine-tuning.
- **Relation to your work: Quantization is orthogonal to your tiered offloading. Your "compressed tier" could use KIVI-style quantization. Cite as complementary.**
- [arXiv link](https://arxiv.org/abs/2402.02750)

**16. GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference**
- Authors: Kang et al.; Venue: NeurIPS 2024
- Key idea: Three-component approach: ultra-low precision quantization for majority entries + low-rank matrix for quantization error + sparse matrix for outliers.
- Results: Maintains FP16-like accuracy with up to 24.42% improvement over SOTA at 2-bit.
- **Relation to your work: GEAR's multi-component compression could serve as the compression method for your lower memory tiers.**
- [NeurIPS 2024](https://neurips.cc/virtual/2024/106424)

**17. MiniCache: KV Cache Compression in Depth Dimension**
- Authors: multiple; Venue: NeurIPS 2024
- Key idea: Exploits inter-layer KV cache similarity (depth dimension). Merges KV states across adjacent layers by decomposing into magnitude + direction, interpolating directions.
- Results: Up to 5.02x compression combined with 4-bit quantization.
- **Relation to your work: Orthogonal -- MiniCache compresses across layers while you compress across tokens within layers. Could be combined.**
- [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fd0705710bf01b88a60a3d479ea341d9-Abstract-Conference.html)

**18. PyramidKV / PyramidInfer**
- PyramidKV (arXiv 2024 / OpenReview): Observes "pyramidal information funneling" -- attention is broad in lower layers, focused in higher layers. Dynamically adjusts KV cache size per layer. Preserves performance with 12% of KV cache on LongBench.
- PyramidInfer (ACL Findings 2024): Layer-wise crucial context retention during prefill phase. 2.2x throughput, 54% GPU memory reduction.
- **Relation to your work: The layer-varying importance insight could enhance your approach -- different layers might need different HBM/DDR ratios.**
- [PyramidKV](https://openreview.net/forum?id=jZVNmDiU86) | [PyramidInfer](https://aclanthology.org/2024.findings-acl.195/)

**19. Attention Score is not All You Need for Token Importance (VATP)**
- Authors: multiple; Venue: EMNLP 2024
- Key idea: Value vector norms are non-uniformly distributed and provide complementary importance signal. Attention sink tokens have massive attention scores but small value norms. Proposes Value-Aware Token Pruning (VATP) combining both signals.
- **Relation to your work: Directly challenges pure attention-score-based importance. Your importance scoring should account for value norms too, or at minimum discuss this finding.**
- [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.1178.pdf)

**20. Dynamic Memory Compression (DMC)**
- Authors: NVIDIA; Venue: ICML 2024
- Key idea: Learns during continued pre-training whether to append or accumulate tokens into KV cache, achieving different compression ratios per head/layer.
- Results: Up to 3.7x throughput on H100.
- **Relation to your work: DMC requires continued pre-training, while your approach is training-free. Different design philosophy.**
- [arXiv link](https://arxiv.org/abs/2403.09636)

---

## PART 2: Does Existing Work Target Reasoning Models Specifically?

Yes, and this is a rapidly growing subfield (mostly 2025-2026):

| Paper | Reasoning-Specific? | Models Tested |
|-------|---------------------|---------------|
| **ThinKV** (Oct 2025) | Yes -- segments CoT into thought types | DeepSeek-R1-Distill, GPT-OSS, AceReason |
| **R-KV** (NeurIPS 2025) | Yes -- redundancy-aware for CoT | DeepSeek-R1, reasoning models |
| **Hold Onto That Thought** (NeurIPS 2025) | Yes -- benchmarks compression on reasoning | DeepSeek-R1-Distill-Qwen-7B/14B |
| **TriAttention** (Apr 2026) | Yes -- targets long reasoning generation | AIME25, MATH-500 |
| All others (StreamingLLM, H2O, etc.) | No -- general LLM compression | Standard LLMs |

**Key finding**: The reasoning-specific KV cache papers are all very recent (late 2025 - early 2026), meaning this is a hot and timely topic. However, none of them frame the problem as a **memory hierarchy** design (HBM -> DDR -> compressed -> evicted). They all operate within a single memory tier (HBM eviction/compression).

---

## PART 3: Key Differences and Potential Conflicts with Your Approach

### Your Unique Contributions (gaps you fill)
1. **Memory hierarchy framing**: No existing work explicitly proposes tiered HBM/DDR/compressed storage based on reasoning token importance. ArkVale and InfiniGen do offloading but not with reasoning-specific importance scoring.
2. **Semantics-aware importance for reasoning**: ThinKV segments by thought type, R-KV uses redundancy -- your attention-based importance scoring with semantic awareness is a distinct mechanism.
3. **The "not all thoughts need HBM" insight**: Framing this as a memory hierarchy problem rather than a compression problem is novel.

### Potential Conflicts / Challenges to Address
1. **ThinKV overlap**: ThinKV already does thought-type-aware differentiated compression on the same models. You need to clearly show your memory hierarchy approach provides benefits ThinKV's single-tier approach cannot (e.g., graceful degradation, larger effective cache via DDR).
2. **R-KV's strong results**: R-KV achieves 100% accuracy at 10% cache and even 105% at 16%. Your approach needs to match or explain why tiered storage offers advantages beyond raw accuracy (e.g., throughput, cost, latency predictability).
3. **VATP criticism**: Your attention-based importance scoring may be challenged by the EMNLP 2024 finding that attention scores alone are insufficient -- value vector norms matter too.
4. **"Hold Onto That Thought" benchmark**: This paper evaluates on your exact model + dataset. Ensure your baseline comparisons are consistent with their findings.
5. **Offloading latency**: Papers like ScoutAttention and HeadInfer show that PCIe bandwidth is a real bottleneck for GPU-CPU offloading. Your paper needs to address whether the latency of fetching from DDR undermines the throughput gains.

### Recommended Positioning
Your paper sits at the intersection of two subfields that haven't been combined: (a) reasoning-specific KV importance (ThinKV, R-KV) and (b) tiered KV cache offloading (InfiniGen, ArkVale, HeadInfer). The novelty is applying semantic/reasoning-aware importance scoring to drive memory hierarchy placement decisions, rather than simple eviction or single-tier compression.

Sources:
- [StreamingLLM (Xiao et al., ICLR 2024)](https://arxiv.org/abs/2309.17453)
- [H2O (NeurIPS 2023)](https://arxiv.org/abs/2306.14048)
- [ScissorHands (NeurIPS 2024)](https://openreview.net/forum?id=JZfg6wGi6g)
- [SnapKV (NeurIPS 2024)](https://arxiv.org/abs/2404.14469)
- [FastGen](https://arxiv.org/abs/2310.01801)
- [PyramidKV](https://openreview.net/forum?id=jZVNmDiU86)
- [PyramidInfer (ACL Findings 2024)](https://aclanthology.org/2024.findings-acl.195/)
- [ThinKV](https://arxiv.org/abs/2510.01290)
- [R-KV (NeurIPS 2025)](https://arxiv.org/abs/2505.24133)
- [Hold Onto That Thought (NeurIPS 2025)](https://arxiv.org/abs/2512.12008)
- [TriAttention](https://arxiv.org/abs/2604.04921)
- [InfiniGen (OSDI 2024)](https://www.usenix.org/conference/osdi24/presentation/lee)
- [Quest (ICML 2024)](https://arxiv.org/abs/2406.10774)
- [HeadInfer](https://arxiv.org/abs/2502.12574)
- [ArkVale (NeurIPS 2024)](https://openreview.net/forum?id=4oAt5L4lYe)
- [ScoutAttention (DAC 2026)](https://arxiv.org/abs/2603.27138)
- [KIVI (ICML 2024)](https://arxiv.org/abs/2402.02750)
- [GEAR (NeurIPS 2024)](https://neurips.cc/virtual/2024/106424)
- [MiniCache (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fd0705710bf01b88a60a3d479ea341d9-Abstract-Conference.html)
- [VATP (EMNLP 2024)](https://arxiv.org/abs/2406.12335)
- [DMC (ICML 2024)](https://arxiv.org/abs/2403.09636)
- [Multi-Tier Dynamic Storage](https://link.springer.com/article/10.1007/s40747-025-02200-4)
- [Adaptive Multi-Objective Tiered Storage](https://arxiv.org/abs/2603.08739)
- [CacheGen (SIGCOMM 2024)](https://arxiv.org/abs/2310.07240)
- [NVIDIA KV Cache Offloading Blog](https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/)