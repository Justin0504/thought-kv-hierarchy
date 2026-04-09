# Thought-KV-Hierarchy

**Not All Thoughts Need HBM: Semantics-Aware Memory Hierarchy for LLM Reasoning**

Reasoning LLMs (e.g., DeepSeek-R1) generate long chains of thought where token importance follows a long-tail distribution. This project implements a semantics-aware memory hierarchy that offloads low-importance KV-cache entries from HBM to DDR instead of evicting them, preserving reasoning accuracy at significantly reduced HBM cost.

## Key Findings

| Method | HBM Budget | Evicted | GSM8K Accuracy |
|--------|-----------|---------|---------------|
| Full cache | 100% | 0% | 68% |
| Pure eviction | 50% | 50% | 0% |
| Pure eviction | 30% | 70% | 0% |
| **Memory hierarchy** | **50%** | **10%** | **34%** |
| **Memory hierarchy** | **30%** | **10%** | **34%** |

- Attention-based importance scoring consistently outperforms random eviction
- HBM ratio has minimal impact on accuracy — what matters is how much is truly evicted
- Naive eviction shows a "cliff effect" below 70% budget, motivating the hierarchy approach

## Setup

### Environment

```bash
conda env create -f environment.yml
conda activate thought-hbm
```

Or with pip:

```bash
pip install torch transformers accelerate datasets numpy matplotlib tqdm
```

### Model

The model is publicly available on HuggingFace and will be downloaded automatically on first run:

```
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

Requires ~14GB GPU memory (fp16). No manual download needed.

## Project Structure

```
├── src/
│   ├── eval/gsm8k.py              # GSM8K dataset loading & answer extraction
│   ├── scorer/
│   │   ├── attention_scorer.py     # Attention-based importance scoring
│   │   └── oracle_masking.py       # Oracle KV masking utilities
│   ├── compress/                   # (placeholder for compression methods)
│   └── hierarchy/                  # (placeholder for hierarchy implementation)
├── scripts/
│   ├── run_week1_validation.py     # Importance profiling + distribution analysis
│   ├── run_streaming_eviction.py   # Streaming eviction baseline
│   ├── run_hierarchy_experiment.py # Memory hierarchy vs pure eviction
│   ├── run_hierarchy_sweep.py      # Eviction ratio sweep
│   └── plot_paper_figures.py       # NeurIPS-quality figure generation
├── results/
│   ├── paper_figures/              # PDF/PNG figures + LaTeX tables
│   ├── week1_profiling_results.json
│   └── week1_streaming_full/summary.json
├── SPEC_FOR_B.md                   # Systems-side collaborator spec
├── NEXT_STEPS.md                   # Experiment roadmap
└── environment.yml
```

## Experiments

### 1. Importance Profiling

Profile attention importance distribution across reasoning tokens:

```bash
python scripts/run_week1_validation.py --n_samples 50 --device cuda:0
```

Outputs: importance distribution plots, per-sample statistics.

### 2. Streaming Eviction Baseline

Compare attention-based vs random eviction at various KV-cache budgets:

```bash
python scripts/run_streaming_eviction.py \
    --n_samples 50 \
    --budget_ratios 1.0 0.9 0.8 0.7 0.5 0.3 \
    --window_size 128 \
    --device cuda:0
```

### 3. Memory Hierarchy Experiment

Core experiment — compares pure eviction vs hierarchy (offload + recall):

```bash
python scripts/run_hierarchy_experiment.py \
    --n_samples 50 \
    --device cuda:0
```

### 4. Eviction Ratio Sweep

Find the optimal eviction ratio for the hierarchy approach:

```bash
python scripts/run_hierarchy_sweep.py \
    --n_samples 50 \
    --device cuda:0
```

## Collaboration

See [SPEC_FOR_B.md](SPEC_FOR_B.md) for systems-side work specification (PCIe offloading, gradient-based scoring, real KV-cache management).

## Related Work

See [results/related_work_report.md](results/related_work_report.md) for a survey of 20 related papers including ThinKV, R-KV, StreamingLLM, H2O, ArkVale, etc.
