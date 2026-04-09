"""Memory Hierarchy experiment: the core contribution.

Compares two approaches at the same HBM budget:
1. Pure eviction: evict everything beyond HBM budget (tokens lost forever)
2. Hierarchy: only truly evict bottom evict_ratio%, offload rest to DDR
   (offloaded tokens still participate in attention)

This demonstrates that a memory hierarchy (HBM + DDR + evict) vastly
outperforms naive eviction at the same HBM budget.

Usage:
    python scripts/run_hierarchy_experiment.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --n_samples 50 --device cuda:0
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.eval.gsm8k import load_gsm8k, extract_answer, check_answer


# ── KV-cache operations ──────────────────────────────────────────────

def evict_kv_cache(past_kv, keep_indices):
    """Create new DynamicCache keeping only specified positions."""
    indices = torch.tensor(keep_indices, device=past_kv.key_cache[0].device, dtype=torch.long)
    new_cache = DynamicCache()
    for layer_idx in range(len(past_kv.key_cache)):
        k = past_kv.key_cache[layer_idx].index_select(2, indices)
        v = past_kv.value_cache[layer_idx].index_select(2, indices)
        new_cache.key_cache.append(k)
        new_cache.value_cache.append(v)
    return new_cache


def quantize_kv_positions(past_kv, positions, bits=4):
    """Simulate quantization of KV entries at given cache positions.

    Quantize to `bits`-bit by clamping to a grid. This simulates the
    quality loss from storing in compressed tier (T2).
    """
    if not positions:
        return past_kv
    pos_tensor = torch.tensor(positions, device=past_kv.key_cache[0].device, dtype=torch.long)
    n_levels = 2 ** bits
    for layer_idx in range(len(past_kv.key_cache)):
        for cache in [past_kv.key_cache, past_kv.value_cache]:
            # [batch, heads, seq, dim]
            selected = cache[layer_idx].index_select(2, pos_tensor)
            vmin = selected.min()
            vmax = selected.max()
            if vmax - vmin < 1e-8:
                continue
            # Uniform quantization
            scale = (vmax - vmin) / (n_levels - 1)
            quantized = torch.round((selected - vmin) / scale) * scale + vmin
            # Write back
            cache[layer_idx][:, :, pos_tensor] = quantized
    return past_kv


# ── Generation with hierarchy ────────────────────────────────────────

def generate_with_hierarchy(model, tokenizer, question, device,
                            hbm_ratio=1.0, evict_ratio=0.0,
                            quantize_offloaded=False, quant_bits=4,
                            sink_size=4, window_size=128,
                            max_new_tokens=2048):
    """Generate with tiered memory hierarchy.

    Args:
        hbm_ratio: fraction of reasoning tokens kept in HBM (T0).
                   Only used for tracking/reporting, not for eviction.
        evict_ratio: fraction of reasoning tokens truly evicted (T3).
                     This determines actual information loss.
        quantize_offloaded: if True, quantize T2-tier tokens.
        quant_bits: quantization bits for T2 tier.

    Tier assignment (of reasoning tokens, excluding protected):
        T0 (HBM):       top hbm_ratio% by importance
        T1 (DDR):        middle tokens (full precision, "offloaded")
        T2 (compressed): low importance tokens (quantized if enabled)
        T3 (evicted):    bottom evict_ratio% — truly removed

    In this simulation:
        - T0, T1, T2 all remain in KV cache (participate in attention)
        - T3 tokens are physically removed from KV cache
        - T2 tokens are optionally quantized
    """
    prompt = (
        "Please solve this math problem step by step.\n\n"
        f"Question: {question}\n\nLet me think step by step."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    # Prefill
    with torch.no_grad():
        out = model(**inputs, use_cache=True, output_attentions=True)
        past_kv = out.past_key_values

    current_len = prompt_len
    importance = np.zeros(current_len, dtype=np.float64)
    for layer_attn in out.attentions:
        s = layer_attn[0].float().mean(dim=0).sum(dim=0).cpu().numpy()
        if not np.isnan(s).any():
            importance[:len(s)] += s

    position_map = list(range(current_len))
    generated_ids = []
    next_token = out.logits[:, -1:, :].argmax(dim=-1)

    # Stats tracking
    tier_stats = {"t0": 0, "t1": 0, "t2": 0, "t3_evicted": 0, "total_steps": 0}
    eviction_interval = 64  # Check every N steps to reduce overhead

    for step in range(max_new_tokens):
        tok_id = next_token.item()
        if tok_id == tokenizer.eos_token_id:
            break
        generated_ids.append(tok_id)

        with torch.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=True,
            )
            past_kv = out.past_key_values

        current_len += 1
        position_map.append(current_len - 1)

        if current_len > len(importance):
            importance = np.concatenate([
                importance, np.zeros(current_len - len(importance))
            ])

        # Update importance
        for layer_attn in out.attentions:
            s = layer_attn[0].float().mean(dim=0).squeeze(0).cpu().numpy()
            if np.isnan(s).any():
                continue
            for ci, orig_pos in enumerate(position_map):
                if ci < len(s):
                    importance[orig_pos] += s[ci]

        # Periodic tier management
        cache_len = past_kv.get_seq_length()
        n_reasoning = len(generated_ids)

        if (step + 1) % eviction_interval == 0 and evict_ratio > 0 and n_reasoning > sink_size + window_size:
            # Identify protected positions
            protected = set()
            for i in range(min(prompt_len, cache_len)):
                protected.add(i)
            for i in range(prompt_len, min(prompt_len + sink_size, cache_len)):
                protected.add(i)
            for i in range(max(0, cache_len - window_size), cache_len):
                protected.add(i)

            # Candidates for tier assignment
            candidates = [i for i in range(cache_len) if i not in protected]
            if not candidates:
                next_token = out.logits[:, -1:, :].argmax(dim=-1)
                continue

            # Sort candidates by importance
            cand_imp = np.array([importance[position_map[c]] for c in candidates])
            sorted_idx = np.argsort(cand_imp)  # ascending (least important first)

            n_cand = len(candidates)
            n_evict = int(n_cand * evict_ratio)  # T3: truly evict

            # T3: bottom evict_ratio% — remove from cache
            evict_positions = set()
            for i in range(n_evict):
                evict_positions.add(candidates[sorted_idx[i]])

            if evict_positions:
                keep_sorted = sorted(
                    [i for i in range(cache_len) if i not in evict_positions]
                )
                past_kv = evict_kv_cache(past_kv, keep_sorted)
                position_map = [position_map[i] for i in keep_sorted]
                tier_stats["t3_evicted"] += len(evict_positions)

            # Optional: quantize low-importance offloaded tokens (T2)
            if quantize_offloaded:
                new_cache_len = past_kv.get_seq_length()
                new_candidates = [
                    i for i in range(new_cache_len)
                    if i not in protected and i >= prompt_len
                ]
                if new_candidates:
                    new_cand_imp = np.array([
                        importance[position_map[c]] for c in new_candidates
                    ])
                    new_sorted = np.argsort(new_cand_imp)
                    # Bottom 50% of remaining candidates get quantized (T2)
                    n_quant = len(new_candidates) // 2
                    quant_positions = [
                        new_candidates[new_sorted[i]] for i in range(n_quant)
                    ]
                    past_kv = quantize_kv_positions(
                        past_kv, quant_positions, bits=quant_bits
                    )

            tier_stats["total_steps"] += 1

        next_token = out.logits[:, -1:, :].argmax(dim=-1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True), tier_stats


def generate_with_pure_eviction(model, tokenizer, question, device,
                                 hbm_ratio=1.0, sink_size=4, window_size=128,
                                 max_new_tokens=2048):
    """Pure eviction baseline (same as streaming eviction script)."""
    prompt = (
        "Please solve this math problem step by step.\n\n"
        f"Question: {question}\n\nLet me think step by step."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model(**inputs, use_cache=True, output_attentions=True)
        past_kv = out.past_key_values

    current_len = prompt_len
    importance = np.zeros(current_len, dtype=np.float64)
    for layer_attn in out.attentions:
        s = layer_attn[0].float().mean(dim=0).sum(dim=0).cpu().numpy()
        if not np.isnan(s).any():
            importance[:len(s)] += s

    position_map = list(range(current_len))
    generated_ids = []
    next_token = out.logits[:, -1:, :].argmax(dim=-1)

    for step in range(max_new_tokens):
        tok_id = next_token.item()
        if tok_id == tokenizer.eos_token_id:
            break
        generated_ids.append(tok_id)

        with torch.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=True,
            )
            past_kv = out.past_key_values

        current_len += 1
        position_map.append(current_len - 1)

        if current_len > len(importance):
            importance = np.concatenate([
                importance, np.zeros(current_len - len(importance))
            ])

        for layer_attn in out.attentions:
            s = layer_attn[0].float().mean(dim=0).squeeze(0).cpu().numpy()
            if np.isnan(s).any():
                continue
            for ci, orig_pos in enumerate(position_map):
                if ci < len(s):
                    importance[orig_pos] += s[ci]

        cache_len = past_kv.get_seq_length()
        n_reasoning = len(generated_ids)
        budget = prompt_len + max(int(n_reasoning * hbm_ratio), sink_size + window_size)

        if cache_len > budget and hbm_ratio < 1.0:
            keep = set()
            for i in range(min(prompt_len, cache_len)):
                keep.add(i)
            for i in range(prompt_len, min(prompt_len + sink_size, cache_len)):
                keep.add(i)
            for i in range(max(0, cache_len - window_size), cache_len):
                keep.add(i)

            n_more = max(0, budget - len(keep))
            candidates = [i for i in range(cache_len) if i not in keep]

            if candidates and n_more > 0:
                cand_imp = np.array([importance[position_map[c]] for c in candidates])
                top_idx = np.argsort(cand_imp)[-n_more:]
                for idx in top_idx:
                    keep.add(candidates[idx])

            keep_sorted = sorted(keep)
            past_kv = evict_kv_cache(past_kv, keep_sorted)
            position_map = [position_map[i] for i in keep_sorted]

        next_token = out.logits[:, -1:, :].argmax(dim=-1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results/hierarchy_experiment")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device,
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()

    samples = load_gsm8k(n_samples=args.n_samples)
    print(f"Loaded {len(samples)} samples\n")

    # ── Experiment configurations ──
    # Each config: (label, mode, params)
    configs = [
        # Baseline: full cache
        ("Full (100% HBM)", "full", {}),

        # Pure eviction at various HBM budgets
        ("Eviction (50% HBM)", "eviction", {"hbm_ratio": 0.5}),
        ("Eviction (30% HBM)", "eviction", {"hbm_ratio": 0.3}),

        # Hierarchy: same HBM budget, but only evict small fraction
        # 50% HBM, evict only 10% (40% offloaded to DDR)
        ("Hierarchy (50% HBM, 10% evict)", "hierarchy",
         {"hbm_ratio": 0.5, "evict_ratio": 0.10}),
        # 50% HBM, evict 20% (30% offloaded to DDR)
        ("Hierarchy (50% HBM, 20% evict)", "hierarchy",
         {"hbm_ratio": 0.5, "evict_ratio": 0.20}),
        # 30% HBM, evict only 10% (60% offloaded to DDR)
        ("Hierarchy (30% HBM, 10% evict)", "hierarchy",
         {"hbm_ratio": 0.3, "evict_ratio": 0.10}),
        # 30% HBM, evict 20% (50% offloaded to DDR)
        ("Hierarchy (30% HBM, 20% evict)", "hierarchy",
         {"hbm_ratio": 0.3, "evict_ratio": 0.20}),

        # Hierarchy + quantization
        ("Hierarchy+Q4 (50% HBM, 10% evict)", "hierarchy",
         {"hbm_ratio": 0.5, "evict_ratio": 0.10,
          "quantize_offloaded": True, "quant_bits": 4}),
        ("Hierarchy+Q4 (30% HBM, 10% evict)", "hierarchy",
         {"hbm_ratio": 0.3, "evict_ratio": 0.10,
          "quantize_offloaded": True, "quant_bits": 4}),
    ]

    all_results = {}

    for label, mode, params in configs:
        print(f"\n{'='*60}")
        print(f"=== {label} ===")
        print(f"{'='*60}")
        correct = 0
        tested = 0

        for sample in tqdm(samples, desc=label[:40]):
            try:
                if mode == "full":
                    text = generate_with_pure_eviction(
                        model, tokenizer, sample["question"], args.device,
                        hbm_ratio=1.0, sink_size=args.sink_size,
                        window_size=args.window_size,
                        max_new_tokens=args.max_new_tokens,
                    )
                elif mode == "eviction":
                    text = generate_with_pure_eviction(
                        model, tokenizer, sample["question"], args.device,
                        hbm_ratio=params["hbm_ratio"],
                        sink_size=args.sink_size,
                        window_size=args.window_size,
                        max_new_tokens=args.max_new_tokens,
                    )
                elif mode == "hierarchy":
                    text, stats = generate_with_hierarchy(
                        model, tokenizer, sample["question"], args.device,
                        hbm_ratio=params.get("hbm_ratio", 0.5),
                        evict_ratio=params.get("evict_ratio", 0.1),
                        quantize_offloaded=params.get("quantize_offloaded", False),
                        quant_bits=params.get("quant_bits", 4),
                        sink_size=args.sink_size,
                        window_size=args.window_size,
                        max_new_tokens=args.max_new_tokens,
                    )

                pred = extract_answer(text)
                if check_answer(pred, sample["answer"]):
                    correct += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Error: {e}")
            tested += 1

        acc = correct / max(1, tested)
        all_results[label] = {"accuracy": acc, "correct": correct, "tested": tested}
        print(f"  Accuracy: {acc:.1%} ({correct}/{tested})")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("MEMORY HIERARCHY EXPERIMENT RESULTS")
    print("=" * 70)
    for label, res in all_results.items():
        print(f"  {label:45s}  {res['accuracy']:.1%} ({res['correct']}/{res['tested']})")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(10, 5))

    # Group by HBM budget for plotting
    categories = list(all_results.keys())
    accs = [all_results[k]["accuracy"] for k in categories]

    colors = []
    for cat in categories:
        if "Full" in cat:
            colors.append("#1565C0")
        elif "Eviction" in cat:
            colors.append("#E53935")
        elif "+Q4" in cat:
            colors.append("#2E7D32")
        else:
            colors.append("#FF8F00")

    bars = ax.barh(range(len(categories)), accs, color=colors, height=0.6)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([c.replace("Hierarchy", "Hier.") for c in categories], fontsize=9)
    ax.set_xlabel("GSM8K Accuracy")
    ax.set_title("Memory Hierarchy vs Pure Eviction\n(DeepSeek-R1-Distill-Qwen-7B, n={})".format(args.n_samples))
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{acc:.0%}", va="center", fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1565C0", label="Full (baseline)"),
        Patch(facecolor="#E53935", label="Pure eviction"),
        Patch(facecolor="#FF8F00", label="Hierarchy (offload)"),
        Patch(facecolor="#2E7D32", label="Hierarchy + quantization"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "hierarchy_vs_eviction.png"), dpi=150)
    plt.close()

    # Save JSON
    summary = {
        "model": args.model,
        "n_samples": args.n_samples,
        "sink_size": args.sink_size,
        "window_size": args.window_size,
        "results": {k: v for k, v in all_results.items()},
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
