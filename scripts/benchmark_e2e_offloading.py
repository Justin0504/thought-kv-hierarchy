"""Task 3: End-to-End Offloading System Prototype

Implements real GPU-CPU KV cache offloading during generation and
measures actual tokens/sec throughput.

Compares:
  1. Full attention (100% GPU, no eviction) — baseline
  2. Pure eviction (evict beyond budget) — existing method
  3. Hierarchy offloading (offload to CPU, prefetch for attention)

Usage:
    cd ~/thought-kv-hierarchy
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ~/thought-kv-hierarchy/venv/bin/python3 scripts/benchmark_e2e_offloading.py \
        --n_samples 50 --device cuda:0
"""

import argparse
import json
import os
import sys
import gc
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DynamicCache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.eval.gsm8k import load_gsm8k, extract_answer, check_answer
from src.system.hierarchy_offloader import HierarchyOffloader, get_n_layers, get_kv


# ── KV-cache eviction (same as algorithm-side) ──────────────────────

def evict_kv_cache(past_kv, keep_indices, device):
    indices = torch.tensor(keep_indices, device=device, dtype=torch.long)
    new_cache = DynamicCache()
    for layer_idx in range(get_n_layers(past_kv)):
        k, v = get_kv(past_kv, layer_idx)
        new_cache.update(k.index_select(2, indices), v.index_select(2, indices), layer_idx)
    return new_cache


# ── Mode 1: Full attention (baseline) ───────────────────────────────

def generate_full(model, tokenizer, question, device, max_new_tokens=2048):
    """Full attention, no eviction, no offloading."""
    prompt = (
        "Please solve this math problem step by step.\n\n"
        f"Question: {question}\n\nLet me think step by step."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        past_kv = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        del out; torch.cuda.empty_cache()

        generated_ids = []
        for step in range(max_new_tokens):
            tok_id = next_token.item()
            if tok_id == tokenizer.eos_token_id:
                break
            generated_ids.append(tok_id)
            out = model(next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
            del out; torch.cuda.empty_cache()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    del past_kv; torch.cuda.empty_cache(); gc.collect()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    n_tokens = len(generated_ids)
    return text, n_tokens, elapsed


# ── Mode 2: Pure eviction ───────────────────────────────────────────

def generate_eviction(model, tokenizer, question, device,
                      hbm_ratio=0.5, sink_size=4, window_size=128,
                      max_new_tokens=2048):
    """Streaming eviction — tokens beyond budget are permanently lost."""
    prompt = (
        "Please solve this math problem step by step.\n\n"
        f"Question: {question}\n\nLet me think step by step."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model(**inputs, use_cache=True, output_attentions=True)
        past_kv = out.past_key_values

    current_len = prompt_len
    importance = np.zeros(current_len, dtype=np.float64)
    position_map = list(range(current_len))
    for layer_attn in out.attentions:
        s = layer_attn[0].float().mean(dim=0).sum(dim=0).cpu().numpy()
        if not np.isnan(s).any():
            importance[:len(s)] += s

    generated_ids = []
    next_token = out.logits[:, -1:, :].argmax(dim=-1)
    del out; torch.cuda.empty_cache()

    for step in range(max_new_tokens):
        tok_id = next_token.item()
        if tok_id == tokenizer.eos_token_id:
            break
        generated_ids.append(tok_id)

        with torch.no_grad():
            out = model(next_token, past_key_values=past_kv, use_cache=True, output_attentions=True)
            past_kv = out.past_key_values

        current_len += 1
        position_map.append(current_len - 1)
        if current_len > len(importance):
            importance = np.concatenate([importance, np.zeros(current_len - len(importance))])

        for layer_attn in out.attentions:
            s = layer_attn[0].float().mean(dim=0).squeeze(0).cpu().numpy()
            if not np.isnan(s).any():
                for ci, orig_pos in enumerate(position_map):
                    if ci < len(s):
                        importance[orig_pos] += s[ci]

        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        del out; torch.cuda.empty_cache()

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
            past_kv = evict_kv_cache(past_kv, keep_sorted, device)
            position_map = [position_map[i] for i in keep_sorted]

        next_token_holder = next_token

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    del past_kv; torch.cuda.empty_cache(); gc.collect()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, len(generated_ids), elapsed


# ── Mode 3: Hierarchy offloading (real GPU-CPU movement) ─────────────

def generate_hierarchy(model, tokenizer, question, device,
                       hbm_ratio=0.5, evict_ratio=0.1,
                       sink_size=4, window_size=128,
                       manage_interval=64, max_new_tokens=2048):
    """Real hierarchy: offload to CPU DDR, prefetch for attention."""
    prompt = (
        "Please solve this math problem step by step.\n\n"
        f"Question: {question}\n\nLet me think step by step."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    offloader = HierarchyOffloader(
        n_layers=28, n_heads=28, head_dim=128, device=device,
        hbm_ratio=hbm_ratio, evict_ratio=evict_ratio,
        sink_size=sink_size, window_size=window_size,
        manage_interval=manage_interval,
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model(**inputs, use_cache=True, output_attentions=True)
        past_kv = out.past_key_values

    current_len = prompt_len
    importance = np.zeros(current_len, dtype=np.float64)
    position_map = list(range(current_len))
    for layer_attn in out.attentions:
        s = layer_attn[0].float().mean(dim=0).sum(dim=0).cpu().numpy()
        if not np.isnan(s).any():
            importance[:len(s)] += s

    generated_ids = []
    next_token = out.logits[:, -1:, :].argmax(dim=-1)
    del out; torch.cuda.empty_cache()

    for step in range(max_new_tokens):
        tok_id = next_token.item()
        if tok_id == tokenizer.eos_token_id:
            break
        generated_ids.append(tok_id)

        # --- Prefetch CPU tokens for attention if any are offloaded ---
        if offloader.cpu_positions:
            full_cache, prefetch_ms = offloader.prefetch_and_build_full_cache(past_kv)
        else:
            full_cache = past_kv
            prefetch_ms = 0.0

        # --- Forward pass (timed for compute) ---
        torch.cuda.synchronize()
        compute_start = torch.cuda.Event(enable_timing=True)
        compute_end = torch.cuda.Event(enable_timing=True)
        compute_start.record()

        with torch.no_grad():
            out = model(next_token, past_key_values=full_cache,
                        use_cache=True, output_attentions=True)

        compute_end.record()
        torch.cuda.synchronize()
        offloader.stats["compute_time_ms"] += compute_start.elapsed_time(compute_end)
        offloader.stats["total_steps"] += 1

        # --- Cleanup: remove prefetched tokens, keep only GPU-resident + new ---
        if offloader.cpu_positions:
            past_kv = offloader.cleanup_prefetched(out.past_key_values, past_kv)
        else:
            past_kv = out.past_key_values

        current_len += 1
        position_map.append(current_len - 1)
        if current_len > len(importance):
            importance = np.concatenate([importance, np.zeros(current_len - len(importance))])

        for layer_attn in out.attentions:
            s = layer_attn[0].float().mean(dim=0).squeeze(0).cpu().numpy()
            if not np.isnan(s).any():
                for ci, orig_pos in enumerate(position_map):
                    if ci < len(s):
                        importance[orig_pos] += s[ci]

        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        del out, full_cache; torch.cuda.empty_cache()

        # --- Tier management: offload + evict ---
        past_kv, position_map = offloader.manage_cache(
            past_kv, importance, position_map, prompt_len, step
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    del past_kv; torch.cuda.empty_cache(); gc.collect()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, len(generated_ids), elapsed, offloader.get_summary()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results/system/task3")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()

    samples = load_gsm8k(n_samples=args.n_samples)
    print(f"Loaded {len(samples)} samples\n")

    # ═══ Configurations ═══
    configs = [
        ("Full (100% GPU)", "full", {}),
        ("Eviction (50% HBM)", "eviction", {"hbm_ratio": 0.5}),
        ("Eviction (30% HBM)", "eviction", {"hbm_ratio": 0.3}),
        ("Hierarchy (50% HBM, 10% evict)", "hierarchy",
         {"hbm_ratio": 0.5, "evict_ratio": 0.1}),
        ("Hierarchy (30% HBM, 10% evict)", "hierarchy",
         {"hbm_ratio": 0.3, "evict_ratio": 0.1}),
        ("Hierarchy (50% HBM, 5% evict)", "hierarchy",
         {"hbm_ratio": 0.5, "evict_ratio": 0.05}),
    ]

    all_results = {}

    for label, mode, params in configs:
        print(f"\n{'='*60}")
        print(f"=== {label} ===")
        print(f"{'='*60}")

        correct = 0
        tested = 0
        total_tokens = 0
        total_time = 0.0
        offloader_summaries = []

        for i, sample in enumerate(tqdm(samples, desc=label[:40])):
            try:
                if mode == "full":
                    text, n_tok, elapsed = generate_full(
                        model, tokenizer, sample["question"], args.device,
                        max_new_tokens=args.max_new_tokens)
                    summary = None
                elif mode == "eviction":
                    text, n_tok, elapsed = generate_eviction(
                        model, tokenizer, sample["question"], args.device,
                        hbm_ratio=params["hbm_ratio"],
                        sink_size=args.sink_size, window_size=args.window_size,
                        max_new_tokens=args.max_new_tokens)
                    summary = None
                elif mode == "hierarchy":
                    text, n_tok, elapsed, summary = generate_hierarchy(
                        model, tokenizer, sample["question"], args.device,
                        hbm_ratio=params.get("hbm_ratio", 0.5),
                        evict_ratio=params.get("evict_ratio", 0.1),
                        sink_size=args.sink_size, window_size=args.window_size,
                        max_new_tokens=args.max_new_tokens)
                    offloader_summaries.append(summary)

                pred = extract_answer(text)
                if check_answer(pred, sample["answer"]):
                    correct += 1
                total_tokens += n_tok
                total_time += elapsed

            except torch.cuda.OutOfMemoryError:
                print(f"  Sample {i}: OOM")
                torch.cuda.empty_cache(); gc.collect()
            except Exception as e:
                print(f"  Sample {i}: Error: {e}")
                torch.cuda.empty_cache(); gc.collect()
            tested += 1

        acc = correct / max(1, tested)
        tokens_per_sec = total_tokens / max(0.001, total_time)
        avg_time = total_time / max(1, tested)

        result = {
            "accuracy": acc,
            "correct": correct,
            "tested": tested,
            "total_tokens": total_tokens,
            "total_time_sec": round(total_time, 2),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "avg_time_per_sample_sec": round(avg_time, 2),
        }

        # Aggregate offloader stats for hierarchy runs
        if offloader_summaries:
            avg_offload_ms = np.mean([s["offload"]["total_ms"] for s in offloader_summaries])
            avg_prefetch_ms = np.mean([s["prefetch"]["total_ms"] for s in offloader_summaries])
            avg_compute_ms = np.mean([s["compute"]["total_ms"] for s in offloader_summaries])
            avg_offload_tokens = np.mean([s["offload"]["total_tokens"] for s in offloader_summaries])
            avg_prefetch_tokens = np.mean([s["prefetch"]["total_tokens"] for s in offloader_summaries])
            avg_overhead = np.mean([s["transfer_overhead_pct"] for s in offloader_summaries])

            result["offloader_avg"] = {
                "offload_ms": round(avg_offload_ms, 2),
                "prefetch_ms": round(avg_prefetch_ms, 2),
                "compute_ms": round(avg_compute_ms, 2),
                "offload_tokens": round(avg_offload_tokens, 1),
                "prefetch_tokens": round(avg_prefetch_tokens, 1),
                "transfer_overhead_pct": round(avg_overhead, 1),
            }

        all_results[label] = result
        print(f"  Accuracy: {acc:.1%} ({correct}/{tested})")
        print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")
        print(f"  Avg time/sample: {avg_time:.1f}s")
        if offloader_summaries:
            print(f"  Avg offload: {avg_offload_ms:.1f}ms, prefetch: {avg_prefetch_ms:.1f}ms")
            print(f"  Transfer overhead: {avg_overhead:.1f}%")

    # ═══ Save results ═══
    with open(os.path.join(args.output_dir, "e2e_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # ═══ Summary ═══
    print("\n" + "=" * 70)
    print("END-TO-END OFFLOADING RESULTS")
    print("=" * 70)
    print(f"{'Config':<40s} {'Accuracy':>8s} {'tok/s':>8s} {'Avg(s)':>8s}")
    print("-" * 70)
    for label, res in all_results.items():
        print(f"  {label:<38s} {res['accuracy']:>7.1%} {res['tokens_per_sec']:>7.1f} {res['avg_time_per_sample_sec']:>7.1f}")

    # ═══ Plots ═══
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 13, "figure.dpi": 150, "savefig.bbox": "tight",
    })

    # Fig 1: Accuracy + Throughput comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    labels_short = [l.replace("Hierarchy", "Hier.").replace("Eviction", "Evict.") for l in all_results.keys()]
    accs = [r["accuracy"] for r in all_results.values()]
    tps = [r["tokens_per_sec"] for r in all_results.values()]

    colors = []
    for l in all_results.keys():
        if "Full" in l:
            colors.append("#1565C0")
        elif "Eviction" in l:
            colors.append("#E53935")
        else:
            colors.append("#FF8F00")

    bars1 = ax1.barh(range(len(labels_short)), accs, color=colors, height=0.6)
    ax1.set_yticks(range(len(labels_short)))
    ax1.set_yticklabels(labels_short, fontsize=9)
    ax1.set_xlabel("GSM8K Accuracy")
    ax1.set_title("Accuracy Comparison")
    ax1.set_xlim(0, 1.0)
    ax1.invert_yaxis()
    for bar, a in zip(bars1, accs):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{a:.0%}", va="center", fontsize=9)

    bars2 = ax2.barh(range(len(labels_short)), tps, color=colors, height=0.6)
    ax2.set_yticks(range(len(labels_short)))
    ax2.set_yticklabels(labels_short, fontsize=9)
    ax2.set_xlabel("Tokens/sec")
    ax2.set_title("Throughput Comparison")
    ax2.invert_yaxis()
    for bar, t in zip(bars2, tps):
        ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                 f"{t:.1f}", va="center", fontsize=9)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1565C0", label="Full (baseline)"),
        Patch(facecolor="#E53935", label="Pure eviction"),
        Patch(facecolor="#FF8F00", label="Hierarchy (offload)"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "fig_accuracy_throughput.png"), dpi=200)
    plt.savefig(os.path.join(args.output_dir, "fig_accuracy_throughput.pdf"))
    plt.close()
    print("  Saved fig_accuracy_throughput")

    # Fig 2: Latency breakdown for hierarchy configs
    hierarchy_results = {k: v for k, v in all_results.items() if "offloader_avg" in v}
    if hierarchy_results:
        fig, ax = plt.subplots(figsize=(9, 5))
        labels_h = list(hierarchy_results.keys())
        offload_ms = [v["offloader_avg"]["offload_ms"] for v in hierarchy_results.values()]
        prefetch_ms = [v["offloader_avg"]["prefetch_ms"] for v in hierarchy_results.values()]
        compute_ms = [v["offloader_avg"]["compute_ms"] for v in hierarchy_results.values()]

        x = np.arange(len(labels_h))
        w = 0.6
        ax.bar(x, compute_ms, w, label="Compute (attention + FFN)", color="#1565C0", alpha=0.85)
        ax.bar(x, prefetch_ms, w, bottom=compute_ms, label="CPU→GPU prefetch", color="#FF8F00", alpha=0.85)
        bottoms = [c + p for c, p in zip(compute_ms, prefetch_ms)]
        ax.bar(x, offload_ms, w, bottom=bottoms, label="GPU→CPU offload", color="#E53935", alpha=0.85)

        for i, (c, p, o) in enumerate(zip(compute_ms, prefetch_ms, offload_ms)):
            total = c + p + o
            overhead = (p + o) / total * 100 if total > 0 else 0
            ax.text(i, total + 50, f"{overhead:.0f}% overhead", ha="center", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("Hierarchy ", "").replace("(", "\n(") for l in labels_h], fontsize=9)
        ax.set_ylabel("Total Time (ms)")
        ax.set_title("Latency Breakdown: Compute vs Transfer\n(per sample, averaged)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "fig_latency_breakdown.png"), dpi=200)
        plt.savefig(os.path.join(args.output_dir, "fig_latency_breakdown.pdf"))
        plt.close()
        print("  Saved fig_latency_breakdown")

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
