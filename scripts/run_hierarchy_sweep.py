"""Sweep evict_ratio to find the sweet spot for memory hierarchy.

Tests: evict_ratio = 0%, 3%, 5%, 7%, 10%, 15%
All at 30% HBM budget (since we proved HBM ratio doesn't matter).
Also tests 8-bit quantization (instead of 4-bit which failed).

Usage:
    python scripts/run_hierarchy_sweep.py \
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


def evict_kv_cache(past_kv, keep_indices):
    indices = torch.tensor(keep_indices, device=past_kv.key_cache[0].device, dtype=torch.long)
    new_cache = DynamicCache()
    for layer_idx in range(len(past_kv.key_cache)):
        k = past_kv.key_cache[layer_idx].index_select(2, indices)
        v = past_kv.value_cache[layer_idx].index_select(2, indices)
        new_cache.key_cache.append(k)
        new_cache.value_cache.append(v)
    return new_cache


def quantize_kv_positions(past_kv, positions, bits=8):
    if not positions:
        return past_kv
    pos_tensor = torch.tensor(positions, device=past_kv.key_cache[0].device, dtype=torch.long)
    n_levels = 2 ** bits
    for layer_idx in range(len(past_kv.key_cache)):
        for cache in [past_kv.key_cache, past_kv.value_cache]:
            selected = cache[layer_idx].index_select(2, pos_tensor)
            vmin = selected.min()
            vmax = selected.max()
            if vmax - vmin < 1e-8:
                continue
            scale = (vmax - vmin) / (n_levels - 1)
            quantized = torch.round((selected - vmin) / scale) * scale + vmin
            cache[layer_idx][:, :, pos_tensor] = quantized
    return past_kv


def generate_with_hierarchy(model, tokenizer, question, device,
                            evict_ratio=0.0, quantize_offloaded=False,
                            quant_bits=8, sink_size=4, window_size=128,
                            max_new_tokens=2048):
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

    eviction_interval = 64
    total_evicted = 0

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

        if ((step + 1) % eviction_interval == 0
            and evict_ratio > 0
            and n_reasoning > sink_size + window_size):

            protected = set()
            for i in range(min(prompt_len, cache_len)):
                protected.add(i)
            for i in range(prompt_len, min(prompt_len + sink_size, cache_len)):
                protected.add(i)
            for i in range(max(0, cache_len - window_size), cache_len):
                protected.add(i)

            candidates = [i for i in range(cache_len) if i not in protected]
            if not candidates:
                next_token = out.logits[:, -1:, :].argmax(dim=-1)
                continue

            cand_imp = np.array([importance[position_map[c]] for c in candidates])
            sorted_idx = np.argsort(cand_imp)

            n_evict = int(len(candidates) * evict_ratio)

            evict_positions = set()
            for i in range(n_evict):
                evict_positions.add(candidates[sorted_idx[i]])

            if evict_positions:
                keep_sorted = sorted(
                    [i for i in range(cache_len) if i not in evict_positions]
                )
                past_kv = evict_kv_cache(past_kv, keep_sorted)
                position_map = [position_map[i] for i in keep_sorted]
                total_evicted += len(evict_positions)

            if quantize_offloaded:
                new_cache_len = past_kv.get_seq_length()
                new_protected = set()
                for i in range(min(prompt_len, new_cache_len)):
                    new_protected.add(i)
                for i in range(prompt_len, min(prompt_len + sink_size, new_cache_len)):
                    new_protected.add(i)
                for i in range(max(0, new_cache_len - window_size), new_cache_len):
                    new_protected.add(i)

                new_candidates = [
                    i for i in range(new_cache_len) if i not in new_protected
                ]
                if new_candidates:
                    new_cand_imp = np.array([
                        importance[position_map[c]] for c in new_candidates
                    ])
                    new_sorted = np.argsort(new_cand_imp)
                    n_quant = len(new_candidates) // 2
                    quant_positions = [
                        new_candidates[new_sorted[i]] for i in range(n_quant)
                    ]
                    past_kv = quantize_kv_positions(
                        past_kv, quant_positions, bits=quant_bits
                    )

        next_token = out.logits[:, -1:, :].argmax(dim=-1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True), total_evicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results/hierarchy_sweep")
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

    configs = [
        # Baseline
        ("Full (0% evict)", 0.0, False, 8),
        # Sweep evict_ratio
        ("Hierarchy (3% evict)", 0.03, False, 8),
        ("Hierarchy (5% evict)", 0.05, False, 8),
        ("Hierarchy (7% evict)", 0.07, False, 8),
        ("Hierarchy (10% evict)", 0.10, False, 8),
        ("Hierarchy (15% evict)", 0.15, False, 8),
        # With 8-bit quantization
        ("Hierarchy+Q8 (5% evict)", 0.05, True, 8),
        ("Hierarchy+Q8 (10% evict)", 0.10, True, 8),
    ]

    all_results = {}

    for label, evict_ratio, do_quant, qbits in configs:
        print(f"\n{'='*60}")
        print(f"=== {label} ===")
        print(f"{'='*60}")
        correct = 0
        tested = 0
        total_evicted_all = 0

        for sample in tqdm(samples, desc=label[:40]):
            try:
                text, n_evicted = generate_with_hierarchy(
                    model, tokenizer, sample["question"], args.device,
                    evict_ratio=evict_ratio,
                    quantize_offloaded=do_quant,
                    quant_bits=qbits,
                    sink_size=args.sink_size,
                    window_size=args.window_size,
                    max_new_tokens=args.max_new_tokens,
                )
                pred = extract_answer(text)
                if check_answer(pred, sample["answer"]):
                    correct += 1
                total_evicted_all += n_evicted
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Error: {e}")
            tested += 1

        acc = correct / max(1, tested)
        avg_evicted = total_evicted_all / max(1, tested)
        all_results[label] = {
            "accuracy": acc, "correct": correct, "tested": tested,
            "avg_evicted": avg_evicted,
        }
        print(f"  Accuracy: {acc:.1%} ({correct}/{tested})")
        print(f"  Avg tokens evicted: {avg_evicted:.0f}")

    # Summary
    print("\n" + "=" * 70)
    print("HIERARCHY SWEEP RESULTS")
    print("=" * 70)
    for label, res in all_results.items():
        print(f"  {label:40s}  {res['accuracy']:.1%}  (evicted ~{res['avg_evicted']:.0f} tokens)")

    # Plot
    sweep_data = []
    quant_data = []
    for label, res in all_results.items():
        if "+Q8" in label:
            ratio = float(label.split("(")[1].split("%")[0]) / 100
            quant_data.append((ratio, res["accuracy"]))
        else:
            ratio = float(label.split("(")[1].split("%")[0]) / 100
            sweep_data.append((ratio, res["accuracy"]))

    sweep_data.sort()
    quant_data.sort()

    plt.figure(figsize=(7, 4))
    if sweep_data:
        plt.plot([d[0]*100 for d in sweep_data], [d[1]*100 for d in sweep_data],
                 marker="o", linewidth=2, markersize=8, label="Hierarchy (fp16)", color="#1565C0")
    if quant_data:
        plt.plot([d[0]*100 for d in quant_data], [d[1]*100 for d in quant_data],
                 marker="s", linewidth=2, markersize=7, label="Hierarchy + Q8", color="#2E7D32",
                 linestyle="--")

    # Add annotations
    for d in sweep_data:
        plt.annotate(f"{d[1]:.0%}", (d[0]*100, d[1]*100),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=9)

    plt.xlabel("Eviction Ratio (%)")
    plt.ylabel("GSM8K Accuracy (%)")
    plt.title("Memory Hierarchy: Accuracy vs Eviction Ratio\n(30% HBM budget, DeepSeek-R1-Distill-Qwen-7B, n={})".format(args.n_samples))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-2, 80)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "hierarchy_sweep.png"), dpi=150)
    plt.close()

    summary = {
        "model": args.model, "n_samples": args.n_samples,
        "sink_size": args.sink_size, "window_size": args.window_size,
        "results": all_results,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
