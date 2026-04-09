"""Streaming KV-cache eviction experiment.

During generation, maintain a BUDGET for KV-cache size.
When cache exceeds budget, evict least important positions.
Always keep: attention sinks (first few tokens) + recent window.

Usage:
    python scripts/run_streaming_eviction.py \
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
    """Create new DynamicCache keeping only specified positions."""
    indices = torch.tensor(keep_indices, device=past_kv.key_cache[0].device, dtype=torch.long)
    new_cache = DynamicCache()
    for layer_idx in range(len(past_kv.key_cache)):
        k = past_kv.key_cache[layer_idx].index_select(2, indices)
        v = past_kv.value_cache[layer_idx].index_select(2, indices)
        # DynamicCache stores as list, we write directly
        new_cache.key_cache.append(k)
        new_cache.value_cache.append(v)
    return new_cache


def generate_with_budget(model, tokenizer, question, device,
                         budget_ratio=1.0, sink_size=4, window_size=64,
                         max_new_tokens=2048, evict_method="attention"):
    """Generate with limited KV-cache budget via streaming eviction."""
    prompt = f"Please solve this math problem step by step.\n\nQuestion: {question}\n\nLet me think step by step."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    # Prefill
    with torch.no_grad():
        out = model(**inputs, use_cache=True, output_attentions=True)
        past_kv = out.past_key_values

    # Init importance: [current_len]
    current_len = prompt_len
    importance = np.zeros(current_len, dtype=np.float64)
    for layer_attn in out.attentions:
        s = layer_attn[0].float().mean(dim=0).sum(dim=0).cpu().numpy()
        if not np.isnan(s).any():
            importance[:len(s)] += s

    # position_map[cache_idx] = original_position
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

        # Extend importance array
        if current_len > len(importance):
            importance = np.concatenate([importance, np.zeros(current_len - len(importance))])

        # Update importance from this step
        for layer_attn in out.attentions:
            s = layer_attn[0].float().mean(dim=0).squeeze(0).cpu().numpy()  # [cache_len]
            if np.isnan(s).any():
                continue
            for ci, orig_pos in enumerate(position_map):
                if ci < len(s):
                    importance[orig_pos] += s[ci]

        # Eviction check
        cache_len = past_kv.get_seq_length()
        n_reasoning = len(generated_ids)
        budget = prompt_len + max(int(n_reasoning * budget_ratio), sink_size + window_size)

        if cache_len > budget and budget_ratio < 1.0:
            keep = set()
            # Always keep prompt
            for i in range(min(prompt_len, cache_len)):
                keep.add(i)
            # Keep sinks (first few reasoning tokens)
            for i in range(prompt_len, min(prompt_len + sink_size, cache_len)):
                keep.add(i)
            # Keep recent window
            for i in range(max(0, cache_len - window_size), cache_len):
                keep.add(i)

            # How many more to keep
            n_more = max(0, budget - len(keep))
            candidates = [i for i in range(cache_len) if i not in keep]

            if candidates and n_more > 0:
                if evict_method == "attention":
                    cand_imp = np.array([importance[position_map[c]] for c in candidates])
                    top_idx = np.argsort(cand_imp)[-n_more:]
                    for idx in top_idx:
                        keep.add(candidates[idx])
                else:  # random
                    chosen = np.random.choice(len(candidates), min(n_more, len(candidates)), replace=False)
                    for idx in chosen:
                        keep.add(candidates[idx])

            keep_sorted = sorted(keep)
            past_kv = evict_kv_cache(past_kv, keep_sorted)
            position_map = [position_map[i] for i in keep_sorted]

        next_token = out.logits[:, -1:, :].argmax(dim=-1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results/week1_streaming")
    parser.add_argument("--budget_ratios", type=float, nargs="+", default=[1.0, 0.7, 0.5, 0.3])
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=64)
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

    all_results = {}

    for budget in args.budget_ratios:
        methods = ["attention", "random"] if budget < 1.0 else ["attention"]
        for method in methods:
            label = f"budget={budget:.0%}" + (f" ({method})" if budget < 1.0 else " (full)")
            print(f"\n=== {label} ===")
            correct = 0
            tested = 0

            for sample in tqdm(samples, desc=label):
                try:
                    text = generate_with_budget(
                        model, tokenizer, sample["question"], args.device,
                        budget_ratio=budget, sink_size=args.sink_size,
                        window_size=args.window_size,
                        max_new_tokens=args.max_new_tokens,
                        evict_method=method,
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
            all_results[label] = acc
            print(f"  Accuracy: {acc:.1%} ({correct}/{tested})")

    # Plot
    print("\n" + "=" * 60)
    print("STREAMING EVICTION RESULTS")
    print("=" * 60)

    attn_data = []
    rand_data = []
    for label, acc in all_results.items():
        budget_val = float(label.split("=")[1].split("%")[0]) / 100
        print(f"  {label}: {acc:.1%}")
        if "random" in label:
            rand_data.append((budget_val, acc))
        else:
            attn_data.append((budget_val, acc))

    plt.figure(figsize=(7, 4))
    if attn_data:
        attn_data.sort()
        plt.plot([d[0]*100 for d in attn_data], [d[1] for d in attn_data],
                 marker="o", linewidth=2, label="Attention-based", markersize=8)
    if rand_data:
        rand_data.sort()
        plt.plot([d[0]*100 for d in rand_data], [d[1] for d in rand_data],
                 marker="s", linewidth=2, label="Random", linestyle="--", markersize=8)
    plt.xlabel("KV-cache Budget (%)")
    plt.ylabel("Accuracy")
    plt.title("Streaming Eviction: Accuracy vs KV-cache Budget")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "streaming_eviction.png"), dpi=150)
    plt.close()

    summary = {
        "model": args.model, "n_samples": args.n_samples,
        "sink_size": args.sink_size, "window_size": args.window_size,
        "results": {k: float(v) for k, v in all_results.items()},
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
