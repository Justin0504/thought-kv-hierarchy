"""Supplementary Experiment: GPU HBM Memory Measurement

Measures actual GPU memory usage (torch.cuda.max_memory_allocated)
for each hierarchy configuration, providing hard evidence for
"50-70% HBM reduction" claims.

Runs 10 samples per config (enough for stable memory measurement).
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
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.eval.gsm8k import load_gsm8k, extract_answer, check_answer
from src.system.hierarchy_offloader import HierarchyOffloader, get_n_layers, get_kv


def evict_kv_cache(past_kv, keep_indices, device):
    indices = torch.tensor(keep_indices, device=device, dtype=torch.long)
    new_cache = DynamicCache()
    for layer_idx in range(get_n_layers(past_kv)):
        k, v = get_kv(past_kv, layer_idx)
        new_cache.update(k.index_select(2, indices), v.index_select(2, indices), layer_idx)
    return new_cache


def measure_full(model, tokenizer, question, device, max_new_tokens=2048):
    prompt = f"Please solve this math problem step by step.\n\nQuestion: {question}\n\nLet me think step by step."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    torch.cuda.reset_peak_memory_stats(device)
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

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    pred = extract_answer(text)
    del past_kv; torch.cuda.empty_cache(); gc.collect()
    return peak_mem, len(generated_ids), pred


def measure_eviction(model, tokenizer, question, device, hbm_ratio=0.5,
                     sink_size=4, window_size=128, max_new_tokens=2048):
    prompt = f"Please solve this math problem step by step.\n\nQuestion: {question}\n\nLet me think step by step."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats(device)
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

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    pred = extract_answer(text)
    del past_kv; torch.cuda.empty_cache(); gc.collect()
    return peak_mem, len(generated_ids), pred


def measure_hierarchy(model, tokenizer, question, device,
                      hbm_ratio=0.5, evict_ratio=0.1,
                      sink_size=4, window_size=128,
                      manage_interval=64, max_new_tokens=2048):
    prompt = f"Please solve this math problem step by step.\n\nQuestion: {question}\n\nLet me think step by step."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    offloader = HierarchyOffloader(
        n_layers=28, n_heads=28, head_dim=128, device=device,
        hbm_ratio=hbm_ratio, evict_ratio=evict_ratio,
        sink_size=sink_size, window_size=window_size,
        manage_interval=manage_interval,
    )

    torch.cuda.reset_peak_memory_stats(device)
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

        if offloader.cpu_positions:
            full_cache, _ = offloader.prefetch_and_build_full_cache(past_kv)
        else:
            full_cache = past_kv

        with torch.no_grad():
            out = model(next_token, past_key_values=full_cache,
                        use_cache=True, output_attentions=True)

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

        past_kv, position_map = offloader.manage_cache(
            past_kv, importance, position_map, prompt_len, step
        )

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    pred = extract_answer(text)
    del past_kv; torch.cuda.empty_cache(); gc.collect()
    return peak_mem, len(generated_ids), pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results/supplementary")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        trust_remote_code=True, attn_implementation="eager",
    ).to(args.device)
    model.eval()

    # Model weight memory (constant across configs)
    model_mem = torch.cuda.memory_allocated(args.device) / 1024**3
    print(f"Model weights: {model_mem:.2f} GB")

    samples = load_gsm8k(n_samples=args.n_samples)
    print(f"Loaded {len(samples)} samples\n")

    configs = [
        ("Full (100% GPU)", "full", {}),
        ("Eviction (50% HBM)", "eviction", {"hbm_ratio": 0.5}),
        ("Eviction (30% HBM)", "eviction", {"hbm_ratio": 0.3}),
        ("Hierarchy (50% HBM, 10% evict)", "hierarchy", {"hbm_ratio": 0.5, "evict_ratio": 0.1}),
        ("Hierarchy (30% HBM, 10% evict)", "hierarchy", {"hbm_ratio": 0.3, "evict_ratio": 0.1}),
        ("Hierarchy (50% HBM, 5% evict)", "hierarchy", {"hbm_ratio": 0.5, "evict_ratio": 0.05}),
        ("Hierarchy (30% HBM, 5% evict)", "hierarchy", {"hbm_ratio": 0.3, "evict_ratio": 0.05}),
    ]

    all_results = {}
    for label, mode, params in configs:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        peak_mems = []
        n_tokens_list = []
        correct = 0
        tested = 0

        for i, sample in enumerate(tqdm(samples, desc=label[:35])):
            try:
                if mode == "full":
                    peak, n_tok, pred = measure_full(
                        model, tokenizer, sample["question"], args.device)
                elif mode == "eviction":
                    peak, n_tok, pred = measure_eviction(
                        model, tokenizer, sample["question"], args.device,
                        hbm_ratio=params["hbm_ratio"])
                elif mode == "hierarchy":
                    peak, n_tok, pred = measure_hierarchy(
                        model, tokenizer, sample["question"], args.device,
                        hbm_ratio=params["hbm_ratio"],
                        evict_ratio=params.get("evict_ratio", 0.1))

                peak_mems.append(peak)
                n_tokens_list.append(n_tok)
                if check_answer(pred, sample["answer"]):
                    correct += 1
            except torch.cuda.OutOfMemoryError:
                print(f"  Sample {i}: OOM")
                torch.cuda.empty_cache(); gc.collect()
            except Exception as e:
                print(f"  Sample {i}: Error: {e}")
                torch.cuda.empty_cache(); gc.collect()
            tested += 1

        kv_mem_mean = np.mean(peak_mems) - model_mem if peak_mems else 0
        result = {
            "accuracy": correct / max(1, tested),
            "correct": correct,
            "tested": tested,
            "model_mem_gb": round(model_mem, 2),
            "peak_mem_gb": {"mean": round(np.mean(peak_mems), 2), "std": round(np.std(peak_mems), 2)} if peak_mems else {},
            "kv_cache_mem_gb": round(kv_mem_mean, 2),
            "avg_tokens": round(np.mean(n_tokens_list), 1) if n_tokens_list else 0,
            "hbm_reduction_pct": 0,
        }
        all_results[label] = result

        print(f"  Peak GPU mem: {np.mean(peak_mems):.2f} ± {np.std(peak_mems):.2f} GB")
        print(f"  KV cache mem: {kv_mem_mean:.2f} GB")
        print(f"  Accuracy: {correct}/{tested}")

    # Compute HBM reduction relative to full
    if "Full (100% GPU)" in all_results:
        full_peak = all_results["Full (100% GPU)"]["peak_mem_gb"].get("mean", 0)
        for label, res in all_results.items():
            if label != "Full (100% GPU)" and res["peak_mem_gb"]:
                reduction = (full_peak - res["peak_mem_gb"]["mean"]) / full_peak * 100
                res["hbm_reduction_pct"] = round(reduction, 1)

    # Save
    with open(os.path.join(args.output_dir, "memory_measurement.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print("MEMORY MEASUREMENT SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<40s} {'Peak(GB)':>9s} {'KV(GB)':>8s} {'Reduction':>10s} {'Acc':>6s}")
    print("-" * 80)
    for label, res in all_results.items():
        peak = res["peak_mem_gb"].get("mean", 0) if res["peak_mem_gb"] else 0
        kv = res["kv_cache_mem_gb"]
        red = f"{res['hbm_reduction_pct']:.1f}%" if res["hbm_reduction_pct"] else "---"
        acc = f"{res['accuracy']:.0%}"
        print(f"  {label:<38s} {peak:>8.2f} {kv:>8.2f} {red:>10s} {acc:>6s}")

    print(f"\nResults saved to {args.output_dir}/memory_measurement.json")


if __name__ == "__main__":
    main()
