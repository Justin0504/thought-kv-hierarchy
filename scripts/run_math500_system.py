"""Supplementary Experiment: System Prototype on MATH-500 + Larger GSM8K (n=200)

Extends B's system prototype experiments to:
1. MATH-500 benchmark (matching A's algorithm experiments)
2. GSM8K with n=200 (narrower confidence intervals)

Uses fp16 model (matching A's algorithm setup) for consistency.
"""

import argparse
import json
import os
import sys
import gc
import time
import re

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.eval.gsm8k import load_gsm8k, extract_answer, check_answer
from src.system.hierarchy_offloader import HierarchyOffloader, get_n_layers, get_kv


# ── MATH-500 loader ──

def load_math500(n_samples=50):
    from datasets import load_dataset
    # Try multiple dataset names (HF naming changed over time)
    for name in ["HuggingFaceH4/MATH-500", "lighteval/MATH", "hendrycks/competition_math"]:
        try:
            ds = load_dataset(name, split="test")
            break
        except Exception:
            continue
    else:
        # Fallback: try MATH Hard subset
        try:
            ds = load_dataset("lighteval/MATH-Hard", split="test")
        except Exception:
            raise RuntimeError("Cannot load MATH dataset. Tried: HuggingFaceH4/MATH-500, lighteval/MATH, hendrycks/competition_math, lighteval/MATH-Hard")

    samples = []
    for item in ds:
        if len(samples) >= n_samples:
            break
        # Handle different field names across dataset versions
        question = item.get("problem", item.get("question", ""))
        answer = item.get("solution", item.get("answer", ""))
        # Extract boxed answer if present
        boxed = re.findall(r'\\boxed\{([^}]+)\}', str(answer))
        final_answer = boxed[-1] if boxed else str(answer).strip()
        if question and final_answer:
            samples.append({"question": question, "answer": final_answer})
    return samples


def extract_math_answer(text):
    # Try boxed first
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()
    # Try "answer is X" pattern
    patterns = [
        r'(?:final answer|answer) (?:is|=)\s*[\\$]*([^\n\\.,$]+)',
        r'(?:=)\s*([0-9]+(?:\.[0-9]+)?)\s*$',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip().strip('$\\')
    return ""


def check_math_answer(pred, gold):
    if not pred or not gold:
        return False
    pred_clean = pred.strip().lower().replace(" ", "")
    gold_clean = gold.strip().lower().replace(" ", "")
    if pred_clean == gold_clean:
        return True
    try:
        if abs(float(pred_clean) - float(gold_clean)) < 1e-6:
            return True
    except:
        pass
    return False


# ── KV cache eviction ──

def evict_kv_cache(past_kv, keep_indices, device):
    indices = torch.tensor(keep_indices, device=device, dtype=torch.long)
    new_cache = DynamicCache()
    for layer_idx in range(get_n_layers(past_kv)):
        k, v = get_kv(past_kv, layer_idx)
        new_cache.update(k.index_select(2, indices), v.index_select(2, indices), layer_idx)
    return new_cache


# ── Generation functions ──

def generate_full(model, tokenizer, question, device, max_new_tokens=2048):
    prompt = f"Please solve this math problem step by step.\n\nQuestion: {question}\n\nLet me think step by step."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

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

    elapsed = time.perf_counter() - t0
    del past_kv; torch.cuda.empty_cache(); gc.collect()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, len(generated_ids), elapsed


def generate_eviction(model, tokenizer, question, device,
                      hbm_ratio=0.5, sink_size=4, window_size=128,
                      max_new_tokens=2048):
    prompt = f"Please solve this math problem step by step.\n\nQuestion: {question}\n\nLet me think step by step."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

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

        if cache_len > budget:
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

    elapsed = time.perf_counter() - t0
    del past_kv; torch.cuda.empty_cache(); gc.collect()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, len(generated_ids), elapsed


def generate_hierarchy(model, tokenizer, question, device,
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

    elapsed = time.perf_counter() - t0
    del past_kv; torch.cuda.empty_cache(); gc.collect()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, len(generated_ids), elapsed, offloader.get_summary()


def run_benchmark(model, tokenizer, samples, configs, device, benchmark_name,
                  answer_extractor, answer_checker):
    """Run all configs on a set of samples."""
    results = {}
    for label, mode, params in configs:
        print(f"\n--- {label} ---")
        correct = 0
        tested = 0
        total_tokens = 0
        total_time = 0.0

        for i, sample in enumerate(tqdm(samples, desc=f"{benchmark_name} {label[:30]}")):
            try:
                if mode == "full":
                    text, n_tok, elapsed = generate_full(model, tokenizer, sample["question"], device)
                elif mode == "eviction":
                    text, n_tok, elapsed = generate_eviction(
                        model, tokenizer, sample["question"], device,
                        hbm_ratio=params["hbm_ratio"])
                elif mode == "hierarchy":
                    text, n_tok, elapsed, _ = generate_hierarchy(
                        model, tokenizer, sample["question"], device,
                        hbm_ratio=params["hbm_ratio"],
                        evict_ratio=params.get("evict_ratio", 0.1))

                pred = answer_extractor(text)
                if answer_checker(pred, sample["answer"]):
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
        results[label] = {
            "accuracy": acc,
            "correct": correct,
            "tested": tested,
            "total_tokens": total_tokens,
            "tokens_per_sec": round(total_tokens / max(0.001, total_time), 2),
        }
        print(f"  {label}: {acc:.1%} ({correct}/{tested}), {results[label]['tokens_per_sec']:.1f} tok/s")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results/supplementary")
    parser.add_argument("--gsm8k_n", type=int, default=200)
    parser.add_argument("--math_n", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        trust_remote_code=True, attn_implementation="eager",
    ).to(args.device)
    model.eval()

    configs = [
        ("Full (100% GPU)", "full", {}),
        ("Eviction (50% HBM)", "eviction", {"hbm_ratio": 0.5}),
        ("Hierarchy (50% HBM, 10% evict)", "hierarchy", {"hbm_ratio": 0.5, "evict_ratio": 0.1}),
        ("Hierarchy (50% HBM, 5% evict)", "hierarchy", {"hbm_ratio": 0.5, "evict_ratio": 0.05}),
        ("Hierarchy (30% HBM, 10% evict)", "hierarchy", {"hbm_ratio": 0.3, "evict_ratio": 0.1}),
    ]

    all_results = {}

    # ═══ MATH-500 ═══
    print("\n" + "=" * 70)
    print(f"MATH-500 System Prototype (n={args.math_n})")
    print("=" * 70)
    math_samples = load_math500(n_samples=args.math_n)
    print(f"Loaded {len(math_samples)} MATH-500 samples")
    all_results["math500"] = run_benchmark(
        model, tokenizer, math_samples, configs, args.device,
        "MATH", extract_math_answer, check_math_answer
    )

    # ═══ GSM8K (n=200) ═══
    print("\n" + "=" * 70)
    print(f"GSM8K System Prototype (n={args.gsm8k_n})")
    print("=" * 70)
    gsm_samples = load_gsm8k(n_samples=args.gsm8k_n)
    print(f"Loaded {len(gsm_samples)} GSM8K samples")
    all_results["gsm8k_200"] = run_benchmark(
        model, tokenizer, gsm_samples, configs, args.device,
        "GSM8K", extract_answer, check_answer
    )

    # Save
    with open(os.path.join(args.output_dir, "math500_gsm8k200_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for bench, res in all_results.items():
        print(f"\n{bench}:")
        for label, r in res.items():
            print(f"  {label:<40s} {r['accuracy']:>6.1%} ({r['correct']}/{r['tested']})")

    print(f"\nAll saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
