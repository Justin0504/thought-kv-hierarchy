"""Task 2: Gradient-based Importance Scoring

Compares attention-based vs gradient-based importance scoring:
1. Spearman rank correlation between the two methods
2. Streaming eviction accuracy using each scoring method

Usage:
    cd ~/thought-kv-hierarchy
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ~/thought-kv-hierarchy/venv/bin/python3 scripts/benchmark_gradient_importance.py \
        --n_samples 50 --device cuda:0
"""

import argparse
import json
import os
import sys
import gc

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.eval.gsm8k import load_gsm8k, extract_answer, check_answer
from src.system.gradient_scorer import (
    compute_gradient_importance, get_n_layers, get_kv, kv_cache_to_cpu
)


# ── KV-cache operations ─────────────────────────────────────────────

def evict_kv_cache(past_kv, keep_indices):
    device = get_kv(past_kv, 0)[0].device
    indices = torch.tensor(keep_indices, device=device, dtype=torch.long)
    new_cache = DynamicCache()
    for layer_idx in range(get_n_layers(past_kv)):
        k, v = get_kv(past_kv, layer_idx)
        new_cache.update(k.index_select(2, indices), v.index_select(2, indices), layer_idx)
    return new_cache


# ── Part 1: Correlation Analysis ─────────────────────────────────────

def generate_and_compare_importance(model, tokenizer, question, device,
                                     sink_size=4, window_size=128,
                                     max_new_tokens=2048):
    """Generate full response, then compute both importance scores."""
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
    importance_attn = np.zeros(current_len, dtype=np.float64)
    for layer_attn in out.attentions:
        s = layer_attn[0].float().mean(dim=0).sum(dim=0).cpu().numpy()
        if not np.isnan(s).any():
            importance_attn[:len(s)] += s

    generated_ids = []
    next_token = out.logits[:, -1:, :].argmax(dim=-1)
    del out
    torch.cuda.empty_cache()

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
        if current_len > len(importance_attn):
            importance_attn = np.concatenate([
                importance_attn, np.zeros(current_len - len(importance_attn))
            ])

        for layer_attn in out.attentions:
            s = layer_attn[0].float().mean(dim=0).squeeze(0).cpu().numpy()
            if not np.isnan(s).any():
                for ci in range(min(len(s), current_len)):
                    importance_attn[ci] += s[ci]

        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        del out
        torch.cuda.empty_cache()

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    seq_len = past_kv.get_seq_length()
    importance_attn = importance_attn[:seq_len]

    # Move KV to CPU, free GPU, then compute gradient importance
    last_tok = generated_ids[-1] if generated_ids else next_token.item()
    kv_cpu = kv_cache_to_cpu(past_kv)
    del past_kv
    torch.cuda.empty_cache()
    gc.collect()

    importance_grad = compute_gradient_importance(model, kv_cpu, last_tok, device)
    del kv_cpu
    torch.cuda.empty_cache()
    gc.collect()

    # Correlations on reasoning tokens only
    start_idx = prompt_len + sink_size
    end_idx = max(start_idx, seq_len - window_size)

    result = {"n_tokens": seq_len, "n_reasoning": len(generated_ids)}

    if end_idx > start_idx + 5:
        attn_slice = importance_attn[start_idx:end_idx]
        grad_slice = importance_grad[start_idx:end_idx]
        spearman = stats.spearmanr(attn_slice, grad_slice)
        kendall = stats.kendalltau(attn_slice, grad_slice)
        result["spearman_rho"] = float(spearman.statistic)
        result["spearman_pvalue"] = float(spearman.pvalue)
        result["kendall_tau"] = float(kendall.statistic)
        result["kendall_pvalue"] = float(kendall.pvalue)
        result["n_compared"] = int(end_idx - start_idx)
    else:
        result["spearman_rho"] = None
        result["kendall_tau"] = None
        result["n_compared"] = 0

    return result, text


# ── Part 2: Streaming Eviction ───────────────────────────────────────

def generate_with_eviction(model, tokenizer, question, device,
                           scoring_method="attention", hbm_ratio=0.5,
                           sink_size=4, window_size=128,
                           max_new_tokens=2048, grad_interval=64):
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
    position_map = list(range(current_len))

    if scoring_method == "attention":
        for layer_attn in out.attentions:
            s = layer_attn[0].float().mean(dim=0).sum(dim=0).cpu().numpy()
            if not np.isnan(s).any():
                importance[:len(s)] += s

    generated_ids = []
    next_token = out.logits[:, -1:, :].argmax(dim=-1)
    del out
    torch.cuda.empty_cache()

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

        if scoring_method == "attention":
            for layer_attn in out.attentions:
                s = layer_attn[0].float().mean(dim=0).squeeze(0).cpu().numpy()
                if np.isnan(s).any():
                    continue
                for ci, orig_pos in enumerate(position_map):
                    if ci < len(s):
                        importance[orig_pos] += s[ci]

        next_token_val = out.logits[:, -1:, :].argmax(dim=-1)
        del out
        torch.cuda.empty_cache()

        # Eviction
        cache_len = past_kv.get_seq_length()
        n_reasoning = len(generated_ids)
        budget = prompt_len + max(int(n_reasoning * hbm_ratio), sink_size + window_size)

        if cache_len > budget and hbm_ratio < 1.0:
            if scoring_method == "gradient" and (step + 1) % grad_interval == 0:
                try:
                    kv_cpu = kv_cache_to_cpu(past_kv)
                    del past_kv
                    torch.cuda.empty_cache()
                    grad_imp = compute_gradient_importance(
                        model, kv_cpu, tok_id, device
                    )
                    # Rebuild cache on GPU
                    past_kv = DynamicCache()
                    for li, (kc, vc) in enumerate(kv_cpu):
                        past_kv.update(kc.to(device), vc.to(device), li)
                    del kv_cpu
                    torch.cuda.empty_cache()

                    for ci in range(min(len(grad_imp), len(position_map))):
                        importance[position_map[ci]] = grad_imp[ci]
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()

            cache_len = past_kv.get_seq_length()
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

        next_token = next_token_val

    del past_kv
    torch.cuda.empty_cache()
    gc.collect()
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results/system/task2")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--skip_eviction", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()

    samples = load_gsm8k(n_samples=args.n_samples)
    print(f"Loaded {len(samples)} samples\n")

    # ═══ Part 1: Correlation ═══
    print("=" * 60)
    print("Part 1: Attention vs Gradient Importance Correlation")
    print("=" * 60)

    correlation_results = []
    for i, sample in enumerate(tqdm(samples, desc="Correlation")):
        try:
            result, text = generate_and_compare_importance(
                model, tokenizer, sample["question"], args.device,
                sink_size=args.sink_size, window_size=args.window_size,
                max_new_tokens=args.max_new_tokens,
            )
            result["sample_idx"] = i
            pred = extract_answer(text)
            result["correct"] = check_answer(pred, sample["answer"])

            if result["spearman_rho"] is not None:
                print(f"  Sample {i}: Spearman={result['spearman_rho']:.3f}, "
                      f"Kendall={result['kendall_tau']:.3f}, "
                      f"tokens={result['n_tokens']}, correct={result['correct']}")
            correlation_results.append(result)

        except torch.cuda.OutOfMemoryError:
            print(f"  Sample {i}: OOM, skipping")
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"  Sample {i}: Error: {e}")
            import traceback; traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()

    valid = [r for r in correlation_results if r.get("spearman_rho") is not None]
    if valid:
        sp_vals = [r["spearman_rho"] for r in valid]
        kt_vals = [r["kendall_tau"] for r in valid]
        corr_summary = {
            "n_valid": len(valid), "n_total": len(correlation_results),
            "spearman_rho": {"mean": float(np.mean(sp_vals)), "std": float(np.std(sp_vals)),
                             "median": float(np.median(sp_vals)),
                             "min": float(np.min(sp_vals)), "max": float(np.max(sp_vals))},
            "kendall_tau": {"mean": float(np.mean(kt_vals)), "std": float(np.std(kt_vals)),
                            "median": float(np.median(kt_vals)),
                            "min": float(np.min(kt_vals)), "max": float(np.max(kt_vals))},
        }
        print(f"\n  Summary ({len(valid)} samples):")
        print(f"    Spearman: {np.mean(sp_vals):.3f} +/- {np.std(sp_vals):.3f}")
        print(f"    Kendall:  {np.mean(kt_vals):.3f} +/- {np.std(kt_vals):.3f}")
    else:
        corr_summary = {"n_valid": 0, "n_total": len(correlation_results)}

    with open(os.path.join(args.output_dir, "correlation_results.json"), "w") as f:
        json.dump({"summary": corr_summary, "per_sample": correlation_results}, f, indent=2)

    # ═══ Part 2: Eviction ═══
    eviction_results = {}
    if not args.skip_eviction:
        print("\n" + "=" * 60)
        print("Part 2: Streaming Eviction — Attention vs Gradient")
        print("=" * 60)

        budget_ratios = [0.9, 0.7, 0.5]
        for scoring in ["attention", "gradient"]:
            for ratio in budget_ratios:
                label = f"{scoring} (budget={ratio:.0%})"
                print(f"\n--- {label} ---")
                correct = 0
                tested = 0
                for i, sample in enumerate(tqdm(samples, desc=label[:35])):
                    try:
                        text = generate_with_eviction(
                            model, tokenizer, sample["question"], args.device,
                            scoring_method=scoring, hbm_ratio=ratio,
                            sink_size=args.sink_size, window_size=args.window_size,
                            max_new_tokens=args.max_new_tokens,
                        )
                        pred = extract_answer(text)
                        if check_answer(pred, sample["answer"]):
                            correct += 1
                    except torch.cuda.OutOfMemoryError:
                        print(f"  Sample {i}: OOM")
                        torch.cuda.empty_cache(); gc.collect()
                    except Exception as e:
                        print(f"  Sample {i}: Error: {e}")
                        torch.cuda.empty_cache(); gc.collect()
                    tested += 1
                acc = correct / max(1, tested)
                eviction_results[label] = {"accuracy": acc, "correct": correct, "tested": tested}
                print(f"  {label}: {acc:.1%} ({correct}/{tested})")

        with open(os.path.join(args.output_dir, "eviction_results.json"), "w") as f:
            json.dump(eviction_results, f, indent=2)

    # ═══ Plots ═══
    print("\n" + "=" * 60)
    print("Generating Figures")
    print("=" * 60)

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 13, "figure.dpi": 150, "savefig.bbox": "tight",
    })

    if valid:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1.hist(sp_vals, bins=20, color="#1565C0", alpha=0.8, edgecolor="white")
        ax1.axvline(np.mean(sp_vals), color="#D32F2F", linestyle="--", linewidth=2,
                    label=f"Mean = {np.mean(sp_vals):.3f}")
        ax1.set_xlabel("Spearman Rank Correlation"); ax1.set_ylabel("Count")
        ax1.set_title("Spearman rho: Attention vs Gradient"); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.hist(kt_vals, bins=20, color="#FF8F00", alpha=0.8, edgecolor="white")
        ax2.axvline(np.mean(kt_vals), color="#D32F2F", linestyle="--", linewidth=2,
                    label=f"Mean = {np.mean(kt_vals):.3f}")
        ax2.set_xlabel("Kendall Rank Correlation"); ax2.set_ylabel("Count")
        ax2.set_title("Kendall tau: Attention vs Gradient"); ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "fig_correlation_dist.png"), dpi=200)
        plt.savefig(os.path.join(args.output_dir, "fig_correlation_dist.pdf"))
        plt.close()
        print("  Saved fig_correlation_dist")

    if eviction_results:
        fig, ax = plt.subplots(figsize=(9, 5))
        budget_ratios_plot = [0.9, 0.7, 0.5]
        attn_accs, grad_accs, labels = [], [], []
        for ratio in budget_ratios_plot:
            ak = f"attention (budget={ratio:.0%})"; gk = f"gradient (budget={ratio:.0%})"
            if ak in eviction_results and gk in eviction_results:
                labels.append(f"{ratio:.0%}")
                attn_accs.append(eviction_results[ak]["accuracy"])
                grad_accs.append(eviction_results[gk]["accuracy"])
        if labels:
            x = np.arange(len(labels)); w = 0.35
            b1 = ax.bar(x - w/2, attn_accs, w, color="#1565C0", label="Attention-based", alpha=0.85)
            b2 = ax.bar(x + w/2, grad_accs, w, color="#D32F2F", label="Gradient-based", alpha=0.85)
            for bar in b1:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f"{bar.get_height():.0%}", ha="center", fontsize=10)
            for bar in b2:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f"{bar.get_height():.0%}", ha="center", fontsize=10)
            ax.set_xlabel("HBM Budget"); ax.set_ylabel("GSM8K Accuracy")
            ax.set_title("Streaming Eviction: Attention vs Gradient\n(DeepSeek-R1-7B, n=50)")
            ax.set_xticks(x); ax.set_xticklabels(labels)
            ax.legend(); ax.set_ylim(0, 1.0); ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "fig_eviction_comparison.png"), dpi=200)
        plt.savefig(os.path.join(args.output_dir, "fig_eviction_comparison.pdf"))
        plt.close()
        print("  Saved fig_eviction_comparison")

    # ═══ Summary ═══
    print("\n" + "=" * 60)
    print("TASK 2 RESULTS SUMMARY")
    print("=" * 60)
    if valid:
        print(f"\nCorrelation ({len(valid)} samples):")
        print(f"  Spearman = {corr_summary['spearman_rho']['mean']:.3f} +/- {corr_summary['spearman_rho']['std']:.3f}")
        print(f"  Kendall  = {corr_summary['kendall_tau']['mean']:.3f} +/- {corr_summary['kendall_tau']['std']:.3f}")
    if eviction_results:
        print(f"\nEviction Accuracy:")
        for label, res in eviction_results.items():
            print(f"  {label:35s}  {res['accuracy']:.1%} ({res['correct']}/{res['tested']})")
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
