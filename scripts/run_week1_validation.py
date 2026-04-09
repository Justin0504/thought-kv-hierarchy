"""Week 1 validation: verify the core hypothesis.

1. Run DeepSeek-R1-Distill on GSM8K, collect attention importance scores
2. Analyze the distribution — is it long-tailed?
3. Oracle masking: mask bottom-K% KV positions, measure accuracy drop

Usage:
    python scripts/run_week1_validation.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --n_samples 50 \
        --device cuda:0 \
        --output_dir results/week1
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
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.eval.gsm8k import load_gsm8k, extract_answer, check_answer
from src.scorer.attention_scorer import compute_importance_from_attentions


# ── helpers ──────────────────────────────────────────────────────────

def generate_with_profiling(model, tokenizer, question, device, max_new_tokens=2048):
    """Generate reasoning chain and extract attention importance."""
    prompt = (
        "Please solve this math problem step by step.\n\n"
        f"Question: {question}\n\nLet me think step by step."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            output_attentions=True,
            return_dict_in_generate=True,
        )

    importance = compute_importance_from_attentions(outputs.attentions, prompt_len)
    generated_ids = outputs.sequences[0][prompt_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return text, importance, prompt_len


def generate_with_masking(model, tokenizer, question, kv_mask, device, max_new_tokens=512):
    """Regenerate with certain KV positions masked via attention_mask.

    Simple approach: do a full forward pass but block attention to masked positions.
    """
    prompt = (
        "Please solve this math problem step by step.\n\n"
        f"Question: {question}\n\nLet me think step by step."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # We generate token-by-token with a custom attention mask
    input_ids = inputs["input_ids"]
    past_kv = None
    generated = []

    for step in range(max_new_tokens):
        with torch.no_grad():
            if past_kv is None:
                out = model(input_ids=input_ids, use_cache=True)
            else:
                out = model(input_ids=input_ids[:, -1:], past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values

        next_token = out.logits[:, -1, :].argmax(dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break
        generated.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    # Now we have full KV cache. Do one more forward with masking to get final answer.
    # Actually, simpler: re-run the full sequence with attention mask applied.
    full_ids = torch.cat(
        [inputs["input_ids"], torch.tensor([generated], device=device)], dim=1
    ) if generated else inputs["input_ids"]

    seq_len = full_ids.shape[1]
    attn_mask = torch.ones(1, seq_len, device=device, dtype=torch.long)

    # Apply the KV mask (block attention to masked positions)
    mask_len = min(len(kv_mask), seq_len)
    for i in range(mask_len):
        if not kv_mask[i]:
            attn_mask[0, i] = 0

    with torch.no_grad():
        out = model(input_ids=full_ids, attention_mask=attn_mask)

    # Generate a short continuation from the masked context
    continuation = []
    past = None
    cur_ids = full_ids
    for step in range(256):
        with torch.no_grad():
            if past is None:
                o = model(input_ids=cur_ids, attention_mask=attn_mask, use_cache=True)
            else:
                # extend attention mask by 1
                attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=device, dtype=torch.long)], dim=1)
                o = model(input_ids=cur_ids[:, -1:], past_key_values=past, attention_mask=attn_mask, use_cache=True)
            past = o.past_key_values
        nt = o.logits[:, -1, :].argmax(dim=-1)
        if nt.item() == tokenizer.eos_token_id:
            break
        continuation.append(nt.item())
        cur_ids = torch.cat([cur_ids, nt.unsqueeze(0)], dim=1)

    text = tokenizer.decode(generated + continuation, skip_special_tokens=True)
    return text


# ── plotting ─────────────────────────────────────────────────────────

def plot_importance_distribution(all_scores, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    flat = np.concatenate([s for s in all_scores if len(s) > 0])
    if len(flat) == 0:
        print("WARNING: no scores to plot")
        return

    axes[0].hist(flat, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Importance Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("(a) Token Importance Distribution")
    axes[0].set_yscale("log")

    sorted_flat = np.sort(flat)
    cdf = np.arange(1, len(sorted_flat) + 1) / len(sorted_flat)
    axes[1].plot(sorted_flat, cdf)
    axes[1].set_xlabel("Importance Score")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("(b) Cumulative Distribution")

    k_values = list(range(5, 105, 5))
    coverages = []
    for k in k_values:
        ratios = []
        for s in all_scores:
            if len(s) == 0:
                continue
            ss = np.sort(s)[::-1]
            n = max(1, int(len(ss) * k / 100))
            ratios.append(ss[:n].sum() / (ss.sum() + 1e-12))
        coverages.append(np.mean(ratios) if ratios else 0)

    axes[2].plot(k_values, coverages, marker="o", markersize=3)
    axes[2].axhline(y=0.8, color="red", linestyle="--", alpha=0.5, label="80% coverage")
    axes[2].set_xlabel("Top-K% Tokens")
    axes[2].set_ylabel("Fraction of Total Importance")
    axes[2].set_title("(c) Importance Concentration")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "importance_distribution.png"), dpi=150)
    plt.close()
    print("Saved importance_distribution.png")


def plot_masking_accuracy(mask_results, output_dir):
    ratios = sorted(mask_results.keys())
    accs = [mask_results[r] for r in ratios]

    plt.figure(figsize=(6, 4))
    plt.plot(ratios, accs, marker="o", linewidth=2)
    plt.xlabel("Fraction of KV-cache Masked")
    plt.ylabel("Accuracy")
    plt.title("Oracle Masking: Accuracy vs KV-cache Reduction")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "masking_accuracy.png"), dpi=150)
    plt.close()
    print("Saved masking_accuracy.png")


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results/week1")
    parser.add_argument("--mask_ratios", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7])
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--skip_masking", action="store_true", help="Only run profiling, skip masking experiment")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    print(f"Loading GSM8K (n={args.n_samples})")
    samples = load_gsm8k(n_samples=args.n_samples)

    # ── Phase 1: Profile ─────────────────────────────────────────
    print("\n=== Phase 1: Attention Importance Profiling ===")
    all_scores = []       # reasoning-token scores per sample
    all_prompt_lens = []
    baseline_correct = 0
    profiling_results = []

    for sample in tqdm(samples, desc="Profiling"):
        try:
            text, importance, prompt_len = generate_with_profiling(
                model, tokenizer, sample["question"], args.device, args.max_new_tokens
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM on sample {sample['id']}, skipping")
            torch.cuda.empty_cache()
            all_scores.append(np.array([]))
            all_prompt_lens.append(0)
            profiling_results.append({"id": sample["id"], "correct": False, "oom": True})
            continue

        pred = extract_answer(text)
        correct = check_answer(pred, sample["answer"])
        baseline_correct += correct

        # Reasoning token scores (after prompt)
        if len(importance.scores) > prompt_len:
            reasoning_scores = importance.scores[prompt_len:]
        else:
            reasoning_scores = importance.scores
        all_scores.append(reasoning_scores)
        all_prompt_lens.append(prompt_len)

        top20 = importance.top_k_ratio(20) if len(importance.scores) > 0 else 0.0

        profiling_results.append({
            "id": sample["id"],
            "correct": correct,
            "predicted": pred,
            "gold": sample["answer"],
            "n_reasoning_tokens": len(reasoning_scores),
            "n_total_tokens": len(importance.scores),
            "top20_coverage": float(top20),
            "generated_text": text[:500],
        })

    n_valid = sum(1 for s in all_scores if len(s) > 0)
    baseline_acc = baseline_correct / len(samples)
    top20_coverages = [r["top20_coverage"] for r in profiling_results if r.get("top20_coverage", 0) > 0]

    print(f"\nBaseline accuracy: {baseline_acc:.1%} ({baseline_correct}/{len(samples)})")
    print(f"Valid profiling samples: {n_valid}/{len(samples)}")
    if top20_coverages:
        print(f"Top-20% importance coverage: mean={np.mean(top20_coverages):.1%}, median={np.median(top20_coverages):.1%}")

    with open(os.path.join(args.output_dir, "profiling_results.json"), "w") as f:
        json.dump(profiling_results, f, indent=2, default=str)

    # Save raw scores for B's correlation analysis
    scores_export = []
    for i, s in enumerate(all_scores):
        scores_export.append({
            "sample_id": samples[i]["id"],
            "prompt_len": all_prompt_lens[i],
            "scores": s.tolist() if len(s) > 0 else [],
        })
    with open(os.path.join(args.output_dir, "attention_scores.json"), "w") as f:
        json.dump(scores_export, f)
    print("Saved attention_scores.json (for B's correlation analysis)")

    plot_importance_distribution(all_scores, args.output_dir)

    if args.skip_masking:
        print("\nSkipping masking experiment (--skip_masking)")
        return

    # ── Phase 2: Oracle masking ──────────────────────────────────
    print("\n=== Phase 2: Oracle Masking Experiment ===")
    mask_results = {0.0: baseline_acc}

    for ratio in args.mask_ratios:
        if ratio == 0.0:
            continue
        print(f"\nMask ratio: {ratio:.0%}")
        correct = 0
        n_tested = 0
        for i, sample in enumerate(tqdm(samples, desc=f"Mask {ratio:.0%}")):
            scores = all_scores[i]
            if len(scores) == 0:
                continue

            prompt_len = all_prompt_lens[i]
            total_len = prompt_len + len(scores)

            # Build mask: never mask prompt, mask bottom-K% of reasoning tokens
            kv_mask = np.ones(total_len, dtype=bool)
            n_mask = int(len(scores) * ratio)
            if n_mask > 0:
                threshold_indices = np.argsort(scores)[:n_mask]
                for idx in threshold_indices:
                    kv_mask[prompt_len + idx] = False

            try:
                text = generate_with_masking(
                    model, tokenizer, sample["question"], kv_mask, args.device
                )
                pred = extract_answer(text)
                if check_answer(pred, sample["answer"]):
                    correct += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
            n_tested += 1

        acc = correct / max(1, n_tested)
        mask_results[ratio] = acc
        print(f"Accuracy @ mask {ratio:.0%}: {acc:.1%} (drop: {baseline_acc - acc:+.1%})")

    plot_masking_accuracy(mask_results, args.output_dir)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("WEEK 1 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Samples: {args.n_samples}")
    print(f"Baseline accuracy: {baseline_acc:.1%}")
    if top20_coverages:
        print(f"Top-20% importance coverage: {np.mean(top20_coverages):.1%}")

    print(f"\nMasking results:")
    for ratio in sorted(mask_results.keys()):
        acc = mask_results[ratio]
        print(f"  Mask {ratio:>3.0%} -> accuracy {acc:.1%} (drop {baseline_acc - acc:+.1%})")

    hypothesis = np.mean(top20_coverages) > 0.6 if top20_coverages else False
    safe_50 = mask_results.get(0.5, 0) >= baseline_acc - 0.05
    print(f"\nLong-tail hypothesis: {'SUPPORTED' if hypothesis else 'NOT SUPPORTED'}")
    print(f"Safe to mask 50%:    {'YES' if safe_50 else 'NO'}")

    summary = {
        "model": args.model,
        "n_samples": args.n_samples,
        "baseline_accuracy": baseline_acc,
        "mean_top20_coverage": float(np.mean(top20_coverages)) if top20_coverages else None,
        "mask_results": {str(k): v for k, v in mask_results.items()},
        "hypothesis_supported": bool(hypothesis),
        "masking_safe_50pct": bool(safe_50),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
