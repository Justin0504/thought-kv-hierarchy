"""Oracle masking experiment (v2): proper evaluation.

Correct approach:
1. Generate full reasoning chain normally (no masking)
2. Get the complete KV-cache
3. Mask bottom-K% importance positions in the KV-cache
4. From the masked KV-cache, generate ONLY the final answer
5. Check if the answer is still correct

This simulates what would happen in production: the model reasons freely,
but we offload/discard low-importance KV entries during decoding.

Usage:
    python scripts/run_oracle_masking.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --n_samples 50 \
        --device cuda:0 \
        --output_dir results/week1_masking_v2
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


def run_sample(model, tokenizer, question, device, max_reasoning=2048, max_answer=128):
    """Generate full reasoning, then test masked answer extraction.

    Returns:
        full_text: complete generation
        token_importance: per-position attention importance [seq_len]
        prompt_len: number of prompt tokens
        full_ids: all token ids [1, seq_len]
    """
    prompt = (
        "Please solve this math problem step by step. "
        "After your reasoning, write your final numeric answer after ####.\n\n"
        f"Question: {question}\n\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    # Step 1: generate full reasoning chain with attention tracking
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_reasoning,
            do_sample=False,
            temperature=1.0,
            output_attentions=True,
            return_dict_in_generate=True,
        )

    full_ids = outputs.sequences  # [1, total_len]
    total_len = full_ids.shape[1]
    full_text = tokenizer.decode(full_ids[0][prompt_len:], skip_special_tokens=True)

    # Step 2: compute importance from attentions
    # For each decoding step, track how much each KV position was attended
    importance = np.zeros(total_len, dtype=np.float64)
    n_valid = 0

    for step_attns in outputs.attentions:
        layer_scores = []
        for layer_attn in step_attns:
            # [1, heads, q_len, kv_len] → last query, avg heads → [kv_len]
            s = layer_attn[0].float().mean(dim=0)[-1].cpu().numpy()
            if not np.isnan(s).any():
                layer_scores.append(s)
        if not layer_scores:
            continue
        avg = np.mean(layer_scores, axis=0)
        importance[:len(avg)] += avg
        n_valid += 1

    if n_valid > 0:
        importance /= n_valid

    return full_text, importance, prompt_len, full_ids


def test_masked_continuation(model, tokenizer, full_ids, importance, prompt_len,
                              mask_ratio, device, max_answer=256):
    """From full KV-cache with masking, generate a short answer continuation.

    Instead of re-generating everything, we:
    1. Do a forward pass on full_ids to build KV-cache
    2. Apply attention mask to block low-importance positions
    3. Generate a short continuation to extract the answer
    """
    seq_len = full_ids.shape[1]

    # Build mask: keep prompt, mask bottom-K% of reasoning tokens
    reasoning_importance = importance[prompt_len:seq_len]
    n_reasoning = len(reasoning_importance)
    n_mask = int(n_reasoning * mask_ratio)

    attn_mask = torch.ones(1, seq_len, device=device, dtype=torch.long)
    if n_mask > 0 and n_reasoning > 0:
        mask_indices = np.argsort(reasoning_importance)[:n_mask]
        for idx in mask_indices:
            attn_mask[0, prompt_len + idx] = 0

    # Forward pass to build KV-cache with masking
    with torch.no_grad():
        out = model(input_ids=full_ids, attention_mask=attn_mask, use_cache=True)
        past_kv = out.past_key_values

    # Generate short continuation
    answer_prompt = tokenizer.encode("\n\n#### ", add_special_tokens=False, return_tensors="pt").to(device)
    cur_ids = answer_prompt
    extended_mask = torch.cat([attn_mask, torch.ones(1, cur_ids.shape[1], device=device, dtype=torch.long)], dim=1)

    with torch.no_grad():
        out2 = model(input_ids=cur_ids, past_key_values=past_kv, attention_mask=extended_mask, use_cache=True)
        past_kv = out2.past_key_values

    generated = []
    for _ in range(max_answer):
        next_token = out2.logits[:, -1, :].argmax(dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break
        # Stop at newline (answer should be short)
        token_str = tokenizer.decode(next_token.item())
        if "\n" in token_str and len(generated) > 0:
            break
        generated.append(next_token.item())

        extended_mask = torch.cat([extended_mask, torch.ones(1, 1, device=device, dtype=torch.long)], dim=1)
        with torch.no_grad():
            out2 = model(
                input_ids=next_token.unsqueeze(0),
                past_key_values=past_kv,
                attention_mask=extended_mask,
                use_cache=True,
            )
            past_kv = out2.past_key_values

    answer_text = "#### " + tokenizer.decode(generated, skip_special_tokens=True)
    return answer_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results/week1_masking_v2")
    parser.add_argument("--mask_ratios", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--max_new_tokens", type=int, default=2048)
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

    samples = load_gsm8k(n_samples=args.n_samples)
    print(f"Loaded {len(samples)} GSM8K samples")

    # Phase 1: generate all reasoning chains + importance
    print("\n=== Phase 1: Generate reasoning chains ===")
    cache = []  # store (full_text, importance, prompt_len, full_ids) per sample
    baseline_correct = 0

    for sample in tqdm(samples, desc="Generating"):
        try:
            full_text, importance, prompt_len, full_ids = run_sample(
                model, tokenizer, sample["question"], args.device, args.max_new_tokens
            )
            pred = extract_answer(full_text)
            correct = check_answer(pred, sample["answer"])
            baseline_correct += correct
            cache.append({
                "full_text": full_text,
                "importance": importance,
                "prompt_len": prompt_len,
                "full_ids": full_ids,
                "correct": correct,
                "pred": pred,
            })
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM on sample {sample['id']}, skipping")
            torch.cuda.empty_cache()
            cache.append(None)

    valid = [c for c in cache if c is not None]
    baseline_acc = sum(c["correct"] for c in valid) / len(valid)
    print(f"\nBaseline accuracy: {baseline_acc:.1%} ({sum(c['correct'] for c in valid)}/{len(valid)})")

    # Phase 2: test masking at different ratios
    print("\n=== Phase 2: Oracle masking ===")
    results = {0.0: baseline_acc}

    for ratio in args.mask_ratios:
        if ratio == 0.0:
            continue
        print(f"\nMask ratio: {ratio:.0%}")
        correct = 0
        tested = 0

        for i, sample in enumerate(tqdm(samples, desc=f"Mask {ratio:.0%}")):
            if cache[i] is None:
                continue
            c = cache[i]
            try:
                answer_text = test_masked_continuation(
                    model, tokenizer, c["full_ids"], c["importance"],
                    c["prompt_len"], ratio, args.device
                )
                pred = extract_answer(answer_text)
                if check_answer(pred, sample["answer"]):
                    correct += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
            tested += 1

        acc = correct / max(1, tested)
        results[ratio] = acc
        drop = baseline_acc - acc
        print(f"  Accuracy: {acc:.1%} (drop: {drop:+.1%})")

    # Plot
    ratios = sorted(results.keys())
    accs = [results[r] for r in ratios]

    plt.figure(figsize=(7, 4))
    plt.plot([r * 100 for r in ratios], accs, marker="o", linewidth=2, markersize=8)
    plt.xlabel("KV-cache Masked (%)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Oracle Masking: Accuracy vs KV-cache Reduction\n(mask applied to existing reasoning, not re-generation)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    for r, a in zip(ratios, accs):
        plt.annotate(f"{a:.0%}", (r * 100, a), textcoords="offset points", xytext=(0, 10), ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "masking_accuracy_v2.png"), dpi=150)
    plt.close()

    # Summary
    print("\n" + "=" * 60)
    print("ORACLE MASKING RESULTS")
    print("=" * 60)
    for r in ratios:
        a = results[r]
        print(f"  Mask {r:>3.0%} → accuracy {a:.1%} (drop {baseline_acc - a:+.1%})")

    safe_50 = results.get(0.5, 0) >= baseline_acc - 0.05
    print(f"\nSafe to mask 50%: {'YES' if safe_50 else 'NO'}")

    summary = {
        "model": args.model,
        "n_samples": args.n_samples,
        "baseline_accuracy": float(baseline_acc),
        "mask_results": {str(k): float(v) for k, v in results.items()},
        "safe_50pct": bool(safe_50),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
