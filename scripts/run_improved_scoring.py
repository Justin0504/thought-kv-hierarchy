"""Improved importance scoring experiments.

Compares different importance scoring methods:
1. Attention-only (our current method)
2. Attention + Value Norm (VATP-inspired)
3. Attention + Redundancy penalty (R-KV-inspired)
4. Combined: Attention + Value Norm + Redundancy

Tests each scoring method with hierarchy at 5% eviction on GSM8K (n=50).
"""
import sys, os, json, torch, re
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.eval.gsm8k import load_gsm8k, extract_answer, check_answer


def compute_importance_attention_only(attn_weights, past_kv, seq_len):
    """Original: cumulative attention score per position."""
    importance = torch.zeros(seq_len)
    for layer_attn in attn_weights:
        if layer_attn is None:
            continue
        a = layer_attn.float().squeeze(0).squeeze(1)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        importance += a.sum(dim=0).cpu()
    return importance.numpy()


def compute_importance_vatp(attn_weights, past_kv, seq_len):
    """VATP-inspired: attention * value_norm.

    Insight from EMNLP 2024: attention sinks have high attn but low value norms.
    Combining both gives better importance estimation.
    """
    attn_importance = torch.zeros(seq_len)
    value_norms = torch.zeros(seq_len)

    for layer_idx, layer_attn in enumerate(attn_weights):
        if layer_attn is None:
            continue
        a = layer_attn.float().squeeze(0).squeeze(1)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        attn_importance += a.sum(dim=0).cpu()

        # Value norm per position
        v = past_kv.value_cache[layer_idx]  # [1, heads, seq, dim]
        v_norm = v.float().norm(dim=-1).mean(dim=(0, 1)).cpu()  # [seq]
        value_norms += v_norm

    # Normalize both to [0, 1]
    attn_norm = attn_importance / (attn_importance.max() + 1e-8)
    val_norm = value_norms / (value_norms.max() + 1e-8)

    # Combined score: attention * value_norm
    combined = (attn_norm * val_norm).numpy()
    return combined


def compute_importance_redundancy(attn_weights, past_kv, seq_len):
    """R-KV-inspired: attention importance - redundancy penalty.

    Penalizes tokens that are similar to their neighbors (redundant).
    """
    attn_importance = torch.zeros(seq_len)
    for layer_attn in attn_weights:
        if layer_attn is None:
            continue
        a = layer_attn.float().squeeze(0).squeeze(1)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        attn_importance += a.sum(dim=0).cpu()

    # Compute redundancy via key cosine similarity with neighbors
    redundancy = torch.zeros(seq_len)
    for layer_idx in range(len(past_kv.key_cache)):
        k = past_kv.key_cache[layer_idx].float().squeeze(0)  # [heads, seq, dim]
        k_mean = k.mean(dim=0)  # [seq, dim]
        k_norm = torch.nn.functional.normalize(k_mean, dim=-1)

        # Cosine sim with neighbors (window=3)
        for i in range(1, seq_len):
            sim = torch.dot(k_norm[i], k_norm[i-1]).cpu()
            redundancy[i] += max(0, sim.item())
        if seq_len > 2:
            for i in range(seq_len - 1):
                sim = torch.dot(k_norm[i], k_norm[i+1]).cpu()
                redundancy[i] += max(0, sim.item())

    # Normalize
    attn_norm = attn_importance / (attn_importance.max() + 1e-8)
    red_norm = redundancy / (redundancy.max() + 1e-8)

    # R-KV formula: lambda * importance - (1-lambda) * redundancy
    # lambda=0.3 (importance weighted more than redundancy penalty)
    lam = 0.3
    combined = (lam * attn_norm - (1 - lam) * red_norm).numpy()
    return combined


def compute_importance_combined(attn_weights, past_kv, seq_len):
    """Combined: attention * value_norm - redundancy penalty."""
    attn_importance = torch.zeros(seq_len)
    value_norms = torch.zeros(seq_len)

    for layer_idx, layer_attn in enumerate(attn_weights):
        if layer_attn is None:
            continue
        a = layer_attn.float().squeeze(0).squeeze(1)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        attn_importance += a.sum(dim=0).cpu()

        v = past_kv.value_cache[layer_idx]
        v_norm = v.float().norm(dim=-1).mean(dim=(0, 1)).cpu()
        value_norms += v_norm

    # Redundancy
    redundancy = torch.zeros(seq_len)
    for layer_idx in range(min(4, len(past_kv.key_cache))):  # Sample 4 layers for speed
        k = past_kv.key_cache[layer_idx].float().squeeze(0)
        k_mean = k.mean(dim=0)
        k_norm = torch.nn.functional.normalize(k_mean, dim=-1)
        for i in range(1, seq_len):
            sim = torch.dot(k_norm[i], k_norm[i-1]).cpu()
            redundancy[i] += max(0, sim.item())

    attn_n = attn_importance / (attn_importance.max() + 1e-8)
    val_n = value_norms / (value_norms.max() + 1e-8)
    red_n = redundancy / (redundancy.max() + 1e-8)

    combined = (attn_n * val_n - 0.3 * red_n).numpy()
    return combined


SCORING_METHODS = {
    "attention_only": compute_importance_attention_only,
    "vatp": compute_importance_vatp,
    "redundancy": compute_importance_redundancy,
    "combined": compute_importance_combined,
}


def generate_with_scoring(model, tokenizer, question, device,
                          scoring_fn, evict_ratio=0.05,
                          sink_size=4, window_size=128,
                          max_new_tokens=2048):
    """Generate with hierarchy using a specific importance scoring function."""
    prompt = f"Please solve this math problem step by step.\n\nQuestion: {question}\n\nLet me think step by step."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    generated_ids = input_ids.clone()
    past_kv = None
    evict_interval = 64
    n_evicted = 0

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=generated_ids[:, -1:] if past_kv is not None else generated_ids,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=True,
            )

        past_kv = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        seq_len = past_kv.key_cache[0].shape[2]
        reasoning_start = prompt_len + sink_size
        reasoning_end = seq_len - window_size

        if step > 0 and step % evict_interval == 0 and reasoning_end > reasoning_start + 10:
            n_reasoning = reasoning_end - reasoning_start
            n_to_evict = max(1, int(n_reasoning * evict_ratio))

            # Use the specified scoring function
            importance = scoring_fn(outputs.attentions, past_kv, seq_len)

            reasoning_importance = importance[reasoning_start:reasoning_end]
            sorted_indices = np.argsort(reasoning_importance)
            evict_local = sorted_indices[:n_to_evict]
            evict_global = evict_local + reasoning_start

            keep_mask = torch.ones(seq_len, dtype=torch.bool)
            for idx in evict_global:
                keep_mask[idx] = False
            keep_indices = torch.where(keep_mask)[0].to(device)

            new_cache = DynamicCache()
            for layer_idx in range(len(past_kv.key_cache)):
                k = past_kv.key_cache[layer_idx][:, :, keep_indices, :]
                v = past_kv.value_cache[layer_idx][:, :, keep_indices, :]
                new_cache.update(k, v, layer_idx)
            past_kv = new_cache
            n_evicted += n_to_evict

    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return text, n_evicted


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results/improved_scoring")
    parser.add_argument("--evict_ratio", type=float, default=0.05)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
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

    results = {}
    for method_name, scoring_fn in SCORING_METHODS.items():
        print(f"\n{'='*60}")
        print(f"=== Scoring: {method_name} (evict={args.evict_ratio:.0%}) ===")
        print(f"{'='*60}")
        correct = 0
        tested = 0
        for sample in tqdm(samples, desc=f"{method_name}"[:40]):
            try:
                text, n_ev = generate_with_scoring(
                    model, tokenizer, sample["question"], args.device,
                    scoring_fn=scoring_fn,
                    evict_ratio=args.evict_ratio,
                    sink_size=4, window_size=128,
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
        results[method_name] = {"accuracy": acc, "correct": correct, "tested": tested}
        print(f"  Accuracy: {acc:.1%} ({correct}/{tested})")

    print("\n" + "="*60)
    print(f"IMPORTANCE SCORING COMPARISON (evict_ratio={args.evict_ratio:.0%})")
    print("="*60)
    for k, v in results.items():
        print(f"  {k:20s}: {v['accuracy']:.1%} ({v['correct']}/{v['tested']})")

    with open(os.path.join(args.output_dir, "scoring_results.json"), "w") as f:
        json.dump({"evict_ratio": args.evict_ratio, "results": results}, f, indent=2)
    print(f"\nSaved to {args.output_dir}/scoring_results.json")


if __name__ == "__main__":
    main()
