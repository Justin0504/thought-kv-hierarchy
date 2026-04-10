"""H2O (Heavy-Hitter Oracle) baseline comparison.

Implements H2O-style eviction: retain sink tokens + recent window + heavy hitters.
Compare against our hierarchy approach on GSM8K (n=50).

This is the most important baseline for reviewer comparison.
"""
import sys, os, json, torch, re
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.eval.gsm8k import load_gsm8k, extract_answer, check_answer


def compute_attention_importance(attn_weights, seq_len):
    """Compute cumulative attention importance per position (H2O-style)."""
    importance = torch.zeros(seq_len)
    for layer_attn in attn_weights:
        if layer_attn is None:
            continue
        # layer_attn: [batch, heads, 1, seq_len]
        a = layer_attn.float().squeeze(0).squeeze(1)  # [heads, seq_len]
        if a.dim() == 1:
            a = a.unsqueeze(0)
        # Sum across heads
        importance += a.sum(dim=0).cpu()
    return importance.numpy()


def generate_with_h2o(model, tokenizer, question, device,
                      budget=256, sink_size=4, max_new_tokens=2048):
    """Generate with H2O-style eviction: sink + recent + heavy hitters.

    Budget = total KV cache size allowed.
    Strategy: keep sink_size initial tokens + heavy hitters + recent window.
    Recent window = budget // 2, heavy hitter slots = budget - sink_size - recent_window.
    """
    prompt = f"Please solve this math problem step by step.\n\nQuestion: {question}\n\nLet me think step by step."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    recent_window = budget // 2
    hh_slots = budget - sink_size - recent_window

    generated_ids = input_ids.clone()
    past_kv = None
    n_evictions = 0

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

        # Check if cache exceeds budget
        seq_len = past_kv.key_cache[0].shape[2]
        if seq_len > budget:
            # Compute importance from attention weights
            importance = compute_attention_importance(outputs.attentions, seq_len)

            # Protected: first sink_size + last recent_window
            protected_start = set(range(min(sink_size, seq_len)))
            protected_end = set(range(max(0, seq_len - recent_window), seq_len))
            protected = protected_start | protected_end

            # Among non-protected, keep top hh_slots by importance
            candidates = [i for i in range(seq_len) if i not in protected]
            if len(candidates) > hh_slots:
                cand_importance = [(i, importance[i]) for i in candidates]
                cand_importance.sort(key=lambda x: x[1], reverse=True)
                # Keep top heavy hitters
                keep_hh = set(c[0] for c in cand_importance[:hh_slots])
                # Positions to evict
                evict_positions = [i for i in candidates if i not in keep_hh]

                if evict_positions:
                    keep_mask = torch.ones(seq_len, dtype=torch.bool)
                    for pos in evict_positions:
                        keep_mask[pos] = False
                    keep_indices = torch.where(keep_mask)[0]

                    new_cache = DynamicCache()
                    for layer_idx in range(len(past_kv.key_cache)):
                        k = past_kv.key_cache[layer_idx][:, :, keep_indices, :]
                        v = past_kv.value_cache[layer_idx][:, :, keep_indices, :]
                        new_cache.update(k, v, layer_idx)
                    past_kv = new_cache
                    n_evictions += len(evict_positions)

    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return text, n_evictions


def generate_full(model, tokenizer, question, device, max_new_tokens=2048):
    """Generate with full KV cache (no eviction)."""
    prompt = f"Please solve this math problem step by step.\n\nQuestion: {question}\n\nLet me think step by step."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results/h2o_baseline")
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

    # H2O configs: different budgets
    # Budget means total KV cache tokens allowed
    configs = [
        ("H2O (budget=128)", 128),
        ("H2O (budget=256)", 256),
        ("H2O (budget=384)", 384),
        ("H2O (budget=512)", 512),
        ("H2O (budget=1024)", 1024),
    ]

    results = {}
    for label, budget in configs:
        print(f"\n{'='*60}")
        print(f"=== {label} ===")
        print(f"{'='*60}")
        correct = 0
        tested = 0
        for sample in tqdm(samples, desc=label[:40]):
            try:
                text, n_ev = generate_with_h2o(
                    model, tokenizer, sample["question"], args.device,
                    budget=budget, sink_size=4, max_new_tokens=args.max_new_tokens,
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
        results[label] = {"accuracy": acc, "correct": correct, "tested": tested}
        print(f"  Accuracy: {acc:.1%} ({correct}/{tested})")

    print("\n" + "="*60)
    print("H2O BASELINE RESULTS")
    print("="*60)
    for k, v in results.items():
        print(f"  {k}: {v['accuracy']:.1%} ({v['correct']}/{v['tested']})")

    with open(os.path.join(args.output_dir, "h2o_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output_dir}/h2o_results.json")


if __name__ == "__main__":
    main()
