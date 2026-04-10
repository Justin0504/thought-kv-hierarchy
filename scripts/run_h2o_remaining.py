"""Run remaining H2O baselines (budget=512, 1024) and combined scoring."""
import sys, os, json, torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.eval.gsm8k import load_gsm8k, extract_answer, check_answer


def compute_attention_importance(attn_weights, seq_len):
    importance = torch.zeros(seq_len)
    for layer_attn in attn_weights:
        if layer_attn is None:
            continue
        a = layer_attn.float().squeeze(0).squeeze(1)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        importance += a.sum(dim=0).cpu()
    return importance.numpy()


def generate_with_h2o(model, tokenizer, question, device,
                      budget=512, sink_size=4, max_new_tokens=2048):
    prompt = f"Please solve this math problem step by step.\n\nQuestion: {question}\n\nLet me think step by step."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    recent_window = budget // 2
    hh_slots = budget - sink_size - recent_window

    generated_ids = input_ids.clone()
    past_kv = None

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
        if seq_len > budget:
            importance = compute_attention_importance(outputs.attentions, seq_len)
            protected_start = set(range(min(sink_size, seq_len)))
            protected_end = set(range(max(0, seq_len - recent_window), seq_len))
            protected = protected_start | protected_end
            candidates = [i for i in range(seq_len) if i not in protected]
            if len(candidates) > hh_slots:
                cand_importance = [(i, importance[i]) for i in candidates]
                cand_importance.sort(key=lambda x: x[1], reverse=True)
                keep_hh = set(c[0] for c in cand_importance[:hh_slots])
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

    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return text


def generate_with_combined_scoring(model, tokenizer, question, device,
                                    evict_ratio=0.05, sink_size=4, window_size=128,
                                    max_new_tokens=2048):
    prompt = f"Please solve this math problem step by step.\n\nQuestion: {question}\n\nLet me think step by step."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    generated_ids = input_ids.clone()
    past_kv = None
    evict_interval = 64

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

            # Combined scoring: attn * value_norm - redundancy
            attn_imp = torch.zeros(seq_len)
            val_norms = torch.zeros(seq_len)
            for layer_idx, layer_attn in enumerate(outputs.attentions):
                if layer_attn is None:
                    continue
                a = layer_attn.float().squeeze(0).squeeze(1)
                if a.dim() == 1:
                    a = a.unsqueeze(0)
                attn_imp += a.sum(dim=0).cpu()
                v = past_kv.value_cache[layer_idx]
                val_norms += v.float().norm(dim=-1).mean(dim=(0, 1)).cpu()

            redundancy = torch.zeros(seq_len)
            for layer_idx in range(min(4, len(past_kv.key_cache))):
                k = past_kv.key_cache[layer_idx].float().squeeze(0)
                k_mean = k.mean(dim=0)
                k_n = torch.nn.functional.normalize(k_mean, dim=-1)
                for i in range(1, seq_len):
                    sim = torch.dot(k_n[i], k_n[i-1]).cpu()
                    redundancy[i] += max(0, sim.item())

            attn_n = attn_imp / (attn_imp.max() + 1e-8)
            val_n = val_norms / (val_norms.max() + 1e-8)
            red_n = redundancy / (redundancy.max() + 1e-8)
            importance = (attn_n * val_n - 0.3 * red_n).numpy()

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

    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return text


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_samples", type=int, default=50)
    args = parser.parse_args()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        torch_dtype=torch.float16, device_map=args.device,
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    samples = load_gsm8k(n_samples=args.n_samples)
    print(f"Loaded {len(samples)} samples\n")

    results = {}

    # H2O budget=512
    for budget in [512, 1024]:
        label = f"H2O (budget={budget})"
        print(f"\n{'='*60}\n=== {label} ===\n{'='*60}")
        correct = tested = 0
        for sample in tqdm(samples, desc=label[:40]):
            try:
                text = generate_with_h2o(model, tokenizer, sample["question"], args.device, budget=budget)
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

    # Combined scoring
    label = "combined_scoring (5% evict)"
    print(f"\n{'='*60}\n=== {label} ===\n{'='*60}")
    correct = tested = 0
    for sample in tqdm(samples, desc="combined"):
        try:
            text = generate_with_combined_scoring(model, tokenizer, sample["question"], args.device)
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
    print("REMAINING RESULTS")
    print("="*60)
    for k, v in results.items():
        print(f"  {k}: {v['accuracy']:.1%} ({v['correct']}/{v['tested']})")

    os.makedirs("results/supplementary", exist_ok=True)
    with open("results/supplementary/remaining_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/supplementary/remaining_results.json")


if __name__ == "__main__":
    main()
