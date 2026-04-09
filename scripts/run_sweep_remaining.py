"""Run remaining sweep configs: 10%, 15%, Q8(5%), Q8(10%)"""
import sys, os, json, torch
import numpy as np
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_hierarchy_sweep import generate_with_hierarchy
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.eval.gsm8k import load_gsm8k, extract_answer, check_answer

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    torch_dtype=torch.float16, device_map="cuda:0",
    trust_remote_code=True, attn_implementation="eager",
)
model.eval()

samples = load_gsm8k(n_samples=50)
print(f"Loaded {len(samples)} samples\n")

configs = [
    ("Hierarchy (10% evict)", 0.10, False, 8),
    ("Hierarchy (15% evict)", 0.15, False, 8),
    ("Hierarchy+Q8 (5% evict)", 0.05, True, 8),
    ("Hierarchy+Q8 (10% evict)", 0.10, True, 8),
]

results = {}
for label, evict_ratio, do_quant, qbits in configs:
    print(f"\n{'='*60}")
    print(f"=== {label} ===")
    print(f"{'='*60}")
    correct = 0
    tested = 0
    for sample in tqdm(samples, desc=label[:40]):
        try:
            text, n_ev = generate_with_hierarchy(
                model, tokenizer, sample["question"], "cuda:0",
                evict_ratio=evict_ratio, quantize_offloaded=do_quant,
                quant_bits=qbits, sink_size=4, window_size=128, max_new_tokens=2048,
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
    results[label] = acc
    print(f"  Accuracy: {acc:.1%} ({correct}/{tested})")

print("\n" + "="*60)
print("ALL RESULTS")
print("="*60)
for k, v in results.items():
    print(f"  {k}: {v:.1%}")

with open("results/hierarchy_sweep/remaining_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/hierarchy_sweep/remaining_results.json")
