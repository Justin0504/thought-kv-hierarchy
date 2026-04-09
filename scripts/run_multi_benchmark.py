"""Multi-benchmark evaluation for memory hierarchy.

Runs hierarchy experiment on GSM8K, MATH-500, and AIME.

Usage:
    python scripts/run_multi_benchmark.py \
        --benchmarks gsm8k math aime \
        --n_samples 50 --device cuda:0
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.run_hierarchy_sweep import generate_with_hierarchy


# ── Dataset loaders ──────────────────────────────────────────────────

def load_gsm8k(n_samples=50):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    samples = []
    for item in ds:
        ans = item["answer"].split("####")[-1].strip().replace(",", "")
        samples.append({"question": item["question"], "answer": ans, "benchmark": "gsm8k"})
        if len(samples) >= n_samples:
            break
    return samples


def load_math(n_samples=50):
    try:
        ds = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset("lighteval/MATH-Hard", split="test", trust_remote_code=True)
        except Exception:
            print("Warning: Could not load MATH dataset, skipping")
            return []
    samples = []
    for item in ds:
        problem = item.get("problem", item.get("question", ""))
        solution = item.get("solution", item.get("answer", ""))
        # Extract answer from \boxed{}
        answer = extract_boxed(solution)
        if not answer:
            answer = solution.strip()
        samples.append({"question": problem, "answer": answer, "benchmark": "math"})
        if len(samples) >= n_samples:
            break
    return samples


def load_aime(n_samples=30):
    try:
        ds = load_dataset("AI-MO/aimo-validation-aime", split="train", trust_remote_code=True)
        samples = []
        for item in ds:
            problem = item.get("problem", item.get("question", ""))
            answer = str(item.get("answer", ""))
            samples.append({"question": problem, "answer": answer, "benchmark": "aime"})
            if len(samples) >= n_samples:
                break
        return samples
    except Exception:
        print("Warning: Could not load AIME dataset, skipping")
        return []


# ── Answer extraction ────────────────────────────────────────────────

def extract_boxed(text):
    """Extract content from \\boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return None


def extract_answer_general(text, benchmark):
    """Extract answer from model output based on benchmark type."""
    if benchmark == "gsm8k":
        return extract_answer_gsm8k(text)
    elif benchmark == "math":
        return extract_answer_math(text)
    elif benchmark == "aime":
        return extract_answer_aime(text)
    return None


def extract_answer_gsm8k(text):
    patterns = [
        r"####\s*([\-\d,\.]+)",
        r"[Tt]he answer is[:\s]*([\-\d,\.]+)",
        r"\\boxed\{([\-\d,\.]+)\}",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).replace(",", "").strip()
    # Fallback: last number
    nums = re.findall(r"[\-]?\d+\.?\d*", text)
    return nums[-1] if nums else None


def extract_answer_math(text):
    # Try \boxed{} first
    boxed = extract_boxed(text)
    if boxed:
        return normalize_math_answer(boxed)
    # Try "the answer is"
    m = re.search(r"[Tt]he (?:final )?answer is[:\s]*(.+?)(?:\.|$)", text)
    if m:
        return normalize_math_answer(m.group(1).strip())
    # Fallback: last \boxed or last number
    nums = re.findall(r"[\-]?\d+\.?\d*", text)
    return nums[-1] if nums else None


def extract_answer_aime(text):
    # AIME answers are integers 0-999
    boxed = extract_boxed(text)
    if boxed:
        nums = re.findall(r"\d+", boxed)
        if nums:
            return nums[0]
    # Try last integer
    m = re.search(r"[Tt]he answer is[:\s]*(\d+)", text)
    if m:
        return m.group(1)
    nums = re.findall(r"\d+", text)
    return nums[-1] if nums else None


def normalize_math_answer(ans):
    """Normalize math answer for comparison."""
    ans = ans.strip()
    # Remove $, \text{}, etc
    ans = re.sub(r"\$", "", ans)
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\\left|\\right", "", ans)
    ans = re.sub(r"\s+", " ", ans).strip()
    return ans


def check_answer_general(pred, gold, benchmark):
    if pred is None:
        return False
    pred = str(pred).strip()
    gold = str(gold).strip()

    if benchmark == "aime":
        # Integer comparison
        try:
            return int(pred) == int(gold)
        except ValueError:
            return False

    if benchmark == "gsm8k":
        try:
            return abs(float(pred.replace(",", "")) - float(gold.replace(",", ""))) < 0.01
        except ValueError:
            return pred == gold

    # MATH: try numeric, then string
    try:
        return abs(float(pred) - float(gold)) < 0.01
    except ValueError:
        return normalize_math_answer(pred) == normalize_math_answer(gold)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "math"],
                        choices=["gsm8k", "math", "aime"])
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results/multi_benchmark")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device,
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()

    # Configs to test
    configs = [
        ("Full (0% evict)", 0.0),
        ("Hierarchy (3% evict)", 0.03),
        ("Hierarchy (5% evict)", 0.05),
        ("Hierarchy (10% evict)", 0.10),
    ]

    all_results = {}

    for bench_name in args.benchmarks:
        print(f"\n{'#'*70}")
        print(f"# Benchmark: {bench_name.upper()}")
        print(f"{'#'*70}")

        if bench_name == "gsm8k":
            samples = load_gsm8k(args.n_samples)
        elif bench_name == "math":
            samples = load_math(args.n_samples)
        elif bench_name == "aime":
            samples = load_aime(min(args.n_samples, 30))

        if not samples:
            print(f"  No samples loaded for {bench_name}, skipping")
            continue

        print(f"Loaded {len(samples)} samples")
        bench_results = {}

        for label, evict_ratio in configs:
            print(f"\n{'='*60}")
            print(f"=== {bench_name.upper()} / {label} ===")
            print(f"{'='*60}")
            correct = 0
            tested = 0

            for sample in tqdm(samples, desc=f"{bench_name}/{label}"[:40]):
                try:
                    text, _ = generate_with_hierarchy(
                        model, tokenizer, sample["question"], args.device,
                        evict_ratio=evict_ratio,
                        quantize_offloaded=False,
                        quant_bits=8,
                        sink_size=args.sink_size,
                        window_size=args.window_size,
                        max_new_tokens=args.max_new_tokens,
                    )
                    pred = extract_answer_general(text, bench_name)
                    if check_answer_general(pred, sample["answer"], bench_name):
                        correct += 1
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"  Error: {e}")
                tested += 1

            acc = correct / max(1, tested)
            bench_results[label] = {
                "accuracy": acc, "correct": correct, "tested": tested
            }
            print(f"  Accuracy: {acc:.1%} ({correct}/{tested})")

        all_results[bench_name] = bench_results

    # Summary
    print("\n" + "=" * 70)
    print("MULTI-BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Config':<30s}", end="")
    for bench in args.benchmarks:
        if bench in all_results:
            print(f"  {bench.upper():>10s}", end="")
    print()
    print("-" * 70)

    for label, _ in configs:
        print(f"{label:<30s}", end="")
        for bench in args.benchmarks:
            if bench in all_results and label in all_results[bench]:
                acc = all_results[bench][label]["accuracy"]
                print(f"  {acc:>9.1%}", end="")
            else:
                print(f"  {'---':>10s}", end="")
        print()

    # Save
    summary = {
        "model": args.model,
        "n_samples": args.n_samples,
        "sink_size": args.sink_size,
        "window_size": args.window_size,
        "results": all_results,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
