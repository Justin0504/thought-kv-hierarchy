"""GSM8K dataset loading and answer evaluation."""

import re
import json
from pathlib import Path
from datasets import load_dataset


def load_gsm8k(split="test", n_samples=None):
    """Load GSM8K dataset from HuggingFace.

    Returns list of dicts with keys: id, question, answer (numeric).
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)
    samples = []
    for i, item in enumerate(ds):
        if n_samples and i >= n_samples:
            break
        # Extract the final numeric answer after ####
        answer_text = item["answer"]
        numeric = answer_text.split("####")[-1].strip().replace(",", "")
        samples.append({
            "id": i,
            "question": item["question"],
            "answer": numeric,
            "solution": answer_text,
        })
    return samples


def extract_answer(model_output: str) -> str | None:
    """Extract numeric answer from model reasoning output.

    Tries multiple patterns commonly used by reasoning models.
    """
    # Pattern 1: #### followed by number
    m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", model_output)
    if m:
        return m.group(1).replace(",", "")

    # Pattern 2: "the answer is X" / "answer: X"
    m = re.search(
        r"(?:the\s+)?answer\s*(?:is|:)\s*\$?([+-]?[\d,]+\.?\d*)",
        model_output,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).replace(",", "")

    # Pattern 3: boxed{X} (LaTeX style)
    m = re.search(r"\\boxed\{([+-]?[\d,]+\.?\d*)\}", model_output)
    if m:
        return m.group(1).replace(",", "")

    # Pattern 4: last standalone number in output
    numbers = re.findall(r"(?<!\w)([+-]?[\d,]+\.?\d*)(?!\w)", model_output)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def check_answer(predicted: str | None, gold: str) -> bool:
    """Check if predicted answer matches gold answer numerically."""
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(gold)) < 1e-5
    except ValueError:
        return predicted.strip() == gold.strip()
