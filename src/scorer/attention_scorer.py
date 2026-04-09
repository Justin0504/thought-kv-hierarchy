"""Attention-based importance scorer for reasoning tokens.

Extracts importance from generate() output attentions.
For each decoding step, accumulates how much attention each past
KV position received (averaged across heads and layers).
"""

import torch
import numpy as np
from dataclasses import dataclass, field


@dataclass
class AttentionImportance:
    """Accumulated attention importance per token position."""
    scores: np.ndarray = field(default_factory=lambda: np.array([]))
    n_steps: int = 0

    def top_k_ratio(self, k_pct: float) -> float:
        """What fraction of total importance is held by top k% positions."""
        if len(self.scores) == 0 or self.scores.sum() < 1e-12:
            return 0.0
        sorted_scores = np.sort(self.scores)[::-1]
        k = max(1, int(len(sorted_scores) * k_pct / 100))
        return float(sorted_scores[:k].sum() / sorted_scores.sum())


def compute_importance_from_attentions(attentions, prompt_len: int = 0):
    """Compute per-position importance from generate() attentions.

    Args:
        attentions: tuple of length n_steps. Each element is a tuple of
            length n_layers, where each tensor is [batch, heads, q_len, kv_len].
            Step 0 is prefill (q_len = prompt_len), steps 1+ are decoding (q_len=1).
        prompt_len: number of prompt tokens.

    Returns:
        AttentionImportance with cumulative scores for each position.
    """
    n_steps = len(attentions)
    if n_steps == 0:
        return AttentionImportance()

    # Find total sequence length from the last step
    last_kv_len = attentions[-1][0].shape[-1]  # kv_len of last step, layer 0
    cumulative = np.zeros(last_kv_len, dtype=np.float64)
    count = 0

    for step_attns in attentions:
        # step_attns: tuple of [batch, heads, q_len, kv_len] per layer
        layer_scores = []
        for layer_attn in step_attns:
            # [1, heads, q_len, kv_len] → last query, avg heads → [kv_len]
            score = layer_attn[0].float().mean(dim=0)[-1].cpu().numpy()
            if not np.isnan(score).any():
                layer_scores.append(score)

        if not layer_scores:
            continue  # skip steps where all layers produced NaN

        kv_len = len(layer_scores[0])
        avg_score = np.mean(layer_scores, axis=0)
        cumulative[:kv_len] += avg_score
        count += 1

    if count > 0:
        cumulative /= count

    return AttentionImportance(scores=cumulative, n_steps=count)
