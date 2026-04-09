"""Oracle masking experiment.

Given importance scores per KV position, mask out the bottom-K% least
important positions and measure the impact on generation quality.

This validates the core hypothesis: if masking 50% of KV-cache barely
hurts accuracy, then hierarchical memory placement is worthwhile.
"""

import torch
import numpy as np


def create_importance_mask(
    importance_scores: np.ndarray,
    mask_ratio: float,
    prompt_len: int = 0,
) -> np.ndarray:
    """Create a binary mask that keeps the top (1-mask_ratio) positions.

    Args:
        importance_scores: [seq_len] importance per position.
        mask_ratio: fraction of positions to mask (0.0 to 1.0).
        prompt_len: number of prompt tokens to always keep (never mask).

    Returns:
        Boolean array [seq_len], True = keep, False = mask out.
    """
    seq_len = len(importance_scores)
    mask = np.ones(seq_len, dtype=bool)

    if prompt_len >= seq_len or mask_ratio <= 0:
        return mask

    # Only consider reasoning tokens (after prompt) for masking
    reasoning_scores = importance_scores[prompt_len:]
    n_reasoning = len(reasoning_scores)
    n_mask = int(n_reasoning * mask_ratio)

    if n_mask == 0:
        return mask

    # Mask the positions with lowest importance
    threshold_idx = np.argsort(reasoning_scores)[n_mask]
    threshold = reasoning_scores[threshold_idx]

    for i in range(prompt_len, seq_len):
        if importance_scores[i] < threshold:
            mask[i] = False

    return mask


def apply_kv_mask(past_key_values, mask: np.ndarray):
    """Zero out KV-cache at masked positions.

    Args:
        past_key_values: tuple of (key, value) tensors per layer,
            each shape [batch, n_heads, seq_len, head_dim].
        mask: boolean array [seq_len], False = zero out.

    Returns:
        Modified past_key_values (same structure, masked positions zeroed).
    """
    device = past_key_values[0][0].device
    mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)
    # Reshape for broadcasting: [1, 1, seq_len, 1]
    mask_4d = mask_tensor.view(1, 1, -1, 1)

    masked_kv = []
    for layer_kv in past_key_values:
        k, v = layer_kv[0], layer_kv[1]
        # Trim mask if shapes don't match (can happen with padding)
        m = mask_4d[:, :, :k.shape[2], :]
        masked_k = k * m
        masked_v = v * m
        if len(layer_kv) == 2:
            masked_kv.append((masked_k, masked_v))
        else:
            masked_kv.append((masked_k, masked_v) + layer_kv[2:])

    return tuple(masked_kv)


def apply_attention_mask(attention_mask, kv_mask: np.ndarray):
    """Modify attention mask to prevent attending to masked KV positions.

    This is more principled than zeroing KV values — it prevents the
    softmax from assigning any weight to masked positions.

    Args:
        attention_mask: [batch, seq_len] or [batch, 1, q_len, kv_len] tensor.
        kv_mask: boolean array [kv_len], False = block attention.

    Returns:
        Modified attention_mask.
    """
    device = attention_mask.device
    block = torch.tensor(~kv_mask, dtype=torch.bool, device=device)

    if attention_mask.dim() == 2:
        # [batch, seq_len] — set masked positions to 0
        attention_mask = attention_mask.clone()
        attention_mask[:, :len(kv_mask)] = attention_mask[:, :len(kv_mask)].masked_fill(
            block[:attention_mask.shape[1]], 0
        )
    elif attention_mask.dim() == 4:
        # [batch, 1, q_len, kv_len] — set to large negative
        attention_mask = attention_mask.clone()
        block_4d = block.view(1, 1, 1, -1)[:, :, :, :attention_mask.shape[-1]]
        attention_mask = attention_mask.masked_fill(block_4d, torch.finfo(attention_mask.dtype).min)

    return attention_mask
