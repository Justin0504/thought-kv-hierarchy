"""Gradient-based importance scorer for KV cache positions.

Memory-optimized with 8-bit model support. Keeps references to
leaf tensors for proper gradient accumulation.
"""

import torch
import numpy as np


def get_n_layers(past_kv):
    if hasattr(past_kv, 'key_cache'):
        return len(past_kv.key_cache)
    return len(past_kv.layers)


def get_kv(past_kv, layer_idx):
    if hasattr(past_kv, 'key_cache'):
        return past_kv.key_cache[layer_idx], past_kv.value_cache[layer_idx]
    return past_kv.layers[layer_idx].keys, past_kv.layers[layer_idx].values


def kv_cache_to_cpu(past_kv):
    n_layers = get_n_layers(past_kv)
    cpu_kv = []
    for i in range(n_layers):
        k, v = get_kv(past_kv, i)
        cpu_kv.append((k.cpu(), v.cpu()))
    return cpu_kv


def compute_gradient_importance(model, past_kv_cpu, last_token_id, device,
                                 max_positions=512):
    """Compute per-position gradient-based importance.

    Keeps references to the leaf tensors (before cache.update) so
    gradients are properly accumulated during backward().
    """
    from transformers import DynamicCache

    full_seq_len = past_kv_cpu[0][0].shape[2]
    n_layers = len(past_kv_cpu)

    # Truncate if needed
    start = max(0, full_seq_len - max_positions)
    trunc_len = full_seq_len - start

    # Build cache; keep leaf tensor references for gradient access
    grad_cache = DynamicCache()
    leaf_keys = []
    leaf_values = []
    for layer_idx in range(n_layers):
        k_cpu, v_cpu = past_kv_cpu[layer_idx]
        k = k_cpu[:, :, start:, :].to(device).requires_grad_(True)
        v = v_cpu[:, :, start:, :].to(device).requires_grad_(True)
        leaf_keys.append(k)
        leaf_values.append(v)
        grad_cache.update(k, v, layer_idx)

    # Forward the last token
    input_ids = torch.tensor([[last_token_id]], device=device)
    outputs = model(input_ids, past_key_values=grad_cache, use_cache=True)

    # Backward from max logit
    logits = outputs.logits[:, -1, :].float()
    target_logit = logits.max()
    target_logit.backward()

    # Collect gradient norms from leaf tensors
    trunc_importance = torch.zeros(trunc_len, device="cpu")
    for layer_idx in range(n_layers):
        k_grad = leaf_keys[layer_idx].grad
        v_grad = leaf_values[layer_idx].grad
        if k_grad is not None:
            trunc_importance += torch.linalg.vector_norm(k_grad.float(), dim=(0, 1, 3)).cpu()
        if v_grad is not None:
            trunc_importance += torch.linalg.vector_norm(v_grad.float(), dim=(0, 1, 3)).cpu()

    del grad_cache, outputs, input_ids, leaf_keys, leaf_values
    torch.cuda.empty_cache()

    # Map back to full sequence
    importance = np.zeros(full_seq_len, dtype=np.float64)
    importance[start:] = trunc_importance.numpy()

    return importance
