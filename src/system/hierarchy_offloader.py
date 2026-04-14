"""Real GPU-CPU KV cache offloading for memory hierarchy.

Implements actual data movement between GPU HBM and CPU DDR
during generation, with asynchronous prefetch support.
"""

import torch
import numpy as np
from transformers import DynamicCache


def get_n_layers(past_kv):
    if hasattr(past_kv, 'key_cache'):
        return len(past_kv.key_cache)
    return len(past_kv.layers)


def get_kv(past_kv, layer_idx):
    if hasattr(past_kv, 'key_cache'):
        return past_kv.key_cache[layer_idx], past_kv.value_cache[layer_idx]
    return past_kv.layers[layer_idx].keys, past_kv.layers[layer_idx].values


class HierarchyOffloader:
    """Manages tiered KV cache across GPU (HBM) and CPU (DDR).

    Tokens are classified into tiers:
      T0 (GPU): prompt + sinks + recent window + top-importance tokens
      T1 (CPU): offloaded tokens, recalled for attention computation
      T3 (evicted): permanently removed

    During attention, T1 tokens are prefetched from CPU to GPU,
    participate in attention, then offloaded back.
    """

    def __init__(self, n_layers, n_heads, head_dim, device,
                 hbm_ratio=0.5, evict_ratio=0.1,
                 sink_size=4, window_size=128,
                 manage_interval=64, dtype=torch.float16):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        self.hbm_ratio = hbm_ratio
        self.evict_ratio = evict_ratio
        self.sink_size = sink_size
        self.window_size = window_size
        self.manage_interval = manage_interval
        self.dtype = dtype

        # CPU-side storage for offloaded KV
        # Structure: list of (k_cpu, v_cpu) per layer, each [1, heads, n_offloaded, dim]
        self.cpu_keys = [[] for _ in range(n_layers)]
        self.cpu_values = [[] for _ in range(n_layers)]
        self.cpu_positions = []  # original position indices of offloaded tokens

        # Timing stats
        self.stats = {
            "offload_count": 0,
            "offload_tokens": 0,
            "offload_time_ms": 0.0,
            "prefetch_count": 0,
            "prefetch_tokens": 0,
            "prefetch_time_ms": 0.0,
            "evict_count": 0,
            "evict_tokens": 0,
            "compute_time_ms": 0.0,
            "total_steps": 0,
        }

    def _get_protected(self, cache_len, prompt_len):
        """Return set of cache positions that must stay on GPU."""
        protected = set()
        for i in range(min(prompt_len, cache_len)):
            protected.add(i)
        for i in range(prompt_len, min(prompt_len + self.sink_size, cache_len)):
            protected.add(i)
        for i in range(max(0, cache_len - self.window_size), cache_len):
            protected.add(i)
        return protected

    def manage_cache(self, past_kv, importance, position_map, prompt_len, step):
        """Perform tier management: offload low-importance to CPU, evict bottom.

        Args:
            past_kv: DynamicCache on GPU
            importance: numpy array of importance scores per original position
            position_map: maps cache index -> original position
            prompt_len: number of prompt tokens
            step: current decoding step

        Returns:
            (new_past_kv, new_position_map) after offloading and eviction
        """
        if (step + 1) % self.manage_interval != 0:
            return past_kv, position_map

        cache_len = past_kv.get_seq_length()
        protected = self._get_protected(cache_len, prompt_len)
        candidates = [i for i in range(cache_len) if i not in protected]

        if len(candidates) < 10:
            return past_kv, position_map

        # Score candidates
        cand_imp = np.array([importance[position_map[c]] for c in candidates])
        sorted_idx = np.argsort(cand_imp)  # ascending

        n_cand = len(candidates)
        n_evict = int(n_cand * self.evict_ratio)
        n_keep_gpu = int(n_cand * self.hbm_ratio)
        n_offload = n_cand - n_evict - n_keep_gpu

        if n_offload <= 0 and n_evict <= 0:
            return past_kv, position_map

        # Classify: bottom evict_ratio% -> evict, next chunk -> offload to CPU, rest -> keep on GPU
        evict_set = set()
        offload_set = set()
        for i in range(n_evict):
            evict_set.add(candidates[sorted_idx[i]])
        for i in range(n_evict, n_evict + max(0, n_offload)):
            offload_set.add(candidates[sorted_idx[i]])

        # --- Step 1: Offload to CPU (timed) ---
        if offload_set:
            offload_sorted = sorted(offload_set)
            offload_indices = torch.tensor(offload_sorted, device=self.device, dtype=torch.long)

            torch.cuda.synchronize()
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()

            offload_positions = [position_map[i] for i in offload_sorted]
            for layer_idx in range(get_n_layers(past_kv)):
                k, v = get_kv(past_kv, layer_idx)
                k_off = k.index_select(2, offload_indices).to("cpu", non_blocking=True)
                v_off = v.index_select(2, offload_indices).to("cpu", non_blocking=True)
                self.cpu_keys[layer_idx].append(k_off)
                self.cpu_values[layer_idx].append(v_off)

            end_ev.record()
            torch.cuda.synchronize()
            offload_ms = start_ev.elapsed_time(end_ev)

            self.cpu_positions.extend(offload_positions)
            self.stats["offload_count"] += 1
            self.stats["offload_tokens"] += len(offload_set)
            self.stats["offload_time_ms"] += offload_ms

        # --- Step 2: Remove offloaded + evicted from GPU cache ---
        remove_set = evict_set | offload_set
        if remove_set:
            keep_sorted = sorted([i for i in range(cache_len) if i not in remove_set])
            keep_indices = torch.tensor(keep_sorted, device=self.device, dtype=torch.long)

            new_cache = DynamicCache()
            for layer_idx in range(get_n_layers(past_kv)):
                k, v = get_kv(past_kv, layer_idx)
                new_cache.update(
                    k.index_select(2, keep_indices),
                    v.index_select(2, keep_indices),
                    layer_idx,
                )
            position_map = [position_map[i] for i in keep_sorted]
            past_kv = new_cache

            self.stats["evict_count"] += 1
            self.stats["evict_tokens"] += len(evict_set)

        return past_kv, position_map

    def prefetch_and_build_full_cache(self, past_kv):
        """Prefetch CPU-offloaded KV back to GPU, build full cache for attention.

        Returns:
            full_cache: DynamicCache with GPU + prefetched CPU tokens
            prefetch_ms: time spent on CPU->GPU transfer
        """
        if not self.cpu_positions:
            return past_kv, 0.0

        torch.cuda.synchronize()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()

        full_cache = DynamicCache()
        for layer_idx in range(get_n_layers(past_kv)):
            k_gpu, v_gpu = get_kv(past_kv, layer_idx)

            # Concatenate all CPU chunks for this layer
            if self.cpu_keys[layer_idx]:
                k_cpu_all = torch.cat(self.cpu_keys[layer_idx], dim=2)
                v_cpu_all = torch.cat(self.cpu_values[layer_idx], dim=2)
                k_prefetched = k_cpu_all.to(self.device, non_blocking=True)
                v_prefetched = v_cpu_all.to(self.device, non_blocking=True)
                # Concatenate: [GPU tokens, prefetched CPU tokens]
                k_full = torch.cat([k_gpu, k_prefetched], dim=2)
                v_full = torch.cat([v_gpu, v_prefetched], dim=2)
            else:
                k_full = k_gpu
                v_full = v_gpu

            full_cache.update(k_full, v_full, layer_idx)

        end_ev.record()
        torch.cuda.synchronize()
        prefetch_ms = start_ev.elapsed_time(end_ev)

        self.stats["prefetch_count"] += 1
        self.stats["prefetch_tokens"] += len(self.cpu_positions)
        self.stats["prefetch_time_ms"] += prefetch_ms

        return full_cache, prefetch_ms

    def cleanup_prefetched(self, full_cache, past_kv):
        """After attention, discard prefetched tokens from GPU, keep only GPU-resident tokens.

        The full_cache has [gpu_tokens, cpu_tokens]. We need to trim back to just gpu_tokens
        and the newly generated token (appended by the model).
        """
        if not self.cpu_positions:
            return full_cache

        n_gpu = past_kv.get_seq_length()
        n_cpu = len(self.cpu_positions)
        n_full = full_cache.get_seq_length()
        # full_cache = [gpu_old, cpu_prefetched, new_token]
        # keep = [gpu_old, new_token] = indices [0..n_gpu-1] + [n_gpu+n_cpu]
        keep = list(range(n_gpu)) + list(range(n_gpu + n_cpu, n_full))
        keep_indices = torch.tensor(keep, device=self.device, dtype=torch.long)

        trimmed = DynamicCache()
        for layer_idx in range(get_n_layers(full_cache)):
            k, v = get_kv(full_cache, layer_idx)
            trimmed.update(
                k.index_select(2, keep_indices),
                v.index_select(2, keep_indices),
                layer_idx,
            )
        return trimmed

    def get_summary(self):
        """Return human-readable stats summary."""
        s = self.stats
        total_time = s["offload_time_ms"] + s["prefetch_time_ms"] + s["compute_time_ms"]
        return {
            "offload": {
                "count": s["offload_count"],
                "total_tokens": s["offload_tokens"],
                "total_ms": round(s["offload_time_ms"], 2),
                "avg_ms": round(s["offload_time_ms"] / max(1, s["offload_count"]), 2),
            },
            "prefetch": {
                "count": s["prefetch_count"],
                "total_tokens": s["prefetch_tokens"],
                "total_ms": round(s["prefetch_time_ms"], 2),
                "avg_ms": round(s["prefetch_time_ms"] / max(1, s["prefetch_count"]), 2),
            },
            "eviction": {
                "count": s["evict_count"],
                "total_tokens": s["evict_tokens"],
            },
            "compute": {
                "total_ms": round(s["compute_time_ms"], 2),
            },
            "total_steps": s["total_steps"],
            "transfer_overhead_pct": round(
                (s["offload_time_ms"] + s["prefetch_time_ms"]) / max(1, total_time) * 100, 1
            ),
        }
