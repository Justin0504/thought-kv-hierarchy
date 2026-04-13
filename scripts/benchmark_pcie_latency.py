"""Task 1: PCIe Offloading Latency Measurement

Measures GPU HBM <-> CPU DDR transfer latency for KV cache data,
matching the model config from the thought-kv-hierarchy experiments.

Model config (must match):
  - n_layers = 28 (DeepSeek-R1-Distill-Qwen-7B)
  - n_heads = 28
  - head_dim = 128
  - dtype = torch.float16

Usage:
    cd ~/spec_B
    source venv/bin/activate
    python scripts/benchmark_pcie_latency.py
"""

import os
import json
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Model config (must match algorithm-side experiments) ──
N_LAYERS = 28
N_HEADS = 28
HEAD_DIM = 128
DTYPE = torch.float16
BYTES_PER_ELEM = 2  # fp16

# Token counts to sweep — covers realistic offload scenarios:
#   - Eviction interval: 64 tokens
#   - Typical offload at 50% HBM: ~600 tokens
#   - Typical offload at 30% HBM: ~840 tokens
#   - Full reasoning chain: ~1422 tokens
TOKEN_COUNTS = [1, 4, 16, 64, 128, 256, 512, 768, 1024, 1422]

N_WARMUP = 5
N_REPEATS = 50


def bytes_per_token():
    """KV cache size per token across all layers."""
    # per layer: 1 * n_heads * 1 * head_dim * 2 (K+V) * bytes_per_elem
    per_layer = N_HEADS * HEAD_DIM * 2 * BYTES_PER_ELEM
    return per_layer * N_LAYERS


def benchmark_single_layer(n_tokens, device="cuda"):
    """Benchmark GPU<->CPU transfer for a single layer's KV cache."""
    k = torch.randn(1, N_HEADS, n_tokens, HEAD_DIM, dtype=DTYPE, device=device)
    v = torch.randn(1, N_HEADS, n_tokens, HEAD_DIM, dtype=DTYPE, device=device)

    # Warmup
    for _ in range(N_WARMUP):
        k_cpu = k.to("cpu", non_blocking=True)
        v_cpu = v.to("cpu", non_blocking=True)
        torch.cuda.synchronize()
        k.copy_(k_cpu.to(device, non_blocking=True))
        v.copy_(v_cpu.to(device, non_blocking=True))
        torch.cuda.synchronize()

    # Measure GPU -> CPU (offload)
    g2c_times = []
    for _ in range(N_REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        k_cpu = k.to("cpu", non_blocking=True)
        v_cpu = v.to("cpu", non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        g2c_times.append(start.elapsed_time(end))

    # Measure CPU -> GPU (prefetch)
    k_cpu = k.to("cpu")
    v_cpu = v.to("cpu")
    torch.cuda.synchronize()

    c2g_times = []
    for _ in range(N_REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        k_gpu = k_cpu.to(device, non_blocking=True)
        v_gpu = v_cpu.to(device, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        c2g_times.append(start.elapsed_time(end))

    return np.array(g2c_times), np.array(c2g_times)


def benchmark_all_layers(n_tokens, device="cuda"):
    """Benchmark transfer across all 28 layers (sequential, as in real offload)."""
    # Create KV cache for all layers
    kv_cache = []
    for _ in range(N_LAYERS):
        k = torch.randn(1, N_HEADS, n_tokens, HEAD_DIM, dtype=DTYPE, device=device)
        v = torch.randn(1, N_HEADS, n_tokens, HEAD_DIM, dtype=DTYPE, device=device)
        kv_cache.append((k, v))

    # Warmup
    for _ in range(N_WARMUP):
        cpu_cache = []
        for k, v in kv_cache:
            cpu_cache.append((k.to("cpu", non_blocking=True), v.to("cpu", non_blocking=True)))
        torch.cuda.synchronize()
        for i, (k_cpu, v_cpu) in enumerate(cpu_cache):
            kv_cache[i][0].copy_(k_cpu.to(device, non_blocking=True))
            kv_cache[i][1].copy_(v_cpu.to(device, non_blocking=True))
        torch.cuda.synchronize()

    # Measure GPU -> CPU (offload all layers)
    g2c_times = []
    for _ in range(N_REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        cpu_cache = []
        for k, v in kv_cache:
            cpu_cache.append((k.to("cpu", non_blocking=True), v.to("cpu", non_blocking=True)))
        end.record()
        torch.cuda.synchronize()
        g2c_times.append(start.elapsed_time(end))

    # Measure CPU -> GPU (prefetch all layers)
    cpu_cache = [(k.to("cpu"), v.to("cpu")) for k, v in kv_cache]
    torch.cuda.synchronize()

    c2g_times = []
    for _ in range(N_REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for k_cpu, v_cpu in cpu_cache:
            k_gpu = k_cpu.to(device, non_blocking=True)
            v_gpu = v_cpu.to(device, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        c2g_times.append(start.elapsed_time(end))

    return np.array(g2c_times), np.array(c2g_times)


def main():
    device = "cuda:0"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"KV config: {N_LAYERS} layers x {N_HEADS} heads x {HEAD_DIM} dim, fp16")
    print(f"Per-token KV size: {bytes_per_token() / 1024:.1f} KB")
    print(f"Warmup: {N_WARMUP}, Repeats: {N_REPEATS}")
    print()

    # ── Single layer benchmark ──
    print("=" * 60)
    print("Single Layer Transfer Benchmark")
    print("=" * 60)
    single_layer_results = {}
    for n_tok in tqdm(TOKEN_COUNTS, desc="Single layer"):
        g2c, c2g = benchmark_single_layer(n_tok, device)
        data_mb = N_HEADS * n_tok * HEAD_DIM * 2 * BYTES_PER_ELEM / 1024 / 1024
        single_layer_results[n_tok] = {
            "data_size_mb": round(data_mb, 3),
            "gpu_to_cpu_ms": {"mean": round(g2c.mean(), 3), "std": round(g2c.std(), 3),
                              "median": round(np.median(g2c), 3)},
            "cpu_to_gpu_ms": {"mean": round(c2g.mean(), 3), "std": round(c2g.std(), 3),
                              "median": round(np.median(c2g), 3)},
        }
        print(f"  {n_tok:5d} tokens ({data_mb:7.2f} MB/layer): "
              f"GPU->CPU {g2c.mean():.3f}ms, CPU->GPU {c2g.mean():.3f}ms")

    # ── All layers benchmark (realistic end-to-end) ──
    print()
    print("=" * 60)
    print("All 28 Layers Transfer Benchmark (end-to-end)")
    print("=" * 60)
    all_layers_results = {}
    for n_tok in tqdm(TOKEN_COUNTS, desc="All layers"):
        try:
            g2c, c2g = benchmark_all_layers(n_tok, device)
            total_mb = bytes_per_token() * n_tok / 1024 / 1024
            bw_g2c = (total_mb / 1024) / (g2c.mean() / 1000)  # GB/s
            bw_c2g = (total_mb / 1024) / (c2g.mean() / 1000)  # GB/s
            all_layers_results[n_tok] = {
                "total_data_mb": round(total_mb, 2),
                "gpu_to_cpu_ms": {"mean": round(g2c.mean(), 3), "std": round(g2c.std(), 3),
                                  "median": round(np.median(g2c), 3)},
                "cpu_to_gpu_ms": {"mean": round(c2g.mean(), 3), "std": round(c2g.std(), 3),
                                  "median": round(np.median(c2g), 3)},
                "bandwidth_gpu_to_cpu_gbps": round(bw_g2c, 2),
                "bandwidth_cpu_to_gpu_gbps": round(bw_c2g, 2),
            }
            print(f"  {n_tok:5d} tokens ({total_mb:7.1f} MB total): "
                  f"GPU->CPU {g2c.mean():.2f}ms ({bw_g2c:.1f} GB/s), "
                  f"CPU->GPU {c2g.mean():.2f}ms ({bw_c2g:.1f} GB/s)")
        except torch.cuda.OutOfMemoryError:
            print(f"  {n_tok:5d} tokens: OOM, skipping")
            torch.cuda.empty_cache()

    # ── Reference: typical decoding step time ──
    print()
    print("=" * 60)
    print("Reference: Attention Compute Time (no model, matmul only)")
    print("=" * 60)
    # Simulate one attention step to compare transfer vs compute
    compute_times = {}
    for n_tok in [128, 256, 512, 1024, 1422]:
        q = torch.randn(1, N_HEADS, 1, HEAD_DIM, dtype=DTYPE, device=device)
        k = torch.randn(1, N_HEADS, n_tok, HEAD_DIM, dtype=DTYPE, device=device)
        v = torch.randn(1, N_HEADS, n_tok, HEAD_DIM, dtype=DTYPE, device=device)
        # Warmup
        for _ in range(N_WARMUP):
            attn = torch.matmul(q, k.transpose(-2, -1)) / (HEAD_DIM ** 0.5)
            out = torch.matmul(torch.softmax(attn, dim=-1), v)
        torch.cuda.synchronize()
        # Measure
        times = []
        for _ in range(N_REPEATS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            attn = torch.matmul(q, k.transpose(-2, -1)) / (HEAD_DIM ** 0.5)
            out = torch.matmul(torch.softmax(attn, dim=-1), v)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        times = np.array(times)
        compute_times[n_tok] = {"mean": round(times.mean(), 3), "std": round(times.std(), 3)}
        print(f"  {n_tok:5d} tokens: attention compute {times.mean():.3f}ms")

    # ── Save results ──
    results = {
        "gpu": torch.cuda.get_device_name(0),
        "kv_config": {
            "n_layers": N_LAYERS, "n_heads": N_HEADS, "head_dim": HEAD_DIM,
            "dtype": "float16", "bytes_per_token": bytes_per_token(),
        },
        "n_warmup": N_WARMUP,
        "n_repeats": N_REPEATS,
        "single_layer": single_layer_results,
        "all_layers": all_layers_results,
        "attention_compute_ref": compute_times,
    }
    with open(os.path.join(output_dir, "pcie_latency_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Plot 1: Latency vs token count (all layers) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    tokens = sorted(all_layers_results.keys())
    g2c_means = [all_layers_results[t]["gpu_to_cpu_ms"]["mean"] for t in tokens]
    c2g_means = [all_layers_results[t]["cpu_to_gpu_ms"]["mean"] for t in tokens]
    total_mbs = [all_layers_results[t]["total_data_mb"] for t in tokens]

    ax1.plot(tokens, g2c_means, "o-", color="#E53935", label="GPU → CPU (offload)", linewidth=2)
    ax1.plot(tokens, c2g_means, "s-", color="#1565C0", label="CPU → GPU (prefetch)", linewidth=2)
    ax1.set_xlabel("Number of Tokens Transferred")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("PCIe Transfer Latency (28 layers, all KV)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Secondary x-axis for data size
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    tick_tokens = [1, 256, 512, 1024, 1422]
    tick_mbs = [bytes_per_token() * t / 1024 / 1024 for t in tick_tokens]
    ax1_top.set_xticks(tick_tokens)
    ax1_top.set_xticklabels([f"{m:.0f} MB" for m in tick_mbs], fontsize=8)
    ax1_top.set_xlabel("Data Size", fontsize=9)

    # Plot 2: Bandwidth utilization
    g2c_bw = [all_layers_results[t]["bandwidth_gpu_to_cpu_gbps"] for t in tokens]
    c2g_bw = [all_layers_results[t]["bandwidth_cpu_to_gpu_gbps"] for t in tokens]

    ax2.plot(tokens, g2c_bw, "o-", color="#E53935", label="GPU → CPU", linewidth=2)
    ax2.plot(tokens, c2g_bw, "s-", color="#1565C0", label="CPU → GPU", linewidth=2)
    # RTX 5080 is PCIe Gen5 x16 = ~63 GB/s theoretical
    ax2.axhline(y=63, color="gray", linestyle="--", alpha=0.5, label="PCIe Gen5 x16 theoretical (63 GB/s)")
    ax2.set_xlabel("Number of Tokens Transferred")
    ax2.set_ylabel("Bandwidth (GB/s)")
    ax2.set_title("PCIe Bandwidth Utilization")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pcie_latency_plot.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "pcie_latency_plot.pdf"))
    plt.close()

    # ── Plot 3: Transfer time vs compute time comparison ──
    fig, ax = plt.subplots(figsize=(8, 5))
    comp_tokens = sorted(compute_times.keys())
    comp_means = [compute_times[t]["mean"] for t in comp_tokens]
    transfer_means = [all_layers_results[t]["cpu_to_gpu_ms"]["mean"]
                      for t in comp_tokens if t in all_layers_results]
    valid_tokens = [t for t in comp_tokens if t in all_layers_results]

    ax.bar([str(t) for t in valid_tokens], transfer_means, width=0.35, label="CPU→GPU transfer",
           color="#1565C0", alpha=0.8)
    ax.bar([str(t) for t in comp_tokens], comp_means, width=0.35, label="Attention compute",
           color="#FF8F00", alpha=0.8, bottom=[transfer_means[i] if comp_tokens[i] in all_layers_results else 0
                                               for i in range(len(comp_tokens))])
    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Transfer Overhead vs Compute Time\n(Can transfer be hidden?)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transfer_vs_compute.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "transfer_vs_compute.pdf"))
    plt.close()

    print(f"\nResults saved to {output_dir}/")
    print("Files: pcie_latency_results.json, pcie_latency_plot.png/pdf, transfer_vs_compute.png/pdf")


if __name__ == "__main__":
    main()
