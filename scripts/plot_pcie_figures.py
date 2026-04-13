"""Generate paper-quality figures for PCIe latency measurement (Task 1).

Produces 4 figures:
  1. Latency vs tokens (all layers) — main result
  2. Bandwidth utilization
  3. Transfer vs compute comparison (log scale)
  4. Single layer vs all layers breakdown

Usage:
    cd ~/thought-kv-hierarchy
    source venv/bin/activate
    python scripts/plot_pcie_figures.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "g2c": "#D32F2F",       # red — GPU to CPU
    "c2g": "#1565C0",       # blue — CPU to GPU
    "compute": "#FF8F00",   # amber — attention compute
    "transfer": "#1565C0",  # blue — transfer
    "theoretical": "#757575",
    "single": "#7B1FA2",    # purple — single layer
    "all": "#1565C0",       # blue — all layers
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def fig1_latency_vs_tokens(data, output_dir):
    """Main result: end-to-end transfer latency across all 28 layers."""
    all_layers = data["all_layers"]
    tokens = sorted([int(k) for k in all_layers.keys()])

    g2c = [all_layers[str(t)]["gpu_to_cpu_ms"]["mean"] for t in tokens]
    c2g = [all_layers[str(t)]["cpu_to_gpu_ms"]["mean"] for t in tokens]
    g2c_std = [all_layers[str(t)]["gpu_to_cpu_ms"]["std"] for t in tokens]
    c2g_std = [all_layers[str(t)]["cpu_to_gpu_ms"]["std"] for t in tokens]
    data_mb = [all_layers[str(t)]["total_data_mb"] for t in tokens]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.errorbar(tokens, g2c, yerr=g2c_std, fmt="o-", color=COLORS["g2c"],
                linewidth=2, markersize=6, capsize=3, label="GPU → CPU (offload)")
    ax.errorbar(tokens, c2g, yerr=c2g_std, fmt="s-", color=COLORS["c2g"],
                linewidth=2, markersize=6, capsize=3, label="CPU → GPU (prefetch)")

    # Annotate key points
    for t, g, c in zip(tokens, g2c, c2g):
        if t in [64, 256, 1024, 1422]:
            mb = all_layers[str(t)]["total_data_mb"]
            ax.annotate(f"{mb:.0f} MB", (t, c), textcoords="offset points",
                        xytext=(8, 5), fontsize=8, color=COLORS["c2g"], alpha=0.8)

    # Mark the eviction interval region
    ax.axvspan(40, 90, alpha=0.08, color="green",
               label="Eviction interval (~64 tokens)")

    # Mark typical offload sizes
    ax.axvline(x=600, color="gray", linestyle=":", alpha=0.4)
    ax.annotate("50% HBM\noffload", (600, max(c2g)*0.6), fontsize=8,
                color="gray", ha="center")
    ax.axvline(x=840, color="gray", linestyle=":", alpha=0.4)
    ax.annotate("30% HBM\noffload", (840, max(c2g)*0.45), fontsize=8,
                color="gray", ha="center")

    ax.set_xlabel("Number of Tokens Transferred")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("PCIe Transfer Latency — All 28 Layers\n(RTX 5080, PCIe Gen5 x16, KV cache fp16)")
    ax.legend(loc="upper left")
    ax.set_xlim(-50, 1500)

    plt.savefig(os.path.join(output_dir, "fig_pcie_latency.png"), dpi=200)
    plt.savefig(os.path.join(output_dir, "fig_pcie_latency.pdf"))
    plt.close()
    print("  Saved fig_pcie_latency")


def fig2_bandwidth(data, output_dir):
    """Bandwidth utilization vs theoretical PCIe Gen5 max."""
    all_layers = data["all_layers"]
    tokens = sorted([int(k) for k in all_layers.keys()])

    g2c_bw = [all_layers[str(t)]["bandwidth_gpu_to_cpu_gbps"] for t in tokens]
    c2g_bw = [all_layers[str(t)]["bandwidth_cpu_to_gpu_gbps"] for t in tokens]
    data_mb = [all_layers[str(t)]["total_data_mb"] for t in tokens]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(tokens, g2c_bw, "o-", color=COLORS["g2c"], linewidth=2,
            markersize=6, label="GPU → CPU (offload)")
    ax.plot(tokens, c2g_bw, "s-", color=COLORS["c2g"], linewidth=2,
            markersize=6, label="CPU → GPU (prefetch)")

    # Theoretical max
    ax.axhline(y=63, color=COLORS["theoretical"], linestyle="--", linewidth=1.5,
               alpha=0.6, label="PCIe Gen5 x16 theoretical (63 GB/s)")

    # Utilization annotations
    peak_g2c = max(g2c_bw)
    peak_c2g = max(c2g_bw)
    ax.annotate(f"Peak: {peak_g2c:.1f} GB/s ({peak_g2c/63*100:.0f}%)",
                (tokens[g2c_bw.index(peak_g2c)], peak_g2c),
                textcoords="offset points", xytext=(10, 10), fontsize=9,
                color=COLORS["g2c"],
                arrowprops=dict(arrowstyle="->", color=COLORS["g2c"], alpha=0.6))
    ax.annotate(f"Peak: {peak_c2g:.1f} GB/s ({peak_c2g/63*100:.0f}%)",
                (tokens[c2g_bw.index(peak_c2g)], peak_c2g),
                textcoords="offset points", xytext=(10, -20), fontsize=9,
                color=COLORS["c2g"],
                arrowprops=dict(arrowstyle="->", color=COLORS["c2g"], alpha=0.6))

    ax.set_xlabel("Number of Tokens Transferred")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("PCIe Bandwidth Utilization\n(RTX 5080, PCIe Gen5 x16)")
    ax.legend(loc="center right")
    ax.set_ylim(0, 70)
    ax.set_xlim(-50, 1500)

    # Add secondary x-axis for data size
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    tick_positions = [64, 256, 512, 1024, 1422]
    tick_labels = [f"{all_layers[str(t)]['total_data_mb']:.0f} MB" for t in tick_positions]
    ax_top.set_xticks(tick_positions)
    ax_top.set_xticklabels(tick_labels, fontsize=8)
    ax_top.set_xlabel("Total KV Data Size", fontsize=10)

    plt.savefig(os.path.join(output_dir, "fig_pcie_bandwidth.png"), dpi=200)
    plt.savefig(os.path.join(output_dir, "fig_pcie_bandwidth.pdf"))
    plt.close()
    print("  Saved fig_pcie_bandwidth")


def fig3_transfer_vs_compute(data, output_dir):
    """Log-scale comparison: transfer time dwarfs compute time."""
    all_layers = data["all_layers"]
    compute = data["attention_compute_ref"]

    seq_lens = sorted([int(k) for k in compute.keys()])

    transfer_ms = [all_layers[str(t)]["cpu_to_gpu_ms"]["mean"] for t in seq_lens]
    compute_ms = [compute[str(t)]["mean"] for t in seq_lens]
    ratios = [t / c for t, c in zip(transfer_ms, compute_ms)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: grouped bar chart (log scale)
    x = np.arange(len(seq_lens))
    width = 0.35

    bars1 = ax1.bar(x - width/2, transfer_ms, width, color=COLORS["transfer"],
                    label="CPU→GPU prefetch (28 layers)", alpha=0.85)
    bars2 = ax1.bar(x + width/2, compute_ms, width, color=COLORS["compute"],
                    label="Attention compute (1 layer)", alpha=0.85)

    ax1.set_yscale("log")
    ax1.set_xlabel("Sequence Length (tokens)")
    ax1.set_ylabel("Time (ms, log scale)")
    ax1.set_title("Transfer Latency vs Compute Time")
    ax1.set_xticks(x)
    ax1.set_xticklabels(seq_lens)
    ax1.legend()

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h * 1.2, f"{h:.1f}",
                 ha="center", va="bottom", fontsize=8, color=COLORS["transfer"])
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h * 0.5, f"{h:.3f}",
                 ha="center", va="top", fontsize=8, color=COLORS["compute"])

    # Right: ratio chart
    ax2.bar([str(s) for s in seq_lens], ratios, color="#6A1B9A", alpha=0.8)
    for i, (s, r) in enumerate(zip(seq_lens, ratios)):
        ax2.text(i, r + 20, f"{r:.0f}x", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color="#6A1B9A")

    ax2.set_xlabel("Sequence Length (tokens)")
    ax2.set_ylabel("Transfer / Compute Ratio")
    ax2.set_title("Overhead Ratio\n(how many x slower is transfer?)")

    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_transfer_vs_compute.png"), dpi=200)
    plt.savefig(os.path.join(output_dir, "fig_transfer_vs_compute.pdf"))
    plt.close()
    print("  Saved fig_transfer_vs_compute")


def fig4_single_vs_all_layers(data, output_dir):
    """Show per-layer overhead: single layer is fast, 28 layers add up."""
    single = data["single_layer"]
    all_l = data["all_layers"]
    tokens = sorted([int(k) for k in single.keys()])

    single_c2g = [single[str(t)]["cpu_to_gpu_ms"]["mean"] for t in tokens]
    all_c2g = [all_l[str(t)]["cpu_to_gpu_ms"]["mean"] for t in tokens]
    # Theoretical 28x single
    theoretical_28x = [s * 28 for s in single_c2g]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(tokens, single_c2g, "^-", color=COLORS["single"], linewidth=2,
            markersize=6, label="Single layer (CPU→GPU)")
    ax.plot(tokens, all_c2g, "s-", color=COLORS["all"], linewidth=2,
            markersize=6, label="All 28 layers (CPU→GPU)")
    ax.plot(tokens, theoretical_28x, "--", color=COLORS["single"], linewidth=1.5,
            alpha=0.4, label="28 × single layer (theoretical)")

    # Annotations for scaling
    for t in [256, 1024]:
        if str(t) in all_l:
            s = single[str(t)]["cpu_to_gpu_ms"]["mean"]
            a = all_l[str(t)]["cpu_to_gpu_ms"]["mean"]
            ratio = a / s
            ax.annotate(f"{ratio:.0f}x", (t, a), textcoords="offset points",
                        xytext=(12, -5), fontsize=9, fontweight="bold",
                        color=COLORS["all"])

    ax.set_xlabel("Number of Tokens Transferred")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Single Layer vs All Layers Transfer\n(CPU→GPU prefetch, fp16)")
    ax.legend(loc="upper left")
    ax.set_xlim(-50, 1500)

    plt.savefig(os.path.join(output_dir, "fig_single_vs_all_layers.png"), dpi=200)
    plt.savefig(os.path.join(output_dir, "fig_single_vs_all_layers.pdf"))
    plt.close()
    print("  Saved fig_single_vs_all_layers")


def fig5_practical_scenarios(data, output_dir):
    """Map latency to practical hierarchy scenarios from the algorithm experiments."""
    all_l = data["all_layers"]

    # Interpolate for specific token counts
    tokens = sorted([int(k) for k in all_l.keys()])
    c2g = [all_l[str(t)]["cpu_to_gpu_ms"]["mean"] for t in tokens]
    g2c = [all_l[str(t)]["gpu_to_cpu_ms"]["mean"] for t in tokens]

    scenarios = [
        ("Eviction interval\n(64 tokens/64 steps)", 64, "Every 64 decode steps"),
        ("Sweet spot\n(3% evict ≈ 43 tok)", 43, "Best accuracy tradeoff"),
        ("5% evict\n(≈71 tokens)", 71, "56% accuracy"),
        ("10% evict\n(≈142 tokens)", 142, "34% accuracy"),
        ("50% HBM offload\n(≈600 tokens)", 600, "Full tier swap"),
        ("Full chain\n(1422 tokens)", 1422, "Worst case"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Interpolate latencies
    scenario_labels = []
    prefetch_times = []
    offload_times = []
    for label, n_tok, desc in scenarios:
        c2g_val = np.interp(n_tok, tokens, c2g)
        g2c_val = np.interp(n_tok, tokens, g2c)
        scenario_labels.append(label)
        prefetch_times.append(c2g_val)
        offload_times.append(g2c_val)

    x = np.arange(len(scenarios))
    width = 0.35
    bars1 = ax.bar(x - width/2, offload_times, width, color=COLORS["g2c"],
                   label="GPU→CPU offload", alpha=0.85)
    bars2 = ax.bar(x + width/2, prefetch_times, width, color=COLORS["c2g"],
                   label="CPU→GPU prefetch", alpha=0.85)

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}ms",
                ha="center", va="bottom", fontsize=8, color=COLORS["g2c"])
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}ms",
                ha="center", va="bottom", fontsize=8, color=COLORS["c2g"])

    # Acceptable overhead threshold
    ax.axhline(y=5, color="green", linestyle="--", alpha=0.5,
               label="< 5ms target (acceptable overhead)")

    ax.set_xlabel("")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Transfer Cost for Practical Hierarchy Scenarios\n(RTX 5080, PCIe Gen5 x16, DeepSeek-R1-7B KV config)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=9)
    ax.legend(loc="upper left", fontsize=9)

    plt.savefig(os.path.join(output_dir, "fig_practical_scenarios.png"), dpi=200)
    plt.savefig(os.path.join(output_dir, "fig_practical_scenarios.pdf"))
    plt.close()
    print("  Saved fig_practical_scenarios")


def main():
    results_path = "results/system/pcie_latency_results.json"
    output_dir = "results/system"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading results...")
    data = load_results(results_path)
    print(f"GPU: {data['gpu']}")
    print(f"Generating figures...\n")

    fig1_latency_vs_tokens(data, output_dir)
    fig2_bandwidth(data, output_dir)
    fig3_transfer_vs_compute(data, output_dir)
    fig4_single_vs_all_layers(data, output_dir)
    fig5_practical_scenarios(data, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
