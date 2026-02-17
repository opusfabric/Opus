import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
from seaborn import color_palette
import os
import sys
import re

# ══════════════════════════════════════════════════════════════════════════════
# Plot styling
# ══════════════════════════════════════════════════════════════════════════════

PERF_COLORS = {
    "opus": "#1f77b4",
    "provision": "#ff7f0e",
    "eps": "#d62728",
    "ideal": "#9467bd",
}

PERF_MARKERS = {
    "opus": "s",
    "provision": "x",
    "eps": "v",
    "ideal": "^",
}

PERF_LINESTYLES = {
    "opus": "-",
    "provision": "--",
    "eps": "-",
    "ideal": "--",
}

TOPOLOGIES = ['EPS Fat-tree', 'EPS Rail-Opt', 'Opus']
TOPOLOGY_COLORS = color_palette("Set2", 3)


# ══════════════════════════════════════════════════════════════════════════════
# DP sweep parsing functions
# ══════════════════════════════════════════════════════════════════════════════

def parse_results(filepath):
    """Parse results_for_sheet_import.txt."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    analytical = None
    baseline = None
    reconfig_realtime = {}
    reconfig_preplanned = {}
    section = None
    reconfig_count = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line == "Analytical":
            section = "analytical"
            continue
        elif line == "Baseline":
            section = "baseline"
            continue
        elif line == "Reconfigurable":
            section = "reconfigurable"
            continue

        parts = line.split("\t")
        if section == "analytical":
            analytical = int(parts[1])
        elif section == "baseline":
            baseline = int(parts[1])
        elif section == "reconfigurable":
            latency_str = parts[0]
            value = int(parts[2])
            if latency_str not in reconfig_count:
                reconfig_count[latency_str] = 0
            if reconfig_count[latency_str] == 0:
                reconfig_realtime[latency_str] = value
            else:
                reconfig_preplanned[latency_str] = value
            reconfig_count[latency_str] += 1

    return analytical, baseline, reconfig_realtime, reconfig_preplanned


def find_dp_variants(base_folder, target_bw=None):
    """Find all DP variants with same PP, TP, and other matching parameters."""
    base_dir = os.path.dirname(base_folder)
    base_name = os.path.basename(base_folder)

    pattern_full = re.compile(
        r"^(.*?)_dp(\d+)_pp(\d+)_tp(\d+)_batch_(\d+)_mb(-?\d+)_(\d+)stack_seq(\d+)(?:_(\d+(?:\.\d+)?)BW)?$"
    )
    pattern_simple = re.compile(
        r"^(.+)_dp(\d+)_pp(\d+)_tp(\d+)_(\d+(?:\.\d+)?)BW$"
    )

    match_full = pattern_full.match(base_name)
    match_simple = pattern_simple.match(base_name)

    if match_full:
        folder_type = "full"
        prefix = match_full.group(1)
        pp = match_full.group(3)
        tp = match_full.group(4)
        batch = match_full.group(5)
        mb = match_full.group(6)
        stack = match_full.group(7)
        seq = match_full.group(8)
        bw = match_full.group(9)
    elif match_simple:
        folder_type = "simple"
        prefix = match_simple.group(1)
        pp = match_simple.group(3)
        tp = match_simple.group(4)
        bw = match_simple.group(5)
        batch = mb = stack = seq = None
    else:
        return {}, 1, 1

    pp_int = int(pp)
    tp_int = int(tp)

    if target_bw is not None:
        bw = str(int(target_bw)) if target_bw == int(target_bw) else str(target_bw)

    bw_candidates = [bw]
    if bw == "100":
        bw_candidates = [None, "100"]
    elif bw is None:
        bw_candidates = [None, "100"]

    variants = {}

    if os.path.isdir(base_dir):
        for entry in os.listdir(base_dir):
            if folder_type == "full":
                entry_match = pattern_full.match(entry)
                if not entry_match:
                    continue
                entry_bw = entry_match.group(9)
                if (entry_match.group(1) == prefix and
                    entry_match.group(3) == pp and
                    entry_match.group(4) == tp and
                    entry_match.group(5) == batch and
                    entry_match.group(6) == mb and
                    entry_match.group(7) == stack and
                    entry_match.group(8) == seq and
                    entry_bw in bw_candidates):
                    dp = int(entry_match.group(2))
                    folder_path = os.path.join(base_dir, entry)
                    results_file = os.path.join(folder_path, "results_for_sheet_import.txt")
                    if os.path.isfile(results_file):
                        variants[dp] = folder_path
            else:
                entry_match = pattern_simple.match(entry)
                if not entry_match:
                    continue
                entry_bw = entry_match.group(5)
                if (entry_match.group(1) == prefix and
                    entry_match.group(3) == pp and
                    entry_match.group(4) == tp and
                    entry_bw in bw_candidates):
                    dp = int(entry_match.group(2))
                    folder_path = os.path.join(base_dir, entry)
                    results_file = os.path.join(folder_path, "results_for_sheet_import.txt")
                    if os.path.isfile(results_file):
                        variants[dp] = folder_path

    return variants, pp_int, tp_int


# ══════════════════════════════════════════════════════════════════════════════
# Cost/Power calculation
# ══════════════════════════════════════════════════════════════════════════════

def get_cost_power_h200(total_server, num_gpu_per_server=8):
    """Cost and power for H200 (8 GPUs per server, 400G)."""
    transceiver_power = 10  # https://www.fs.com/products/110530.html
    switch_port_power = 2524 / 64  # https://www.fs.com/products/149853.html
    ocs_port_power = 75 / 192  # polatis 6000N

    transceiver_cost = 879
    switch_port_cost = 37299 / 64  # https://www.fs.com/products/149853.html
    ocs_port_cost = 520

    total_gpu = total_server * num_gpu_per_server
    num_gpu_ports = total_gpu

    # Fat-tree
    k = int((total_gpu * 4) ** (1/3) + 0.5)
    num_switch_ports_fat_tree = k * (k ** 2 + k ** 2 // 4)

    # Rail-optimized
    k = int((total_server * 4) ** 0.5 + 0.5)
    num_switch_fat_tree_per_rail = total_server // (k // 2) * 2
    num_switch_spine = num_gpu_per_server * total_server // k
    num_switch_ports_rail = k * (num_switch_fat_tree_per_rail * num_gpu_per_server + num_switch_spine)

    # Opus
    num_ocs_ports = (total_server * 2) * num_gpu_per_server

    # H200 uses passive cables - need transceivers on both GPU and switch ports
    costs = (
        switch_port_cost * num_switch_ports_fat_tree + transceiver_cost * (num_gpu_ports + num_switch_ports_fat_tree),
        switch_port_cost * num_switch_ports_rail + transceiver_cost * (num_gpu_ports + num_switch_ports_rail),
        ocs_port_cost * num_ocs_ports + transceiver_cost * num_gpu_ports,
    )
    powers = (
        switch_port_power * num_switch_ports_fat_tree + transceiver_power * (num_gpu_ports + num_switch_ports_fat_tree),
        switch_port_power * num_switch_ports_rail + transceiver_power * (num_gpu_ports + num_switch_ports_rail),
        ocs_port_power * num_ocs_ports + transceiver_power * num_gpu_ports,
    )
    return costs, powers


def get_cost_power_gb200(total_server, num_gpu_per_server=32):
    """Cost and power for GB200 (32 GPUs per server, 800G)."""
    transceiver_power = 16
    switch_port_power = 7000 / 144
    ocs_port_power = 75 / 192

    transceiver_cost = 879
    switch_port_cost = 149939.00 / 144
    ocs_port_cost = 520

    total_gpu = total_server * num_gpu_per_server
    num_gpu_ports = total_gpu

    # Fat-tree
    k = int((total_gpu * 4) ** (1/3) + 0.5)
    num_switch_ports_fat_tree = k * (k ** 2 + k ** 2 // 4)

    # Rail-optimized
    k = int((total_server * 4) ** 0.5 + 0.5)
    num_switch_fat_tree_per_rail = total_server // (k // 2) * 2
    num_switch_spine = num_gpu_per_server * total_server // k
    num_switch_ports_rail = k * (num_switch_fat_tree_per_rail * num_gpu_per_server + num_switch_spine)

    # Opus
    num_ocs_ports = (total_server * 2) * num_gpu_per_server

    costs = (
        switch_port_cost * num_switch_ports_fat_tree + transceiver_cost * num_gpu_ports,
        switch_port_cost * num_switch_ports_rail + transceiver_cost * num_gpu_ports,
        ocs_port_cost * num_ocs_ports + transceiver_cost * num_gpu_ports,
    )
    powers = (
        switch_port_power * num_switch_ports_fat_tree + transceiver_power * num_gpu_ports,
        switch_port_power * num_switch_ports_rail + transceiver_power * num_gpu_ports,
        ocs_port_power * num_ocs_ports + transceiver_power * num_gpu_ports,
    )
    return costs, powers


# ══════════════════════════════════════════════════════════════════════════════
# Combined plotting function
# ══════════════════════════════════════════════════════════════════════════════

def plot_combined(dp_folder_1, dp_folder_2, latency_str="10 ms",
                  hw1_servers=[32, 64, 128, 256], hw1_gpu_per_server=8, hw1_name="H200", hw1_bw=50,
                  hw2_servers=[8, 16, 32, 64], hw2_gpu_per_server=32, hw2_name="GB200", hw2_bw=100,
                  output_path=None):
    """
    Create a 2-column, 3-row figure:
    - Row 1: Performance (step latency) at fixed BW and reconfig latency
    - Row 2: Cost
    - Row 3: Power
    Each column is a different hardware configuration, sharing x-axis (Number of GPUs).
    """

    matplotlib.rcParams.update({"font.size": 18})
    fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex='col')

    bar_width = 0.25

    # ══════════════════════════════════════════════════════════════════════════
    # Process each column (hardware configuration)
    # ══════════════════════════════════════════════════════════════════════════

    hw_configs = [
        (dp_folder_1, hw1_servers, hw1_gpu_per_server, hw1_name, hw1_bw, get_cost_power_h200),
        (dp_folder_2, hw2_servers, hw2_gpu_per_server, hw2_name, hw2_bw, get_cost_power_gb200),
    ]

    for col, (dp_folder, servers, gpu_per_server, hw_name, target_bw, cost_power_func) in enumerate(hw_configs):
        # ── Row 0: Performance - first collect data to determine actual GPU counts ──
        ax_perf = axes[0, col]
        raw_data = []

        if dp_folder and os.path.isdir(dp_folder):
            variants, pp, tp = find_dp_variants(dp_folder, target_bw=target_bw)

            if variants:
                dp_values = sorted(variants.keys())

                for dp in dp_values:
                    folder = variants[dp]
                    results_path = os.path.join(folder, "results_for_sheet_import.txt")
                    analytical, baseline, reconfig_rt, reconfig_pp = parse_results(results_path)

                    if latency_str not in reconfig_rt:
                        continue

                    raw_data.append({
                        "dp": dp,
                        "num_gpu": dp * pp * tp,
                        "analytical": analytical / 1e9,
                        "baseline": baseline / 1e9,
                        "opus": reconfig_rt[latency_str] / 1e9,
                        "provision": reconfig_pp[latency_str] / 1e9,
                    })

        # Derive server counts from actual DP data
        if raw_data:
            num_gpus = np.array([d["num_gpu"] for d in raw_data])
            derived_servers = num_gpus // gpu_per_server
        else:
            num_gpus = np.array(servers) * gpu_per_server
            derived_servers = np.array(servers)

        x_positions = np.arange(len(num_gpus))

        if raw_data:
            # Align with bar chart centers (x_positions + bar_width)
            x_idx = x_positions + bar_width
            ax_perf.plot(x_idx, [d["opus"] for d in raw_data],
                         color=PERF_COLORS["opus"], marker=PERF_MARKERS["opus"],
                         linestyle=PERF_LINESTYLES["opus"], label="Opus", markersize=6, linewidth=1.5,
                         markerfacecolor="none", markeredgecolor=PERF_COLORS["opus"], markeredgewidth=1.5)
            ax_perf.plot(x_idx, [d["provision"] for d in raw_data],
                         color=PERF_COLORS["provision"], marker=PERF_MARKERS["provision"],
                         linestyle=PERF_LINESTYLES["provision"], label="Opus+Provision", markersize=7, linewidth=1.5,
                         markerfacecolor="none", markeredgecolor=PERF_COLORS["provision"], markeredgewidth=1.5)
            ax_perf.plot(x_idx, [d["analytical"] for d in raw_data],
                         color=PERF_COLORS["eps"], marker=PERF_MARKERS["eps"],
                         linestyle=PERF_LINESTYLES["eps"], label="EPS", markersize=6, linewidth=1.5,
                         markerfacecolor="none", markeredgecolor=PERF_COLORS["eps"], markeredgewidth=1.5)
            ax_perf.plot(x_idx, [d["baseline"] for d in raw_data],
                         color=PERF_COLORS["ideal"], marker=PERF_MARKERS["ideal"],
                         linestyle=PERF_LINESTYLES["ideal"], label="Ideal", markersize=6, linewidth=1.5,
                         markerfacecolor="none", markeredgecolor=PERF_COLORS["ideal"], markeredgewidth=1.5)

        ax_perf.set_xticks(x_positions + bar_width)
        ax_perf.set_ylabel("Latency (s)")
        ax_perf.set_title(hw_name, fontsize=18, fontweight='bold')
        ax_perf.grid(True, alpha=0.3)

        # ── Row 1: Cost (use derived server counts) ───────────────────────────
        ax_cost = axes[1, col]
        costs_data = []
        for s in derived_servers:
            costs, _ = cost_power_func(int(s), gpu_per_server)
            costs_data.append(costs)
        costs_data = np.array(costs_data)

        for i, topology in enumerate(TOPOLOGIES):
            ax_cost.bar(x_positions + i * bar_width, costs_data[:, i] / 1e6, bar_width,
                        label=topology, color=TOPOLOGY_COLORS[i])

        ax_cost.set_ylabel("Cost (M $)")
        ax_cost.set_xticks(x_positions + bar_width)
        ax_cost.grid(True, alpha=0.3, axis='y')

        # ── Row 2: Power (use derived server counts) ──────────────────────────
        ax_power = axes[2, col]
        powers_data = []
        for s in derived_servers:
            _, powers = cost_power_func(int(s), gpu_per_server)
            powers_data.append(powers)
        powers_data = np.array(powers_data)

        for i, topology in enumerate(TOPOLOGIES):
            ax_power.bar(x_positions + i * bar_width, powers_data[:, i] / 1000, bar_width,
                         label=topology, color=TOPOLOGY_COLORS[i])

        ax_power.set_ylabel("Power (kW)")
        ax_power.set_xlabel("Number of GPUs")
        ax_power.set_xticks(x_positions + bar_width)
        ax_power.set_xticklabels(num_gpus)
        ax_power.grid(True, alpha=0.3, axis='y')

    # ══════════════════════════════════════════════════════════════════════════
    # Shared legend on top (two rows)
    # ══════════════════════════════════════════════════════════════════════════

    handles_perf, labels_perf = axes[0, 0].get_legend_handles_labels()
    handles_topo, labels_topo = axes[1, 0].get_legend_handles_labels()

    # First row: performance legends (4 items)
    fig.legend(handles_perf, labels_perf, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.04), fontsize=18)
    # Second row: topology legends (3 items)
    fig.legend(handles_topo, labels_topo, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 0.975), fontsize=18)

    plt.tight_layout(rect=(0, 0, 1, 0.90), h_pad=0)

    if output_path is None:
        output_path = "combined_perf_cost_power.pdf"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combined Performance/Cost/Power plot for two hardware configs")
    parser.add_argument("dp_folder_1", help="Path to DP sweep folder for hardware 1 (e.g., H200)")
    parser.add_argument("dp_folder_2", help="Path to DP sweep folder for hardware 2 (e.g., GB200)")
    parser.add_argument("--latency", default="10 ms", help="Reconfiguration latency (default: '10 ms')")
    parser.add_argument("--bw1", type=float, default=50, help="Bandwidth in GB for H200 (default: 50)")
    parser.add_argument("--bw2", type=float, default=100, help="Bandwidth in GB for GB200 (default: 100)")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()

    folder_1 = os.path.abspath(args.dp_folder_1)
    folder_2 = os.path.abspath(args.dp_folder_2)

    plot_combined(folder_1, folder_2, latency_str=args.latency,
                  hw1_bw=args.bw1, hw2_bw=args.bw2, output_path=args.output)
