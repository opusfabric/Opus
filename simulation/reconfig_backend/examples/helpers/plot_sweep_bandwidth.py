import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import os
import sys
import re


def parse_results(filepath):
    """Parse results_for_sheet_import.txt.

    Returns
    -------
    analytical : int   – last sys finish time (col 2) for Analytical
    baseline   : int   – last sys finish time (col 2) for Baseline
    reconfig_realtime   : dict[str, int]  – latency_str -> col 3 value (1st row per latency)
    reconfig_preplanned : dict[str, int]  – latency_str -> col 3 value (2nd row per latency)
    """
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
            analytical = int(parts[1])  # second column
        elif section == "baseline":
            baseline = int(parts[1])  # second column
        elif section == "reconfigurable":
            latency_str = parts[0]
            value = int(parts[2])  # third column

            if latency_str not in reconfig_count:
                reconfig_count[latency_str] = 0

            if reconfig_count[latency_str] == 0:
                reconfig_realtime[latency_str] = value
            else:
                reconfig_preplanned[latency_str] = value

            reconfig_count[latency_str] += 1

    return analytical, baseline, reconfig_realtime, reconfig_preplanned


def find_bandwidth_variants(base_folder):
    """Find all bandwidth variants of a base folder.

    Returns dict: bandwidth (float) -> folder_path
    """
    base_dir = os.path.dirname(base_folder)
    base_name = os.path.basename(base_folder)

    # Remove any existing BW suffix to get the true base name
    bw_pattern = re.compile(r"^(.+?)(?:_(\d+(?:\.\d+)?)BW)?$")
    match = bw_pattern.match(base_name)
    if match:
        true_base = match.group(1)
    else:
        true_base = base_name

    variants = {}

    # Check if the base folder (100GB) exists
    base_100gb = os.path.join(base_dir, true_base)
    if os.path.isdir(base_100gb):
        results_file = os.path.join(base_100gb, "results_for_sheet_import.txt")
        if os.path.isfile(results_file):
            variants[100.0] = base_100gb

    # Look for BW-suffixed variants
    if os.path.isdir(base_dir):
        for entry in os.listdir(base_dir):
            if entry.startswith(true_base + "_") and "BW" in entry:
                match = re.match(rf"^{re.escape(true_base)}_(\d+(?:\.\d+)?)BW$", entry)
                if match:
                    bw = float(match.group(1))
                    folder_path = os.path.join(base_dir, entry)
                    results_file = os.path.join(folder_path, "results_for_sheet_import.txt")
                    if os.path.isfile(results_file):
                        variants[bw] = folder_path

    return variants


def plot_bandwidth_sweep(base_folder, latency_str="10 ms", output_path=None):
    """Plot performance vs bandwidth at a fixed reconfiguration latency."""

    variants = find_bandwidth_variants(base_folder)

    if not variants:
        print(f"Error: No bandwidth variants found for {base_folder}")
        return

    # Sort by bandwidth (in GB), convert to Gbps for plotting
    bandwidths_gb = sorted(variants.keys())
    bandwidths = [bw * 8 for bw in bandwidths_gb]  # Convert GB to Gbps

    print(f"Plotting at reconfiguration latency: {latency_str}")
    print(f"Bandwidths found (GB -> Gbps): {dict(zip(bandwidths_gb, bandwidths))}")

    # First pass: collect raw values and find minimum step latency
    raw_data = []
    print(f"\nData at {latency_str}:")
    print(f"{'BW (Gbps)':<12} {'Analytical':<15} {'Baseline':<15} {'Opus':<15} {'Opus+Prov':<15}")
    print("-" * 72)

    for bw_gb in bandwidths_gb:
        folder = variants[bw_gb]
        results_path = os.path.join(folder, "results_for_sheet_import.txt")
        analytical, baseline, reconfig_rt, reconfig_pp = parse_results(results_path)

        if latency_str not in reconfig_rt:
            print(f"Warning: {latency_str} not found in {folder}, skipping")
            continue

        bw_gbps = bw_gb * 8
        print(f"{bw_gbps:<12.1f} {analytical:<15} {baseline:<15} {reconfig_rt[latency_str]:<15} {reconfig_pp[latency_str]:<15}")

        raw_data.append({
            "analytical": analytical,
            "baseline": baseline,
            "opus": reconfig_rt[latency_str],
            "provision": reconfig_pp[latency_str],
        })

    print()

    # Find minimum step latency across all bandwidths (best case = highest BW analytical)
    min_latency = min(d["analytical"] for d in raw_data)

    # Normalize all values to the minimum latency
    opus_values = [d["opus"] / min_latency for d in raw_data]
    provision_values = [d["provision"] / min_latency for d in raw_data]
    eps_values = [d["analytical"] / min_latency for d in raw_data]
    ideal_values = [d["baseline"] / min_latency for d in raw_data]

    # ── Plot ───────────────────────────────────────────────────────────────
    matplotlib.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(5, 3))

    # Use indices for equal spacing on x-axis
    x_indices = list(range(len(bandwidths)))

    # Opus
    ax.plot(
        x_indices, opus_values,
        color="#1f77b4", marker="s", linestyle="-",
        label="Opus", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#1f77b4", markeredgewidth=1.5,
    )
    # Opus+Provision
    ax.plot(
        x_indices, provision_values,
        color="#ff7f0e", marker="x", linestyle="--",
        label="Opus+Provision", markersize=8, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#ff7f0e", markeredgewidth=1.5,
    )
    # EPS (Analytical) – constant at 1.0
    ax.plot(
        x_indices, eps_values,
        color="#d62728", marker="v", linestyle="-",
        label="EPS", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#d62728", markeredgewidth=1.5,
    )
    # Ideal (Baseline)
    ax.plot(
        x_indices, ideal_values,
        color="#9467bd", marker="^", linestyle="--",
        label="Ideal", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#9467bd", markeredgewidth=1.5,
    )

    ax.set_xlabel("Scale-Out Bandwidth (Gbps)")
    ax.set_ylabel("Norm. Step Latency")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{int(bw)}" for bw in bandwidths])
    ax.set_ylim(bottom=0.99)
    ax.legend(loc="upper right", framealpha=0.9)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is None:
        output_path = os.path.join(os.path.dirname(base_folder), "bandwidth_sweep_plot.pdf")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {output_path}")


def plot_bandwidth_sweep_raw(base_folder, latency_str="10 ms", output_path=None):
    """Plot raw runtimes (in seconds) vs bandwidth at a fixed reconfiguration latency."""

    variants = find_bandwidth_variants(base_folder)

    if not variants:
        print(f"Error: No bandwidth variants found for {base_folder}")
        return

    # Sort by bandwidth (in GB), convert to Gbps for plotting
    bandwidths_gb = sorted(variants.keys())
    bandwidths = [bw * 8 for bw in bandwidths_gb]  # Convert GB to Gbps

    # Collect data for each bandwidth (convert ns to seconds)
    opus_values = []
    provision_values = []
    eps_values = []
    ideal_values = []

    for bw_gb in bandwidths_gb:
        folder = variants[bw_gb]
        results_path = os.path.join(folder, "results_for_sheet_import.txt")
        analytical, baseline, reconfig_rt, reconfig_pp = parse_results(results_path)

        if latency_str not in reconfig_rt:
            print(f"Warning: {latency_str} not found in {folder}, skipping")
            continue

        # Convert nanoseconds to seconds
        opus_values.append(reconfig_rt[latency_str] / 1e9)
        provision_values.append(reconfig_pp[latency_str] / 1e9)
        eps_values.append(analytical / 1e9)
        ideal_values.append(baseline / 1e9)

    # ── Plot ───────────────────────────────────────────────────────────────
    matplotlib.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(5, 3))

    # Use indices for equal spacing on x-axis
    x_indices = list(range(len(bandwidths)))

    # Opus
    ax.plot(
        x_indices, opus_values,
        color="#1f77b4", marker="s", linestyle="-",
        label="Opus", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#1f77b4", markeredgewidth=1.5,
    )
    # Opus+Provision
    ax.plot(
        x_indices, provision_values,
        color="#ff7f0e", marker="x", linestyle="--",
        label="Opus+Provision", markersize=8, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#ff7f0e", markeredgewidth=1.5,
    )
    # EPS (Analytical)
    ax.plot(
        x_indices, eps_values,
        color="#d62728", marker="v", linestyle="-",
        label="EPS", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#d62728", markeredgewidth=1.5,
    )
    # Ideal (Baseline)
    ax.plot(
        x_indices, ideal_values,
        color="#9467bd", marker="^", linestyle="--",
        label="Ideal", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#9467bd", markeredgewidth=1.5,
    )

    ax.set_xlabel("Scale-Out Bandwidth (Gbps)")
    ax.set_ylabel("Step Latency (s)")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{int(bw)}" for bw in bandwidths])
    ax.legend(loc="upper right", framealpha=0.9)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is None:
        output_path = os.path.join(os.path.dirname(base_folder), "bandwidth_sweep_raw.pdf")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {output_path}")


def plot_bandwidth_sweep_normalized(base_folder, latency_str="10 ms", output_path=None):
    """Plot performance (%) vs bandwidth at a fixed reconfiguration latency."""

    variants = find_bandwidth_variants(base_folder)

    if not variants:
        print(f"Error: No bandwidth variants found for {base_folder}")
        return

    # Sort by bandwidth (in GB), convert to Gbps for plotting
    bandwidths_gb = sorted(variants.keys())
    bandwidths = [bw * 8 for bw in bandwidths_gb]  # Convert GB to Gbps

    # First pass: collect raw values
    raw_data = []
    for bw_gb in bandwidths_gb:
        folder = variants[bw_gb]
        results_path = os.path.join(folder, "results_for_sheet_import.txt")
        analytical, baseline, reconfig_rt, reconfig_pp = parse_results(results_path)

        if latency_str not in reconfig_rt:
            print(f"Warning: {latency_str} not found in {folder}, skipping")
            continue

        raw_data.append({
            "analytical": analytical,
            "baseline": baseline,
            "opus": reconfig_rt[latency_str],
            "provision": reconfig_pp[latency_str],
        })

    # Find minimum step latency across all bandwidths (best case)
    min_latency = min(d["analytical"] for d in raw_data)

    # Performance: min_latency / time * 100 (so best analytical = 100%)
    opus_values = [min_latency / d["opus"] * 100 for d in raw_data]
    provision_values = [min_latency / d["provision"] * 100 for d in raw_data]
    eps_values = [min_latency / d["analytical"] * 100 for d in raw_data]
    ideal_values = [min_latency / d["baseline"] * 100 for d in raw_data]

    # ── Plot ───────────────────────────────────────────────────────────────
    matplotlib.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(5, 3))

    # Use indices for equal spacing on x-axis
    x_indices = list(range(len(bandwidths)))

    # Opus
    ax.plot(
        x_indices, opus_values,
        color="#1f77b4", marker="s", linestyle="-",
        label="Opus", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#1f77b4", markeredgewidth=1.5,
    )
    # Opus+Provision
    ax.plot(
        x_indices, provision_values,
        color="#ff7f0e", marker="x", linestyle="--",
        label="Opus+Provision", markersize=8, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#ff7f0e", markeredgewidth=1.5,
    )
    # EPS (Analytical) – always 100%
    ax.plot(
        x_indices, eps_values,
        color="#d62728", marker="v", linestyle="-",
        label="EPS", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#d62728", markeredgewidth=1.5,
    )
    # Ideal (Baseline)
    ax.plot(
        x_indices, ideal_values,
        color="#9467bd", marker="^", linestyle="--",
        label="Ideal", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#9467bd", markeredgewidth=1.5,
    )

    ax.set_xlabel("Scale-Out Bandwidth (Gbps)")
    ax.set_ylabel("Performance (%)")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{int(bw)}" for bw in bandwidths])
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.legend(loc="lower right", framealpha=0.9)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is None:
        output_path = os.path.join(os.path.dirname(base_folder), "bandwidth_sweep_normalized.pdf")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot bandwidth sweep results at fixed reconfig latency")
    parser.add_argument("folder_path", help="Path to the base results folder (100GB version)")
    parser.add_argument("--latency", default="10 ms", help="Reconfiguration latency to use (default: '10 ms')")
    args = parser.parse_args()

    folder_path = os.path.abspath(args.folder_path)
    if not os.path.isdir(folder_path):
        print(f"Error: folder not found: {folder_path}")
        sys.exit(1)

    plot_bandwidth_sweep(folder_path, latency_str=args.latency)
    plot_bandwidth_sweep_raw(folder_path, latency_str=args.latency)
    plot_bandwidth_sweep_normalized(folder_path, latency_str=args.latency)
