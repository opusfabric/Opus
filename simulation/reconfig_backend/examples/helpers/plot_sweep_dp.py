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


def find_dp_variants(base_folder, target_bw=None):
    """Find all DP variants with same PP, TP, and other matching parameters.

    Supports two folder naming patterns:
    1. Full format: {prefix}_dp{dp}_pp{pp}_tp{tp}_batch_{batch}_mb{mb}_{stack}stack_seq{seq}[_{bw}BW]
    2. Simple format: {prefix}_dp{dp}_pp{pp}_tp{tp}_{bw}BW (e.g., gb200_llama_405B_dp16_pp8_tp32_100BW)

    Returns tuple: (dict: dp (int) -> folder_path, pp (int), tp (int))
    """
    base_dir = os.path.dirname(base_folder)
    base_name = os.path.basename(base_folder)

    # Pattern 1: Full format with batch, mb, stack, seq
    pattern_full = re.compile(
        r"^(.*?)_dp(\d+)_pp(\d+)_tp(\d+)_batch_(\d+)_mb(-?\d+)_(\d+)stack_seq(\d+)(?:_(\d+(?:\.\d+)?)BW)?$"
    )
    # Pattern 2: Simple format (e.g., gb200_llama_405B_dp16_pp8_tp32_100BW)
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
        bw = match_full.group(9)  # May be None for 100GB
    elif match_simple:
        folder_type = "simple"
        prefix = match_simple.group(1)
        pp = match_simple.group(3)
        tp = match_simple.group(4)
        bw = match_simple.group(5)
        batch = mb = stack = seq = None
    else:
        print(f"Error: Could not parse folder name: {base_name}")
        return {}, 1, 1

    pp_int = int(pp)
    tp_int = int(tp)

    # Use target_bw if specified, otherwise use the BW from the folder name
    if target_bw is not None:
        # Format BW string to match folder naming (no trailing .0)
        if target_bw == int(target_bw):
            bw = str(int(target_bw))
        else:
            bw = str(target_bw)

    # For 100GB, check both no suffix (None) and "_100BW" suffix ("100")
    bw_candidates = [bw]
    if bw == "100":
        bw_candidates = [None, "100"]
    elif bw is None:
        bw_candidates = [None, "100"]

    variants = {}

    if os.path.isdir(base_dir):
        for entry in os.listdir(base_dir):
            # Try to match with the same pattern type as the base folder
            if folder_type == "full":
                entry_match = pattern_full.match(entry)
                if not entry_match:
                    continue

                # Check if all parameters except DP match
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
                        print(f"  Skipping {entry}: no results file")

            else:  # folder_type == "simple"
                entry_match = pattern_simple.match(entry)
                if not entry_match:
                    continue

                # Check if all parameters except DP match
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
                    else:
                        print(f"  Skipping {entry}: no results file")

    if not variants:
        if folder_type == "full":
            print(f"Debug: Looking for prefix='{prefix}', pp={pp}, tp={tp}, batch={batch}, mb={mb}, stack={stack}, seq={seq}, bw={bw_candidates}")
        else:
            print(f"Debug: Looking for prefix='{prefix}', pp={pp}, tp={tp}, bw={bw_candidates}")

    return variants, pp_int, tp_int


def plot_dp_sweep(base_folder, latency_str="10 ms", target_bw=None, output_path=None, use_num_gpu=False):
    """Plot performance vs DP at a fixed reconfiguration latency and bandwidth."""

    variants, pp, tp = find_dp_variants(base_folder, target_bw=target_bw)

    if not variants:
        print(f"Error: No DP variants found for {base_folder}")
        return

    # Sort by DP
    dp_values = sorted(variants.keys())

    bw_label = f"{target_bw}GB" if target_bw else "100GB"
    print(f"Plotting at reconfiguration latency: {latency_str}, bandwidth: {bw_label}")
    print(f"DP values found: {dp_values}")
    if use_num_gpu:
        print(f"Using num GPUs (DP × PP × TP), PP={pp}, TP={tp}")

    # First pass: collect raw values
    raw_data = []
    print(f"\nData at {latency_str}:")
    print(f"{'DP':<8} {'Analytical':<15} {'Baseline':<15} {'Opus':<15} {'Opus+Prov':<15}")
    print("-" * 68)

    for dp in dp_values:
        folder = variants[dp]
        results_path = os.path.join(folder, "results_for_sheet_import.txt")
        analytical, baseline, reconfig_rt, reconfig_pp = parse_results(results_path)

        if latency_str not in reconfig_rt:
            print(f"Warning: {latency_str} not found in {folder}, skipping")
            continue

        print(f"{dp:<8} {analytical:<15} {baseline:<15} {reconfig_rt[latency_str]:<15} {reconfig_pp[latency_str]:<15}")

        raw_data.append({
            "dp": dp,
            "analytical": analytical,
            "baseline": baseline,
            "opus": reconfig_rt[latency_str],
            "provision": reconfig_pp[latency_str],
        })

    print()

    if not raw_data:
        print("No valid data points found")
        return

    # Find minimum step latency across all DP values (best case)
    min_latency = min(d["analytical"] for d in raw_data)

    # Normalize all values to the minimum latency
    opus_values = [d["opus"] / min_latency for d in raw_data]
    provision_values = [d["provision"] / min_latency for d in raw_data]
    eps_values = [d["analytical"] / min_latency for d in raw_data]
    ideal_values = [d["baseline"] / min_latency for d in raw_data]
    dp_plot = [d["dp"] for d in raw_data]

    # ── Plot ───────────────────────────────────────────────────────────────
    matplotlib.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(5, 3))

    # Use indices for equal spacing on x-axis
    x_indices = list(range(len(dp_plot)))

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

    if use_num_gpu:
        ax.set_xlabel("Number of GPUs")
        ax.set_xticklabels([str(dp * pp * tp) for dp in dp_plot])
    else:
        ax.set_xlabel("Data Parallelism (DP)")
        ax.set_xticklabels([str(dp) for dp in dp_plot])
    ax.set_ylabel("Norm. Step Latency")
    ax.set_xticks(x_indices)
    ax.set_ylim(bottom=0.99)
    ax.legend(loc="upper right", framealpha=0.9)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is None:
        suffix = "_gpu" if use_num_gpu else ""
        output_path = os.path.join(os.path.dirname(base_folder), f"dp_sweep_plot{suffix}.pdf")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {output_path}")


def plot_dp_sweep_raw(base_folder, latency_str="10 ms", target_bw=None, output_path=None, use_num_gpu=False):
    """Plot raw runtimes (in seconds) vs DP."""

    variants, pp, tp = find_dp_variants(base_folder, target_bw=target_bw)

    if not variants:
        print(f"Error: No DP variants found for {base_folder}")
        return

    dp_values = sorted(variants.keys())

    # Collect data (convert ns to seconds)
    raw_data = []
    for dp in dp_values:
        folder = variants[dp]
        results_path = os.path.join(folder, "results_for_sheet_import.txt")
        analytical, baseline, reconfig_rt, reconfig_pp = parse_results(results_path)

        if latency_str not in reconfig_rt:
            continue

        raw_data.append({
            "dp": dp,
            "analytical": analytical / 1e9,
            "baseline": baseline / 1e9,
            "opus": reconfig_rt[latency_str] / 1e9,
            "provision": reconfig_pp[latency_str] / 1e9,
        })

    if not raw_data:
        print("No valid data points found")
        return

    opus_values = [d["opus"] for d in raw_data]
    provision_values = [d["provision"] for d in raw_data]
    eps_values = [d["analytical"] for d in raw_data]
    ideal_values = [d["baseline"] for d in raw_data]
    dp_plot = [d["dp"] for d in raw_data]

    # ── Plot ───────────────────────────────────────────────────────────────
    matplotlib.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(5, 3))

    x_indices = list(range(len(dp_plot)))

    ax.plot(x_indices, opus_values, color="#1f77b4", marker="s", linestyle="-",
            label="Opus", markersize=7, linewidth=1.5,
            markerfacecolor="none", markeredgecolor="#1f77b4", markeredgewidth=1.5)
    ax.plot(x_indices, provision_values, color="#ff7f0e", marker="x", linestyle="--",
            label="Opus+Provision", markersize=8, linewidth=1.5,
            markerfacecolor="none", markeredgecolor="#ff7f0e", markeredgewidth=1.5)
    ax.plot(x_indices, eps_values, color="#d62728", marker="v", linestyle="-",
            label="EPS", markersize=7, linewidth=1.5,
            markerfacecolor="none", markeredgecolor="#d62728", markeredgewidth=1.5)
    ax.plot(x_indices, ideal_values, color="#9467bd", marker="^", linestyle="--",
            label="Ideal", markersize=7, linewidth=1.5,
            markerfacecolor="none", markeredgecolor="#9467bd", markeredgewidth=1.5)

    if use_num_gpu:
        ax.set_xlabel("Number of GPUs")
        ax.set_xticklabels([str(dp * pp * tp) for dp in dp_plot])
    else:
        ax.set_xlabel("Data Parallelism (DP)")
        ax.set_xticklabels([str(dp) for dp in dp_plot])
    ax.set_ylabel("Step Latency (s)")
    ax.set_xticks(x_indices)
    ax.legend(loc="upper right", framealpha=0.9)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is None:
        suffix = "_gpu" if use_num_gpu else ""
        output_path = os.path.join(os.path.dirname(base_folder), f"dp_sweep_raw{suffix}.pdf")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot DP sweep results at fixed reconfig latency and bandwidth")
    parser.add_argument("folder_path", help="Path to one of the results folders")
    parser.add_argument("--latency", default="10 ms", help="Reconfiguration latency to use (default: '10 ms')")
    parser.add_argument("--bw", type=float, default=None, help="Bandwidth in GB (default: use folder's BW, or 100)")
    parser.add_argument("--num-gpu", action="store_true", help="Use number of GPUs (DP × PP × TP) for x-axis instead of DP")
    args = parser.parse_args()

    folder_path = os.path.abspath(args.folder_path)
    if not os.path.isdir(folder_path):
        print(f"Error: folder not found: {folder_path}")
        sys.exit(1)

    plot_dp_sweep(folder_path, latency_str=args.latency, target_bw=args.bw, use_num_gpu=args.num_gpu)
    plot_dp_sweep_raw(folder_path, latency_str=args.latency, target_bw=args.bw, use_num_gpu=args.num_gpu)
