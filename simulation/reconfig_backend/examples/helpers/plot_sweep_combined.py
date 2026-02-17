import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import os
import sys
import re


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


def latency_str_to_seconds(s):
    """Convert e.g. '0 ms', '10 ms', '1 s' -> float seconds."""
    parts = s.strip().split()
    value = float(parts[0])
    unit = parts[1]
    if unit == "ms":
        return value / 1e3
    elif unit == "us":
        return value / 1e6
    elif unit == "s":
        return value
    return value


def find_bandwidth_variants(base_folder):
    """Find all bandwidth variants of a base folder."""
    base_dir = os.path.dirname(base_folder)
    base_name = os.path.basename(base_folder)

    bw_pattern = re.compile(r"^(.+?)(?:_(\d+(?:\.\d+)?)BW)?$")
    match = bw_pattern.match(base_name)
    if match:
        true_base = match.group(1)
    else:
        true_base = base_name

    variants = {}

    base_100gb = os.path.join(base_dir, true_base)
    if os.path.isdir(base_100gb):
        results_file = os.path.join(base_100gb, "results_for_sheet_import.txt")
        if os.path.isfile(results_file):
            variants[100.0] = base_100gb

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


# Plot styling constants
COLORS = {
    "opus": "#1f77b4",
    "provision": "#ff7f0e",
    "eps": "#d62728",
    "ideal": "#9467bd",
}

MARKERS = {
    "opus": "s",
    "provision": "x",
    "eps": "v",
    "ideal": "^",
}

LINESTYLES = {
    "opus": "-",
    "provision": "--",
    "eps": "-",
    "ideal": "--",
}


def plot_combined(folder_path, bw_latency_str="10 ms", output_path=None):
    """Plot reconfig latency and bandwidth sweep side by side with shared legend on top."""

    matplotlib.rcParams.update({"font.size": 14})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))

    # ── Left plot: Reconfiguration Latency ────────────────────────────────
    results_path = os.path.join(folder_path, "results_for_sheet_import.txt")
    analytical, baseline, reconfig_rt, reconfig_pp = parse_results(results_path)

    latencies_str = sorted(reconfig_rt.keys(), key=latency_str_to_seconds)
    latencies_str = [l for l in latencies_str if latency_str_to_seconds(l) <= 1.0]
    latencies_sec = [latency_str_to_seconds(l) for l in latencies_str]
    latencies_ms = [l * 1e3 for l in latencies_sec]
    x_indices_lat = list(range(len(latencies_str)))

    # Raw values in seconds (convert from ns)
    rt_raw = [reconfig_rt[l] / 1e9 for l in latencies_str]
    pp_raw = [reconfig_pp[l] / 1e9 for l in latencies_str]
    analytical_s = analytical / 1e9
    baseline_s = baseline / 1e9

    # Opus
    ax1.plot(
        x_indices_lat, rt_raw,
        color=COLORS["opus"], marker=MARKERS["opus"], linestyle=LINESTYLES["opus"],
        label="Opus", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor=COLORS["opus"], markeredgewidth=1.5,
    )
    # Opus+Provision
    ax1.plot(
        x_indices_lat, pp_raw,
        color=COLORS["provision"], marker=MARKERS["provision"], linestyle=LINESTYLES["provision"],
        label="Opus+Provision", markersize=8, linewidth=1.5,
        markerfacecolor="none", markeredgecolor=COLORS["provision"], markeredgewidth=1.5,
    )
    # EPS
    ax1.plot(
        x_indices_lat, [analytical_s] * len(x_indices_lat),
        color=COLORS["eps"], marker=MARKERS["eps"], linestyle=LINESTYLES["eps"],
        label="EPS", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor=COLORS["eps"], markeredgewidth=1.5,
    )
    # Ideal
    ax1.plot(
        x_indices_lat, [baseline_s] * len(x_indices_lat),
        color=COLORS["ideal"], marker=MARKERS["ideal"], linestyle=LINESTYLES["ideal"],
        label="Ideal", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor=COLORS["ideal"], markeredgewidth=1.5,
    )

    ax1.set_xlabel("Reconfiguration Latency (ms)")
    ax1.set_xticks(x_indices_lat)
    ax1.set_xticklabels([f"{int(l)}" if l >= 1 else f"{l:.1f}" for l in latencies_ms])
    ax1.set_ylabel("Step Latency (s)")
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax1.grid(True, alpha=0.3)

    # ── Right plot: Bandwidth Sweep ───────────────────────────────────────
    variants = find_bandwidth_variants(folder_path)

    if variants:
        bandwidths_gb = sorted(variants.keys())
        bandwidths = [bw * 8 for bw in bandwidths_gb]

        raw_data = []
        for bw_gb in bandwidths_gb:
            folder = variants[bw_gb]
            res_path = os.path.join(folder, "results_for_sheet_import.txt")
            anal, base, rec_rt, rec_pp = parse_results(res_path)

            if bw_latency_str not in rec_rt:
                continue

            raw_data.append({
                "analytical": anal,
                "baseline": base,
                "opus": rec_rt[bw_latency_str],
                "provision": rec_pp[bw_latency_str],
            })

        if raw_data:
            # Raw values in seconds (convert from ns)
            opus_values = [d["opus"] / 1e9 for d in raw_data]
            provision_values = [d["provision"] / 1e9 for d in raw_data]
            eps_values = [d["analytical"] / 1e9 for d in raw_data]
            ideal_values = [d["baseline"] / 1e9 for d in raw_data]

            x_indices_bw = list(range(len(bandwidths)))

            ax2.plot(
                x_indices_bw, opus_values,
                color=COLORS["opus"], marker=MARKERS["opus"], linestyle=LINESTYLES["opus"],
                markersize=7, linewidth=1.5,
                markerfacecolor="none", markeredgecolor=COLORS["opus"], markeredgewidth=1.5,
            )
            ax2.plot(
                x_indices_bw, provision_values,
                color=COLORS["provision"], marker=MARKERS["provision"], linestyle=LINESTYLES["provision"],
                markersize=8, linewidth=1.5,
                markerfacecolor="none", markeredgecolor=COLORS["provision"], markeredgewidth=1.5,
            )
            ax2.plot(
                x_indices_bw, eps_values,
                color=COLORS["eps"], marker=MARKERS["eps"], linestyle=LINESTYLES["eps"],
                markersize=7, linewidth=1.5,
                markerfacecolor="none", markeredgecolor=COLORS["eps"], markeredgewidth=1.5,
            )
            ax2.plot(
                x_indices_bw, ideal_values,
                color=COLORS["ideal"], marker=MARKERS["ideal"], linestyle=LINESTYLES["ideal"],
                markersize=7, linewidth=1.5,
                markerfacecolor="none", markeredgecolor=COLORS["ideal"], markeredgewidth=1.5,
            )

            ax2.set_xlabel("Scale-Out Bandwidth (Gbps)")
            ax2.set_ylabel("Step Latency (s)")
            ax2.set_xticks(x_indices_bw)
            ax2.set_xticklabels([f"{int(bw)}" for bw in bandwidths])
            ax2.grid(True, alpha=0.3)

    # ── Shared legend on top ──────────────────────────────────────────────
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout(rect=(0, 0, 1, 0.92))

    if output_path is None:
        output_path = os.path.join(os.path.dirname(folder_path), "sweep_combined_plot.pdf")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot combined reconfig latency and bandwidth sweep")
    parser.add_argument("folder_path", help="Path to the results folder")
    parser.add_argument("--bw-latency", default="10 ms", help="Reconfiguration latency for bandwidth plot (default: '10 ms')")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()

    folder_path = os.path.abspath(args.folder_path)
    if not os.path.isdir(folder_path):
        print(f"Error: folder not found: {folder_path}")
        sys.exit(1)

    plot_combined(folder_path, bw_latency_str=args.bw_latency, output_path=args.output)
