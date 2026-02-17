import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import os
import sys

# ── Configuration ──────────────────────────────────────────────────────────
BATCH = 256
MB = -1
STACK = 96
BW = 50
DP = 4
TP = 8
PP = 4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.dirname(SCRIPT_DIR)


def get_folder_name(dp, pp, tp, batch, mb, stack, seq=4096, bw=None):
    name = f"stg_dp{dp}_pp{pp}_tp{tp}_batch_{batch}_mb{mb}_{stack}stack_seq{seq}"
    if bw is not None:
        name += f"_{bw}BW"
    return name


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


def latency_str_to_seconds(s):
    """Convert e.g. '0 ms', '10 ms', '1 s' → float seconds."""
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


def plot_reconfig_latency(folder_path, title=None, output_path=None):
    results_path = os.path.join(folder_path, "results_for_sheet_import.txt")
    analytical, baseline, reconfig_rt, reconfig_pp = parse_results(results_path)

    # Sort latencies numerically, drop points above 1 second
    latencies_str = sorted(reconfig_rt.keys(), key=latency_str_to_seconds)
    latencies_str = [l for l in latencies_str if latency_str_to_seconds(l) <= 1.0]
    latencies_sec = [latency_str_to_seconds(l) for l in latencies_str]

    # Convert to milliseconds for tick labels
    latencies_ms = [l * 1e3 for l in latencies_sec]

    # Use indices for equal spacing on x-axis
    x_indices = list(range(len(latencies_str)))

    # Normalize: time / analytical (analytical = 1.0)
    rt_norm = [reconfig_rt[l] / analytical for l in latencies_str]
    pp_norm = [reconfig_pp[l] / analytical for l in latencies_str]
    baseline_norm = baseline / analytical

    # ── Plot ───────────────────────────────────────────────────────────────
    matplotlib.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(5, 3))

    # Opus (Reconfigurable, 1st row per latency)
    ax.plot(
        x_indices, rt_norm,
        color="#1f77b4", marker="s", linestyle="-",
        label="Opus", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#1f77b4", markeredgewidth=1.5,
    )
    # Opus w/ Provisioning (Reconfigurable, 2nd row per latency)
    ax.plot(
        x_indices, pp_norm,
        color="#ff7f0e", marker="x", linestyle="--",
        label="Opus+Provision", markersize=8, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#ff7f0e", markeredgewidth=1.5,
    )
    # EPS (Analytical) – constant at 1.0
    ax.plot(
        x_indices, [1.0] * len(x_indices),
        color="#d62728", marker="v", linestyle="-",
        label="EPS", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#d62728", markeredgewidth=1.5,
    )
    # Ideal (Baseline) – constant across latencies
    ax.plot(
        x_indices, [baseline_norm] * len(x_indices),
        color="#9467bd", marker="^", linestyle="--",
        label="Ideal", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#9467bd", markeredgewidth=1.5,
    )

    ax.set_xlabel("Reconfiguration Latency (ms)")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{int(l)}" if l >= 1 else f"{l:.1f}" for l in latencies_ms])
    ax.set_ylabel("Norm. Step Latency")
    ax.set_ylim(bottom=0.9)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.legend(loc="upper left", framealpha=0.9)

    # if title:
    #     ax.set_title(title, fontsize=13)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is None:
        output_path = os.path.join(folder_path, "reconfig_latency_plot.pdf")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {output_path}")


def plot_reconfig_latency_raw(folder_path, title=None, output_path=None):
    """Plot raw (non-normalized) step latency values."""
    results_path = os.path.join(folder_path, "results_for_sheet_import.txt")
    analytical, baseline, reconfig_rt, reconfig_pp = parse_results(results_path)

    # Sort latencies numerically, drop points above 1 second
    latencies_str = sorted(reconfig_rt.keys(), key=latency_str_to_seconds)
    latencies_str = [l for l in latencies_str if latency_str_to_seconds(l) <= 1.0]
    latencies_sec = [latency_str_to_seconds(l) for l in latencies_str]

    # Convert to milliseconds for tick labels
    latencies_ms = [l * 1e3 for l in latencies_sec]

    # Use indices for equal spacing on x-axis
    x_indices = list(range(len(latencies_str)))

    # Raw values (convert to seconds)
    rt_raw = [reconfig_rt[l] / 1e9 for l in latencies_str]  # ns -> s
    pp_raw = [reconfig_pp[l] / 1e9 for l in latencies_str]
    analytical_s = analytical / 1e9
    baseline_s = baseline / 1e9

    # ── Plot ───────────────────────────────────────────────────────────────
    matplotlib.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(5, 3))

    # Opus (Reconfigurable, 1st row per latency)
    ax.plot(
        x_indices, rt_raw,
        color="#1f77b4", marker="s", linestyle="-",
        label="Opus", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#1f77b4", markeredgewidth=1.5,
    )
    # Opus w/ Provisioning (Reconfigurable, 2nd row per latency)
    ax.plot(
        x_indices, pp_raw,
        color="#ff7f0e", marker="x", linestyle="--",
        label="Opus+Provision", markersize=8, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#ff7f0e", markeredgewidth=1.5,
    )
    # EPS (Analytical) – constant
    ax.plot(
        x_indices, [analytical_s] * len(x_indices),
        color="#d62728", marker="v", linestyle="-",
        label="EPS", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#d62728", markeredgewidth=1.5,
    )
    # Ideal (Baseline) – constant across latencies
    ax.plot(
        x_indices, [baseline_s] * len(x_indices),
        color="#9467bd", marker="^", linestyle="--",
        label="Ideal", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#9467bd", markeredgewidth=1.5,
    )

    ax.set_xlabel("Reconfiguration Latency (ms)")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{int(l)}" if l >= 1 else f"{l:.1f}" for l in latencies_ms])
    ax.set_ylabel("Step Latency (s)")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.legend(loc="upper left", framealpha=0.9)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is None:
        output_path = os.path.join(folder_path, "reconfig_latency_raw.pdf")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {output_path}")


def plot_reconfig_latency_normalized(folder_path, title=None, output_path=None):
    results_path = os.path.join(folder_path, "results_for_sheet_import.txt")
    analytical, baseline, reconfig_rt, reconfig_pp = parse_results(results_path)

    # Sort latencies numerically, drop points above 1 second
    latencies_str = sorted(reconfig_rt.keys(), key=latency_str_to_seconds)
    latencies_str = [l for l in latencies_str if latency_str_to_seconds(l) <= 1.0]
    latencies_sec = [latency_str_to_seconds(l) for l in latencies_str]

    # Convert to milliseconds for tick labels
    latencies_ms = [l * 1e3 for l in latencies_sec]

    # Use indices for equal spacing on x-axis
    x_indices = list(range(len(latencies_str)))

    # Performance: analytical = 100%, others = analytical / time * 100
    rt_norm = [analytical / reconfig_rt[l] * 100 for l in latencies_str]
    pp_norm = [analytical / reconfig_pp[l] * 100 for l in latencies_str]
    baseline_norm = analytical / baseline * 100

    # ── Plot ───────────────────────────────────────────────────────────────
    matplotlib.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(5, 3))

    # Opus (Reconfigurable, 1st row per latency)
    ax.plot(
        x_indices, rt_norm,
        color="#1f77b4", marker="s", linestyle="-",
        label="Opus", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#1f77b4", markeredgewidth=1.5,
    )
    # Opus w/ Provisioning (Reconfigurable, 2nd row per latency)
    ax.plot(
        x_indices, pp_norm,
        color="#ff7f0e", marker="x", linestyle="--",
        label="Opus+Provision", markersize=8, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#ff7f0e", markeredgewidth=1.5,
    )
    # EPS (Analytical) – always 100%
    ax.plot(
        x_indices, [100.0] * len(x_indices),
        color="#d62728", marker="v", linestyle="-",
        label="EPS", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#d62728", markeredgewidth=1.5,
    )
    # Ideal (Baseline) – constant across latencies
    ax.plot(
        x_indices, [baseline_norm] * len(x_indices),
        color="#9467bd", marker="^", linestyle="--",
        label="Ideal", markersize=7, linewidth=1.5,
        markerfacecolor="none", markeredgecolor="#9467bd", markeredgewidth=1.5,
    )

    ax.set_xlabel("Reconfiguration Latency (ms)")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{int(l)}" if l >= 1 else f"{l:.1f}" for l in latencies_ms])
    ax.set_ylabel("Performance (%)")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.legend(loc="lower left", framealpha=0.9)

    # if title:
    #     ax.set_title(title, fontsize=13)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is None:
        output_path = os.path.join(folder_path, "reconfig_latency_normalized.pdf")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot reconfig latency sweep results")
    parser.add_argument("folder_path", help="Path to the results folder")
    parser.add_argument("--title", default=None, help="Plot title")
    args = parser.parse_args()

    folder_path = os.path.abspath(args.folder_path)
    if not os.path.isdir(folder_path):
        print(f"Error: folder not found: {folder_path}")
        sys.exit(1)

    title = args.title or os.path.basename(folder_path)
    plot_reconfig_latency(folder_path, title=title)
    plot_reconfig_latency_raw(folder_path, title=title)
    plot_reconfig_latency_normalized(folder_path, title=title)
