#!/usr/bin/env python3
"""
Parser for AstraSim trace files to extract and plot topology reconfiguration events
and per-rank activity (SEND/RECV and collective operations).

Parses lines like:
[1;31mTM: !!! TIME: 0 Reconfig to topo_id: 1, Devices count: 4, NPUs count: 4, inflight_coll 0[0m
RANK: 0 at time 0 Issuing SEND 0-2
RANK: 0 at time 0 Issuing RECV 2-0
[0;32mRANK: 0 at time 3517745331 finish SEND/RECV[0m
[0;32mRANK: 2 at time 13272621065 Issuing collective CG(id=1, NPUs=[0, 1])[0m
[0;32mRANK: 0 at time 13327951713 finish collective: 106, inflight collective 1[0m
Workload::issue_comp, sys->id=3, node->id=3962, start_time=3517745331, runtime=80129 ns
[2025-12-08 12:09:23.721] [workload] [info] sys[2] finished, 8295500360 cycles, e

Creates a flame graph style visualization comparing provisioned vs non-provisioned runs.
"""

import re
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from collections import defaultdict
import yaml

TP_COLOR = "#000000"  # TP communications

def is_tp_communication(npu_list: list, tp_size: int) -> bool:
    """
    Check if a communication is a TP (Tensor Parallelism) collective.

    TP communication requires:
    1. NPU list length matches TP size
    2. NPUs are sequential (consecutive IDs)
    3. First NPU is a multiple of TP size

    Examples with tp_size=2:
        [0, 1] -> True (starts at 0, len=2)
        [2, 3] -> True (starts at 2, len=2)
        [1, 2] -> False (1 is not multiple of 2)

    Examples with tp_size=4:
        [0, 1, 2, 3] -> True
        [4, 5, 6, 7] -> True
        [1, 2, 3, 4] -> False
    """
    if tp_size is None or tp_size <= 1:
        return False
    if len(npu_list) != tp_size:
        return False
    # Check if sequential
    expected = list(range(npu_list[0], npu_list[0] + len(npu_list)))
    if npu_list != expected:
        return False
    # Check if starts at multiple of tp_size
    return npu_list[0] % tp_size == 0


def parse_trace_file(filepath: str, tp_size: int = None) -> dict:
    """
    Parse a trace file and extract topology reconfiguration events and rank activities.

    Args:
        filepath: Path to the trace file

    Returns:
        Dictionary with 'topo', 'ranks', and 'rank_finish_times' data
    """
    # Pattern for topology reconfig
    topo_pattern = re.compile(
        r"TM: !!! TIME:\s*(\d+)\s+Reconfig to topo_id:\s*(\d+)"
    )

    # Pattern for rank issuing operations (with embedded time)
    # RANK: 0 at time 13272621065 Issuing collective CG(id=1, NPUs=[0, 1])
    # RANK: 0 at time 0 Issuing SEND 0-2
    # RANK: 0 at time 0 Issuing RECV 2-0
    issue_send_pattern = re.compile(
        r"RANK:\s*(\d+)\s+at time\s+(\d+)\s+Issuing\s+(?:SEND|RECV)\s+(\d+)-(\d+)"
    )
    issue_collective_pattern = re.compile(
        r"RANK:\s*(\d+)\s+at time\s+(\d+)\s+Issuing\s+collective\s+CG\(id=(\d+),\s*NPUs=\[([^\]]+)\]"
    )

    # Pattern for rank finish operations (with embedded time)
    # RANK: 0 at time 3517745331 finish SEND/RECV
    # RANK: 0 at time 13327951713 finish collective: 106, inflight collective 1
    finish_send_pattern = re.compile(
        r"RANK:\s*(\d+)\s+at time\s+(\d+)\s+finish\s+SEND/RECV"
    )
    finish_collective_pattern = re.compile(
        r"RANK:\s*(\d+)\s+at time\s+(\d+)\s+finish\s+collective:\s*(\d+)"
    )

    # Pattern for compute operations
    # Workload::issue_comp, sys->id=3, node->id=3962, start_time=3517745331, runtime=80129 ns
    compute_pattern = re.compile(
        r"Workload::issue_comp,\s*sys->id=(\d+),\s*node->id=(\d+),\s*start_time=(\d+),\s*runtime=(\d+)\s*ns"
    )

    # Pattern for rank final finish time
    # [2025-12-08 12:09:23.721] [workload] [info] sys[2] finished, 8295500360 cycles
    rank_finished_pattern = re.compile(
        r"sys\[(\d+)\]\s+finished,\s*(\d+)\s+cycles"
    )

    topo_times = []
    topo_ids = []

    # Track rank activities: {rank: [(start_time, end_time, op_type, op_info), ...]}
    rank_activities = defaultdict(list)
    # Track compute activities: {rank: [(start_time, end_time, node_id), ...]}
    compute_activities = defaultdict(list)
    # Pending operations per rank: {rank: (start_time, op_type, op_info)}
    pending_ops = {}
    # Final finish times per rank
    rank_finish_times = {}

    current_time = 0

    with open(filepath, 'r') as f:
        for line in f:
            # Check for topology reconfig (updates current time)
            topo_match = topo_pattern.search(line)
            if topo_match:
                current_time = int(topo_match.group(1))
                topo_id = int(topo_match.group(2))
                topo_times.append(current_time)
                topo_ids.append(topo_id)
                continue

            # Check for compute operations
            compute_match = compute_pattern.search(line)
            if compute_match:
                rank = int(compute_match.group(1))
                node_id = compute_match.group(2)
                start_time = int(compute_match.group(3))
                runtime = int(compute_match.group(4))
                end_time = start_time + runtime
                compute_activities[rank].append((start_time, end_time, node_id))
                continue

            # Check for rank finished (final time)
            rank_finished_match = rank_finished_pattern.search(line)
            if rank_finished_match:
                rank = int(rank_finished_match.group(1))
                finish_cycles = int(rank_finished_match.group(2))
                rank_finish_times[rank] = finish_cycles
                continue

            # Check for issuing SEND/RECV (time is in the line)
            send_match = issue_send_pattern.search(line)
            if send_match:
                rank = int(send_match.group(1))
                op_time = int(send_match.group(2))
                src_npu = int(send_match.group(3))
                dst_npu = int(send_match.group(4))
                op_info = f"{src_npu}-{dst_npu}"
                # Check if this is TP communication (point-to-point between adjacent NPUs in same TP group)
                npu_list = sorted([src_npu, dst_npu])
                is_tp = is_tp_communication(npu_list, tp_size) if tp_size and len(npu_list) == tp_size else False
                # For SEND/RECV with tp_size=2, check if both NPUs are in same TP group
                if tp_size and tp_size == 2:
                    is_tp = (npu_list[1] - npu_list[0] == 1) and (npu_list[0] % tp_size == 0)
                op_type = 'send_tp' if is_tp else 'send'
                pending_ops[rank] = (op_time, op_type, op_info)
                continue

            # Check for issuing collective (time is in the line)
            coll_match = issue_collective_pattern.search(line)
            if coll_match:
                rank = int(coll_match.group(1))
                op_time = int(coll_match.group(2))
                cg_id = coll_match.group(3)
                npu_list_str = coll_match.group(4)
                # Parse NPU list: "0, 1" or "0, 1, 2, 3"
                npu_list = [int(x.strip()) for x in npu_list_str.split(',')]
                is_tp = is_tp_communication(npu_list, tp_size)
                op_type = 'collective_tp' if is_tp else 'collective'
                pending_ops[rank] = (op_time, op_type, f'CG{cg_id}')
                continue

            # Check for finish SEND/RECV (time is in the line)
            finish_send_match = finish_send_pattern.search(line)
            if finish_send_match:
                rank = int(finish_send_match.group(1))
                finish_time = int(finish_send_match.group(2))
                if rank in pending_ops:
                    start_time, op_type, op_info = pending_ops.pop(rank)
                    rank_activities[rank].append((start_time, finish_time, op_type, op_info))
                continue

            # Check for finish collective (time is in the line)
            finish_coll_match = finish_collective_pattern.search(line)
            if finish_coll_match:
                rank = int(finish_coll_match.group(1))
                finish_time = int(finish_coll_match.group(2))
                if rank in pending_ops:
                    start_time, op_type, op_info = pending_ops.pop(rank)
                    rank_activities[rank].append((start_time, finish_time, op_type, op_info))
                continue

    # Determine end time from rank finish times or last topology time
    end_time = current_time
    if rank_finish_times:
        end_time = max(end_time, max(rank_finish_times.values()))

    return {
        'topo': (topo_times, topo_ids),
        'ranks': dict(rank_activities),
        'compute': dict(compute_activities),
        'rank_finish_times': rank_finish_times,
        'end_time': end_time
    }


def format_time(ns: int) -> str:
    """Format nanoseconds into human-readable time notation."""
    if ns >= 1e9:
        return f"{ns / 1e9:.2f}s"
    elif ns >= 1e6:
        return f"{ns / 1e6:.2f}ms"
    elif ns >= 1e3:
        return f"{ns / 1e3:.2f}µs"
    else:
        return f"{ns}ns"


def get_color_for_topo(topo_id: int, cmap_name: str = 'tab10') -> tuple:
    """Get a distinct color for each topology ID."""
    cmap = plt.get_cmap(cmap_name)
    return cmap(topo_id % 10)


def get_color_for_op(op_type: str) -> tuple:
    """Get color for operation type."""
    colors = {
        'send': '#3498db',          # Blue for SEND/RECV
        'send_tp': TP_COLOR,       # Orange for TP SEND/RECV
        'collective': '#e74c3c',    # Red for collective
        'collective_tp': TP_COLOR, # Orange for TP collective
        'compute': '#9b59b6'        # Purple for compute
    }
    return colors.get(op_type, '#95a5a6')


def create_flame_bars(times: list[int], topo_ids: list[int], end_time: int = None) -> list[tuple]:
    """
    Convert time/topo_id pairs into bar segments for flame graph.
    Always starts with topo_id 0 at time 0.

    Returns:
        List of (start_time, duration, topo_id) tuples
    """
    if not times:
        return []

    # Ensure we start with topo_id 0 at time 0
    working_times = list(times)
    working_ids = list(topo_ids)
    if working_times[0] != 0:
        working_times.insert(0, 0)
        working_ids.insert(0, 0)

    bars = []
    for i in range(len(working_times)):
        start = working_times[i]
        if i + 1 < len(working_times):
            end = working_times[i + 1]
        else:
            end = end_time if end_time else start + (working_times[-1] - working_times[0]) * 0.1
        duration = end - start
        bars.append((start, duration, working_ids[i]))

    return bars


def parse_reconfig_time(directory: str) -> int:
    """
    Parse the reconfig_time from network.yml in the given directory.

    Args:
        directory: Directory containing network.yml

    Returns:
        Reconfig time in nanoseconds (default 0 if not found)
    """
    network_yml = os.path.join(directory, 'network.yml')
    if not os.path.exists(network_yml):
        return 0

    try:
        with open(network_yml, 'r') as f:
            config = yaml.safe_load(f)
            reconfig_time = config.get('reconfig_time', [0])
            if isinstance(reconfig_time, list) and len(reconfig_time) > 0:
                return int(reconfig_time[0])
            return int(reconfig_time) if reconfig_time else 0
    except Exception as e:
        print(f"Warning: Could not parse network.yml: {e}")
        return 0


def plot_flame_graph(directory: str, output_path: str = None, num_ranks: int = None, tp_size: int = None):
    """
    Create a flame graph style plot comparing provisioned, non-provisioned, and analytical runs,
    including per-rank activity bars.

    Args:
        directory: Directory containing debug_provision.txt, debug_no_provision.txt, and/or debug_analytical.txt
        output_path: Optional path to save the plot
        num_ranks: Number of ranks to display (auto-detected if None)
        tp_size: Tensor Parallelism size for identifying TP communications (optional)
    """
    provision_file = os.path.join(directory, 'debug_provision.txt')
    no_provision_file = os.path.join(directory, 'debug_no_provision.txt')
    analytical_file = os.path.join(directory, 'debug_analytical.txt')

    # Parse reconfig time from network.yml
    reconfig_time_ns = parse_reconfig_time(directory)
    if reconfig_time_ns > 0:
        print(f"Reconfig downtime: {format_time(reconfig_time_ns)}")

    if tp_size:
        print(f"TP size: {tp_size}")

    # Parse all files
    data = {}
    # Track which traces have topology (for conditional rendering)
    has_topology = {}

    if os.path.exists(provision_file):
        parsed = parse_trace_file(provision_file, tp_size)
        if parsed['topo'][0]:
            data['provision'] = parsed
            has_topology['provision'] = True
            print(f"Provisioned: {len(parsed['topo'][0])} reconfig events, "
                  f"{len(parsed['ranks'])} ranks with activity")
            if parsed['rank_finish_times']:
                print(f"  Rank finish times: {parsed['rank_finish_times']}")

    if os.path.exists(no_provision_file):
        parsed = parse_trace_file(no_provision_file, tp_size)
        if parsed['topo'][0]:
            data['no_provision'] = parsed
            has_topology['no_provision'] = True
            print(f"No Provision: {len(parsed['topo'][0])} reconfig events, "
                  f"{len(parsed['ranks'])} ranks with activity")
            if parsed['rank_finish_times']:
                print(f"  Rank finish times: {parsed['rank_finish_times']}")

    if os.path.exists(analytical_file):
        parsed = parse_trace_file(analytical_file, tp_size)
        # Analytical trace may not have topology reconfig events - that's okay
        if parsed['ranks'] or parsed['compute'] or parsed['rank_finish_times']:
            data['analytical'] = parsed
            has_topology['analytical'] = False  # No topology bar for analytical
            print(f"Analytical: {len(parsed['ranks'])} ranks with activity, "
                  f"{len(parsed['compute'])} ranks with compute")
            if parsed['rank_finish_times']:
                print(f"  Rank finish times: {parsed['rank_finish_times']}")

    if not data:
        print(f"No debug files found in {directory}")
        print(f"  Looking for: debug_provision.txt, debug_no_provision.txt, debug_analytical.txt")
        return

    # Auto-detect number of ranks from the data if not explicitly limited
    detected_ranks = set()
    for parsed in data.values():
        detected_ranks.update(parsed['ranks'].keys())
        detected_ranks.update(parsed['compute'].keys())
        detected_ranks.update(parsed['rank_finish_times'].keys())

    if detected_ranks:
        max_detected = max(detected_ranks) + 1  # ranks are 0-indexed
        if num_ranks is None or num_ranks < max_detected:
            num_ranks = max_detected
            print(f"Auto-detected {num_ranks} ranks from trace data")
    elif num_ranks is None:
        num_ranks = 4  # fallback default

    # Find global time range
    all_times = []
    global_end_time = 0
    for parsed in data.values():
        times, _ = parsed['topo']
        all_times.extend(times)
        global_end_time = max(global_end_time, parsed['end_time'])

    max_time = max(all_times) if all_times else 100
    min_time = min(all_times) if all_times else 0
    end_time = max(global_end_time, max_time * 1.05)

    # Find all unique topo_ids for legend (always include 0)
    all_topo_ids = {0}
    for parsed in data.values():
        _, topo_ids = parsed['topo']
        all_topo_ids.update(topo_ids)

    # Calculate figure height based on content
    num_traces = len(data)
    # Count rows: traces with topology have 1 extra row for topo bar
    total_rows = 0
    for trace_key in data:
        total_rows += num_ranks * 2  # comm + compute for each rank
        if has_topology.get(trace_key, False):
            total_rows += 1  # topology row
    fig_height = max(8, total_rows * 0.5 + 2)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, fig_height))

    bar_height = 0.7
    row_spacing = 1.0
    trace_spacing = 1.5  # Extra space between provision/no_provision sections

    # Calculate y positions
    current_y = 0.5
    y_positions = {}

    for trace_key in ['analytical', 'no_provision', 'provision']:
        if trace_key not in data:
            continue

        y_positions[trace_key] = {'topo': None, 'ranks': {}, 'compute': {}}

        # Rank rows (bottom to top within each trace section)
        # Each rank has two rows: compute (bottom) and comm (top)
        for rank in range(num_ranks - 1, -1, -1):
            y_positions[trace_key]['compute'][rank] = current_y
            current_y += row_spacing
            y_positions[trace_key]['ranks'][rank] = current_y
            current_y += row_spacing

        # Topology row (only for traces with topology)
        if has_topology.get(trace_key, False):
            y_positions[trace_key]['topo'] = current_y
            current_y += trace_spacing
        else:
            # Still add some spacing between trace sections
            current_y += trace_spacing * 0.5

    # Draw bars for each trace
    for trace_key, parsed in data.items():
        times, topo_ids = parsed['topo']
        rank_activities = parsed['ranks']
        compute_activities = parsed['compute']
        rank_finish_times = parsed['rank_finish_times']
        trace_end = parsed['end_time']

        # Draw topology bars (only for traces with topology)
        if has_topology.get(trace_key, False) and y_positions[trace_key]['topo'] is not None:
            topo_y = y_positions[trace_key]['topo']
            bars = create_flame_bars(times, topo_ids, end_time)

            for i, (start, duration, topo_id) in enumerate(bars):
                # Draw gray reconfig downtime at the start of each topology change (except first)
                if i > 0 and reconfig_time_ns > 0:
                    reconfig_rect = mpatches.FancyBboxPatch(
                        (start, topo_y - bar_height / 2),
                        reconfig_time_ns,
                        bar_height,
                        boxstyle="round,pad=0,rounding_size=0",
                        facecolor='#7f8c8d',  # Gray
                        edgecolor='none',
                        hatch='///',
                        alpha=0.8
                    )
                    ax.add_patch(reconfig_rect)

                # Draw the topology bar (offset by reconfig time if not first)
                bar_start = start
                bar_duration = duration
                if i > 0 and reconfig_time_ns > 0:
                    bar_start = start + reconfig_time_ns
                    bar_duration = duration - reconfig_time_ns
                    if bar_duration <= 0:
                        continue  # Skip if reconfig time is longer than the bar

                color = get_color_for_topo(topo_id)
                rect = mpatches.FancyBboxPatch(
                    (bar_start, topo_y - bar_height / 2),
                    bar_duration,
                    bar_height,
                    boxstyle="round,pad=0,rounding_size=0",
                    facecolor=color,
                    edgecolor='none'
                )
                ax.add_patch(rect)

                # Add topo_id label if bar is wide enough
                if bar_duration > (end_time - min_time) * 0.03:
                    ax.text(
                        bar_start + bar_duration / 2,
                        topo_y,
                        str(topo_id),
                        ha='center',
                        va='center',
                        fontsize=9,
                        fontweight='bold',
                        color='white'
                    )

        # Draw rank activity bars
        for rank in range(num_ranks):
            rank_y = y_positions[trace_key]['ranks'][rank]
            activities = rank_activities.get(rank, [])

            for start_t, end_t, op_type, op_info in activities:
                duration = end_t - start_t
                if duration <= 0:
                    duration = (end_time - min_time) * 0.005  # Minimum visible width

                color = get_color_for_op(op_type)
                rect = mpatches.FancyBboxPatch(
                    (start_t, rank_y - bar_height / 2),
                    duration,
                    bar_height,
                    boxstyle="round,pad=0,rounding_size=0",
                    facecolor=color,
                    edgecolor='none',
                    alpha=0.9
                )
                ax.add_patch(rect)

            # Draw rank finish marker if available
            if rank in rank_finish_times:
                finish_time = rank_finish_times[rank]
                # Draw a vertical line marker at finish time
                ax.plot(
                    [finish_time, finish_time],
                    [rank_y - bar_height / 2, rank_y + bar_height / 2],
                    color='#2ecc71',  # Green
                    linewidth=3,
                    solid_capstyle='round'
                )
                # Add a small triangle marker
                ax.scatter(
                    [finish_time],
                    [rank_y],
                    marker='|',
                    s=200,
                    color='#2ecc71',
                    zorder=5
                )

            # Draw compute activity bars
            compute_y = y_positions[trace_key]['compute'][rank]
            computes = compute_activities.get(rank, [])

            for start_t, end_t, node_id in computes:
                duration = end_t - start_t
                if duration <= 0:
                    duration = (end_time - min_time) * 0.002  # Minimum visible width

                color = get_color_for_op('compute')
                rect = mpatches.FancyBboxPatch(
                    (start_t, compute_y - bar_height / 2),
                    duration,
                    bar_height,
                    boxstyle="round,pad=0,rounding_size=0",
                    facecolor=color,
                    edgecolor='none',
                    alpha=0.9
                )
                ax.add_patch(rect)

    # Configure axes
    ax.set_xlim(min_time - (end_time - min_time) * 0.02, end_time * 1.02)
    ax.set_ylim(0, current_y)

    # Y-axis labels
    y_ticks = []
    y_labels = []

    for trace_key in ['analytical', 'no_provision', 'provision']:
        if trace_key not in y_positions:
            continue

        if trace_key == 'provision':
            trace_label = 'Provisioned'
        elif trace_key == 'no_provision':
            trace_label = 'No Provision'
        else:
            trace_label = 'Analytical'

        # Topology label (only for traces with topology)
        if has_topology.get(trace_key, False) and y_positions[trace_key]['topo'] is not None:
            y_ticks.append(y_positions[trace_key]['topo'])
            y_labels.append(f'{trace_label}\nTopology')

        # Rank labels - single label centered between comm and compute rows
        for rank in range(num_ranks):
            # Position label between compute and comm rows
            compute_y = y_positions[trace_key]['compute'][rank]
            comm_y = y_positions[trace_key]['ranks'][rank]
            center_y = (compute_y + comm_y) / 2

            y_ticks.append(center_y)
            finish_info = ""
            if trace_key in data and rank in data[trace_key]['rank_finish_times']:
                finish_ns = data[trace_key]['rank_finish_times'][rank]
                finish_info = f"\n({format_time(finish_ns)})"
            # Include trace label for analytical since it has no topology row
            rank_label = f'Rank {rank}{finish_info}'
            if not has_topology.get(trace_key, False) and rank == num_ranks - 1:
                # Add trace label to the top rank for traces without topology
                rank_label = f'{trace_label}\n{rank_label}'
            y_labels.append(rank_label)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=9)

    # Format x-axis to display seconds
    def ns_to_seconds_formatter(x, pos):
        return f'{x / 1e9:.1f}'

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(ns_to_seconds_formatter))
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title('Topology Reconfiguration & Rank Activity Timeline', fontsize=14)

    # Create legend
    legend_patches = []

    # Topology colors
    for tid in sorted(all_topo_ids):
        legend_patches.append(
            mpatches.Patch(color=get_color_for_topo(tid), label=f'Topo {tid}')
        )

    # Operation type colors
    legend_patches.append(mpatches.Patch(color='#3498db', label='SEND/RECV'))
    legend_patches.append(mpatches.Patch(color='#e74c3c', label='Collective'))
    if tp_size:
        legend_patches.append(mpatches.Patch(color=TP_COLOR, label='TP Comm'))
    legend_patches.append(mpatches.Patch(color='#9b59b6', label='Compute'))
    legend_patches.append(mpatches.Patch(color='#2ecc71', label='Rank Finished'))
    if reconfig_time_ns > 0:
        legend_patches.append(mpatches.Patch(facecolor='#7f8c8d', hatch='///', label='Reconfig Downtime'))

    ax.legend(
        handles=legend_patches,
        loc='upper right',
        ncol=min(len(legend_patches), 6),
        fontsize=8
    )

    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=450, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Parse AstraSim trace files and create flame graph of topology reconfigurations and rank activity.'
    )
    parser.add_argument('directory', help='Directory containing debug_provision.txt, debug_no_provision.txt, and/or debug_analytical.txt')
    parser.add_argument('-o', '--output', help='Output path for the plot (PNG/PDF). If not specified, displays interactively.')
    parser.add_argument('-n', '--num-ranks', type=int, default=None, help='Number of ranks to display (auto-detected if not specified)')
    parser.add_argument('--tp', type=int, default=None, help='Tensor Parallelism size for identifying TP communications')
    parser.add_argument('--print-events', action='store_true', help='Print all reconfiguration events')

    args = parser.parse_args()

    if args.print_events:
        for filename in ['debug_provision.txt', 'debug_no_provision.txt', 'debug_analytical.txt']:
            filepath = os.path.join(args.directory, filename)
            if os.path.exists(filepath):
                parsed = parse_trace_file(filepath, args.tp)
                times, topo_ids = parsed['topo']
                print(f"\n{filename}:")
                print("=" * 60)
                if times:
                    print("Topology Reconfigurations:")
                    print("-" * 40)
                    for t, tid in zip(times, topo_ids):
                        print(f"  Time: {t:>12}  ->  Topo ID: {tid}")
                else:
                    print("No topology reconfigurations (analytical mode)")

                print("\nRank Comm Activities:")
                print("-" * 40)
                for rank, activities in sorted(parsed['ranks'].items()):
                    print(f"  Rank {rank}:")
                    for start, end, op_type, op_info in activities:
                        print(f"    [{start:>10} - {end:>10}] {op_type}: {op_info}")

                print("\nRank Compute Activities:")
                print("-" * 40)
                for rank, computes in sorted(parsed['compute'].items()):
                    print(f"  Rank {rank}: {len(computes)} compute operations")

                print("\nRank Finish Times:")
                print("-" * 40)
                for rank, finish_time in sorted(parsed['rank_finish_times'].items()):
                    print(f"  Rank {rank}: {format_time(finish_time)} ({finish_time:,} ns)")

    plot_flame_graph(args.directory, args.output, args.num_ranks, args.tp)


if __name__ == '__main__':
    main()
