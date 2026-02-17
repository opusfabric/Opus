#!/usr/bin/env python3
"""
Parser to find stuck collective communications in AstraSim debug traces.

Analyzes issuing and finishing of collectives to find which ones never complete.
"""

import re
import argparse
from collections import defaultdict


def parse_debug_file(filepath: str) -> dict:
    """
    Parse a debug file to track collective lifecycle.

    Note: The "finish collective: X" number is a global sequence counter,
    NOT the CG ID. We track issue/finish counts per rank to find stuck collectives.

    Returns dict with:
        - issued_by_rank: {rank: [(time, cg_id, npu_list), ...]}
        - finished_by_rank: {rank: [(time, seq_id), ...]}
        - cg_stats: {cg_id: {'issued': count, 'npus': set}}
    """
    # Pattern for issuing collective
    # RANK: 0 at time 13272621065 Issuing collective CG(id=1, NPUs=[0, 1])
    issue_pattern = re.compile(
        r"RANK:\s*(\d+)\s+at time\s+(\d+)\s+Issuing\s+collective\s+CG\(id=(\d+),\s*NPUs=\[([^\]]+)\]"
    )

    # Pattern for finish collective (seq_id is a global counter, not CG ID)
    # RANK: 0 at time 13327951713 finish collective: 106
    finish_pattern = re.compile(
        r"RANK:\s*(\d+)\s+at time\s+(\d+)\s+finish\s+collective:\s*(\d+)"
    )

    # Track issued collectives per rank: [(issue_time, cg_id, npu_list), ...]
    issued_by_rank = defaultdict(list)
    # Track finished collectives per rank: [(finish_time, seq_id), ...]
    finished_by_rank = defaultdict(list)
    # Track CG stats: {cg_id: {'issued': count, 'npus': set, 'ranks': set}}
    cg_stats = defaultdict(lambda: {'issued': 0, 'npus': set(), 'ranks': set()})

    last_time = 0
    last_issue_time = {}  # {rank: time}
    last_finish_time = {}  # {rank: time}

    with open(filepath, 'r') as f:
        for line in f:
            # Check for issuing collective
            issue_match = issue_pattern.search(line)
            if issue_match:
                rank = int(issue_match.group(1))
                time = int(issue_match.group(2))
                cg_id = int(issue_match.group(3))
                npu_list_str = issue_match.group(4)
                npu_list = [int(x.strip()) for x in npu_list_str.split(',')]

                issued_by_rank[rank].append((time, cg_id, npu_list))
                cg_stats[cg_id]['issued'] += 1
                cg_stats[cg_id]['npus'].update(npu_list)
                cg_stats[cg_id]['ranks'].add(rank)
                last_time = max(last_time, time)
                last_issue_time[rank] = time
                continue

            # Check for finish collective
            finish_match = finish_pattern.search(line)
            if finish_match:
                rank = int(finish_match.group(1))
                time = int(finish_match.group(2))
                seq_id = int(finish_match.group(3))

                finished_by_rank[rank].append((time, seq_id))
                last_time = max(last_time, time)
                last_finish_time[rank] = time
                continue

    return {
        'issued_by_rank': dict(issued_by_rank),
        'finished_by_rank': dict(finished_by_rank),
        'cg_stats': dict(cg_stats),
        'last_time': last_time,
        'last_issue_time': last_issue_time,
        'last_finish_time': last_finish_time
    }


def analyze_rank_status(data: dict) -> dict:
    """
    Analyze the status of each rank: issued vs finished counts.
    """
    status = {}

    for rank in set(data['issued_by_rank'].keys()) | set(data['finished_by_rank'].keys()):
        issued = data['issued_by_rank'].get(rank, [])
        finished = data['finished_by_rank'].get(rank, [])

        num_issued = len(issued)
        num_finished = len(finished)
        pending = num_issued - num_finished

        # Get CG ID breakdown
        cg_breakdown = defaultdict(int)
        for _, cg_id, _ in issued:
            cg_breakdown[cg_id] += 1

        # Last activity
        last_issue = data['last_issue_time'].get(rank, 0)
        last_finish = data['last_finish_time'].get(rank, 0)

        # Get the pending collectives (last N issued)
        pending_collectives = issued[-pending:] if pending > 0 else []

        status[rank] = {
            'issued': num_issued,
            'finished': num_finished,
            'pending': pending,
            'cg_breakdown': dict(cg_breakdown),
            'last_issue_time': last_issue,
            'last_finish_time': last_finish,
            'pending_collectives': pending_collectives
        }

    return status


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


def main():
    parser = argparse.ArgumentParser(
        description='Find stuck collective communications in AstraSim debug traces.'
    )
    parser.add_argument('filepath', help='Path to the debug file (e.g., debug_no_provision.txt)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed information')

    args = parser.parse_args()

    print(f"Parsing {args.filepath}...")
    data = parse_debug_file(args.filepath)

    print(f"\nSummary:")
    print(f"  Last recorded time: {format_time(data['last_time'])}")

    # Show CG stats
    print(f"\nCommunication Groups (CG) found:")
    for cg_id, stats in sorted(data['cg_stats'].items()):
        print(f"  CG {cg_id}: issued {stats['issued']} times, NPUs={sorted(stats['npus'])}, ranks={sorted(stats['ranks'])}")

    # Analyze rank status
    status = analyze_rank_status(data)

    print(f"\n{'='*70}")
    print("RANK STATUS:")
    print('='*70)

    stuck_ranks = []
    active_ranks = []

    for rank in sorted(status.keys()):
        s = status[rank]
        state = "STUCK" if s['pending'] > 0 else "OK"
        if s['pending'] > 0:
            stuck_ranks.append(rank)
        else:
            active_ranks.append(rank)

        print(f"\nRank {rank}: [{state}]")
        print(f"  Issued: {s['issued']}, Finished: {s['finished']}, Pending: {s['pending']}")
        print(f"  CG breakdown: {s['cg_breakdown']}")
        print(f"  Last issue: {format_time(s['last_issue_time'])}")
        print(f"  Last finish: {format_time(s['last_finish_time'])}")

        if s['pending'] > 0 and args.verbose:
            print(f"  First pending collective:")
            if s['pending_collectives']:
                time, cg_id, npus = s['pending_collectives'][0]
                print(f"    CG {cg_id} at {format_time(time)}, NPUs={npus}")

    # Root cause analysis
    print(f"\n{'='*70}")
    print("ROOT CAUSE ANALYSIS:")
    print('='*70)

    if not stuck_ranks:
        print("\nNo stuck ranks found - all collectives completed successfully!")
        return

    print(f"\nStuck ranks: {stuck_ranks}")
    print(f"Active ranks (still making progress): {active_ranks}")

    # Find divergence point - when did ranks start diverging?
    # Compare last finish times
    max_finish = max(s['last_finish_time'] for s in status.values())
    min_stuck_finish = min(status[r]['last_finish_time'] for r in stuck_ranks)

    print(f"\nTiming analysis:")
    print(f"  Latest finish time overall: {format_time(max_finish)}")
    print(f"  Last finish time for stuck ranks: {format_time(min_stuck_finish)}")

    # Find the first stuck collective
    first_stuck_time = float('inf')
    first_stuck_rank = None
    first_stuck_cg = None
    first_stuck_npus = None

    for rank in stuck_ranks:
        pending = status[rank]['pending_collectives']
        if pending:
            time, cg_id, npus = pending[0]
            if time < first_stuck_time:
                first_stuck_time = time
                first_stuck_rank = rank
                first_stuck_cg = cg_id
                first_stuck_npus = npus

    if first_stuck_rank is not None:
        print(f"\nFirst stuck collective:")
        print(f"  CG ID: {first_stuck_cg}")
        print(f"  Rank: {first_stuck_rank}")
        print(f"  Time: {format_time(first_stuck_time)}")
        print(f"  NPUs in collective: {first_stuck_npus}")

        # Check which NPUs are in this CG
        cg_npus = set(first_stuck_npus)
        participating_ranks = [r for r in stuck_ranks if r in cg_npus]
        missing_ranks = [r for r in cg_npus if r not in stuck_ranks or status[r]['pending'] == 0]

        print(f"\n  This is a collective between NPUs {sorted(cg_npus)}")
        print(f"  Ranks that are stuck waiting: {participating_ranks}")

        # Check issue counts for all NPUs in this CG
        print(f"\n  Issue counts for CG {first_stuck_cg}:")
        issue_counts = {}
        for npu in sorted(cg_npus):
            if npu in status:
                cg_count = status[npu]['cg_breakdown'].get(first_stuck_cg, 0)
                issue_counts[npu] = cg_count
                stuck_marker = " <-- STUCK" if npu in stuck_ranks else ""
                print(f"    Rank {npu}: issued {cg_count} times{stuck_marker}")

        # Check for imbalance
        if len(set(issue_counts.values())) > 1:
            max_issues = max(issue_counts.values())
            min_issues = min(issue_counts.values())
            diff = max_issues - min_issues

            print(f"\n  >>> IMBALANCE DETECTED:")
            print(f"      Max issues: {max_issues}, Min issues: {min_issues}, Difference: {diff}")

            # Find who issued more and who issued less
            more_ranks = [r for r, c in issue_counts.items() if c == max_issues]
            less_ranks = [r for r, c in issue_counts.items() if c == min_issues]
            print(f"      Rank(s) with more issues: {more_ranks}")
            print(f"      Rank(s) with fewer issues: {less_ranks}")

            print(f"\n  >>> ROOT CAUSE: Rank {more_ranks[0]} issued {diff} more collective(s) for CG {first_stuck_cg}")
            print(f"      than Rank {less_ranks[0]}. This means Rank {more_ranks[0]} is waiting for")
            print(f"      Rank {less_ranks[0]} to issue, but Rank {less_ranks[0]} has no more work to do.")
            print(f"\n  >>> This indicates a workload scheduling mismatch between ranks.")

        if missing_ranks:
            print(f"\n  >>> DEADLOCK CAUSE: Rank(s) {missing_ranks} are in this CG but are not waiting")
            print(f"  >>> Possible reasons:")
            print(f"      - These ranks finished all their work and exited")
            print(f"      - Workload imbalance caused early termination")
            print(f"      - Bug in scheduling preventing collective from being issued")


if __name__ == '__main__':
    main()
