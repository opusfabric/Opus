#!/usr/bin/env python3
"""Summarize iteration latency from Perlmutter worker logs."""

import argparse
import re
from datetime import datetime
from pathlib import Path


LINE = re.compile(
    r"\[titan\] (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) .*?step:\s*(\d+)"
)


def node_latencies(path, last):
    timestamps = {}
    with path.open(errors="replace") as stream:
        for line in stream:
            match = LINE.search(line)
            if match:
                step = int(match.group(2))
                timestamps.setdefault(step, datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f"))

    if not timestamps:
        return []
    final_step = max(timestamps)
    return [
        (timestamps[step] - timestamps[step - 1]).total_seconds()
        for step in range(final_step - last + 1, final_step + 1)
        if step in timestamps and step - 1 in timestamps
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_ids", nargs="+", help="job IDs under runs/")
    parser.add_argument("--last", type=int, default=5, help="number of final iterations (default: 5)")
    parser.add_argument("--runs-dir", type=Path, default=Path(__file__).resolve().parents[1] / "runs")
    args = parser.parse_args()

    print("job_id\tnodes\tsamples\taverage_s")
    failed = False
    for job_id in args.job_ids:
        logs = sorted((args.runs_dir / job_id).glob("torchrun_*.log"))
        per_node = [node_latencies(log, args.last) for log in logs]
        valid = [values for values in per_node if len(values) == args.last]
        if not valid:
            print(f"{job_id}\t0\t0\tN/A")
            failed = True
            continue
        # Each node log repeats metrics for local ranks. Use the first timestamp
        # per step, then average all node/iteration samples equally.
        samples = [value for values in valid for value in values]
        average = sum(samples) / len(samples)
        print(f"{job_id}\t{len(valid)}\t{len(samples)}\t{average:.3f}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
