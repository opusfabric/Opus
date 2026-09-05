#!/usr/bin/env python3
import csv
import sys
from pathlib import Path


summary = Path(sys.argv[1])
with summary.open(newline="") as stream:
    rows = [
        row for row in csv.DictReader(stream)
        if row["variant"] == "opus_diverge"
        and row["latency_label"] == "500us"
        and row["status"] == "ok"
    ]

points = sorted((int(row["ep_size"]), int(row["wall_max_ns"])) for row in rows)
if len(points) < 2:
    raise SystemExit(f"missing smoke-test points: found {points}")
if any(right >= left for (_, left), (_, right) in zip(points, points[1:])):
    raise SystemExit(f"EP latency did not decrease monotonically: {points}")

print("EP smoke test passed (500 us Opus diverged rails):")
for ep, nanoseconds in points:
    print(f"  EP={ep:2d}: {nanoseconds / 1e6:.3f} ms")
