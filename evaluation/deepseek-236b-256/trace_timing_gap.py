#!/usr/bin/env python3
"""
Extract actual timing gaps between adjacent collectives of different parallelism
types from Chrome trace JSON files produced by DeepSeek-V2 236B training.

Collective classification (rank 0 perspective, EP=32, DP=8, PP=8, TP=4):
  EP:  nccl:all_to_all  with Input Dims [N, 5120]  (dispatch/gather, N varies)
       nccl:all_to_all  with Input Dims [160]       (routing index exchange)
  DP:  nccl:reduce_scatter_tensor_coalesced          (ZeRO reduce-scatter)
       nccl:all_gather_into_tensor_coalesced          (ZeRO all-gather for attn/dense)
       nccl:_all_gather_base                          (FSDP expert param prefetch)
       nccl:_reduce_scatter_base                      (FSDP expert grad reduce)
       nccl:all_reduce  with dims [1024,5120] etc.   (TP attention all-reduce → DP)
  PP:  c10d::send                                    (fwd activation / bwd grad send)
       c10d::recv_                                   (fwd activation / bwd grad recv)
  TP:  nccl:all_to_all  with Input Dims [4,256,768] (excluded per request)
       nccl:coalesced                                 (excluded)

A "gap" = ts_B_start - (ts_A_start + dur_A)  [positive = idle compute between A and B]

NOTE on time units:
  Despite "displayTimeUnit: ms" (Chrome viewer setting), all ts/dur are in MICROSECONDS.
  Verified: PP send of 2.62 MB takes ~50 μs raw → 52 GB/s ≈ IB bandwidth ✓
            Trace span = 21,028,077 μs = 21.0 seconds ≈ one training step.
  NCCL "dur" values (~50-120 μs) are CPU-side submission times; GPU transfer is async.
  All reported gaps are CPU inter-submission intervals, converted to milliseconds.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

US_TO_MS = 1e-3   # raw μs → ms

TRACE_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Collective classifier
# ─────────────────────────────────────────────────────────────────────────────

def classify(event):
    """Return 'EP', 'DP', 'PP', 'TP', or None."""
    name = event.get("name", "")
    cat  = event.get("cat", "")
    dims = event.get("args", {}).get("Input Dims", None)

    # PP: point-to-point activation / gradient transfers
    if name in ("c10d::send", "c10d::recv_") and cat == "cpu_op":
        return "PP"

    if cat != "user_annotation":
        return None

    if name == "nccl:all_to_all":
        if dims is None:
            return None
        flat = dims[0] if isinstance(dims[0], list) else dims
        # TP all-to-all: shape [4, 256, 768] — 4 = TP size
        if len(flat) == 3 and flat[0] == 4:
            return "TP"
        # EP: dispatch [N, 5120] or gather [N, 5120] or routing [160]
        if len(flat) == 2 and flat[-1] == 5120:
            return "EP"
        if len(flat) == 1 and flat[0] == 160:
            return "EP"
        return None

    if name in (
        "nccl:reduce_scatter_tensor_coalesced",
        "nccl:all_gather_into_tensor_coalesced",
        "nccl:_all_gather_base",
        "nccl:_reduce_scatter_base",
    ):
        return "DP"

    if name == "nccl:all_reduce":
        return "DP"

    return None


def load_collectives(path):
    """Load and classify all collective events from a trace file."""
    with open(path) as f:
        data = json.load(f)

    rank = data["distributedInfo"]["rank"]
    events = data["traceEvents"]

    ops = []
    for e in events:
        ctype = classify(e)
        if ctype is None:
            continue
        ts  = e["ts"]          # milliseconds (displayTimeUnit: ms)
        dur = e.get("dur", 0)  # milliseconds
        ops.append({
            "rank": rank,
            "type": ctype,
            "name": e["name"],
            "ts":   ts,
            "dur":  dur,
            "end":  ts + dur,
            "dims": e.get("args", {}).get("Input Dims", None),
        })

    ops.sort(key=lambda x: x["ts"])
    return rank, ops


def compute_gaps(ops, exclude_tp=True):
    """
    Walk sorted collective list, find adjacent pairs of different types,
    return list of {"pair": (A,B), "gap_ms": float}.
    Gaps are in ms (raw μs values × US_TO_MS).
    """
    if exclude_tp:
        ops = [o for o in ops if o["type"] != "TP"]

    gaps = []
    prev = None
    for curr in ops:
        if prev is not None and prev["type"] != curr["type"]:
            gap_us = curr["ts"] - prev["end"]   # raw μs
            gaps.append({
                "pair":      (prev["type"], curr["type"]),
                "gap_ms":    gap_us * US_TO_MS,
                "prev_name": prev["name"],
                "curr_name": curr["name"],
            })
        prev = curr
    return gaps


# ─────────────────────────────────────────────────────────────────────────────
# Load all 4 rank traces
# ─────────────────────────────────────────────────────────────────────────────

all_rank_gaps   = {}   # rank -> list of gap dicts
all_rank_ops    = {}   # rank -> list of classified ops

for trace_file in sorted(TRACE_DIR.glob("rank*_trace.json")):
    rank, ops = load_collectives(trace_file)
    gaps = compute_gaps(ops)
    all_rank_ops[rank]  = ops
    all_rank_gaps[rank] = gaps
    print(f"Rank {rank}: {len(ops)} collectives classified  →  {len(gaps)} cross-type windows")

# ─────────────────────────────────────────────────────────────────────────────
# Aggregate statistics
# ─────────────────────────────────────────────────────────────────────────────

def stats(vals):
    if not vals:
        return {}
    return {
        "n":      len(vals),
        "mean":   statistics.mean(vals),
        "median": statistics.median(vals),
        "min":    min(vals),
        "max":    max(vals),
        "stdev":  statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }

def fmt(v): return f"{v:.3f} ms"   # v already in ms

SEP = "=" * 80
sep = "-" * 80

print()
print(SEP)
print("  Cross-Type Timing Gap Analysis — from trace files")
print(SEP)

# ── Per-rank breakdown ───────────────────────────────────────────────────────
for rank in sorted(all_rank_ops.keys()):
    ops  = all_rank_ops[rank]
    gaps = all_rank_gaps[rank]

    # Count collectives by type
    from collections import Counter
    type_counts = Counter(o["type"] for o in ops)

    print(f"\n{'─'*40}")
    print(f"  RANK {rank}")
    print(f"{'─'*40}")
    print(f"  Collectives: " + "  ".join(f"{t}={c}" for t,c in sorted(type_counts.items())))
    print()

    # Group gaps by pair
    by_pair = defaultdict(list)
    for g in gaps:
        by_pair[g["pair"]].append(g["gap_ms"])

    pairs_present = sorted(by_pair.keys())
    print(f"  {'Pair (A→B)':<14} {'N':>5} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'StdDev':>10}")
    print(f"  {'':─<14} {'':─>5} {'':─>10} {'':─>10} {'':─>10} {'':─>10} {'':─>10}")
    for pair in pairs_present:
        s = stats(by_pair[pair])
        print(f"  {pair[0]+'→'+pair[1]:<14} {s['n']:>5} "
              f"{fmt(s['mean']):>10} {fmt(s['median']):>10} "
              f"{fmt(s['min']):>10} {fmt(s['max']):>10} {fmt(s['stdev']):>10}")

# ── Aggregated across all ranks ──────────────────────────────────────────────
print()
print(SEP)
print("  AGGREGATE (all ranks combined)")
print(SEP)

combined = defaultdict(list)
for rank, gaps in all_rank_gaps.items():
    for g in gaps:
        combined[g["pair"]].append(g["gap_ms"])

all_vals = []
for vals in combined.values():
    all_vals.extend(vals)

print(f"\n  {'Pair (A→B)':<14} {'N':>5} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'StdDev':>10}")
print(f"  {'':─<14} {'':─>5} {'':─>10} {'':─>10} {'':─>10} {'':─>10} {'':─>10}")
for pair in sorted(combined.keys()):
    s = stats(combined[pair])
    print(f"  {pair[0]+'→'+pair[1]:<14} {s['n']:>5} "
          f"{fmt(s['mean']):>10} {fmt(s['median']):>10} "
          f"{fmt(s['min']):>10} {fmt(s['max']):>10} {fmt(s['stdev']):>10}")

print()
overall = stats(all_vals)
print(f"  Overall median across ALL cross-type windows: {fmt(overall['median'])}")
print(f"  Overall mean:                                 {fmt(overall['mean'])}")
print(f"  Total windows:                                {overall['n']}")

# ── Collective operation durations (for reference) ──────────────────────────
print()
print(SEP)
print("  Measured Collective Durations (rank 0)")
print(SEP)

ops0 = all_rank_ops[0]
by_name = defaultdict(list)
for o in ops0:
    by_name[(o["type"], o["name"])].append(o["dur"] * US_TO_MS)   # μs → ms

print(f"\n  {'Type':<6} {'Name':<50} {'N':>5} {'Mean(ms)':>10} {'Med(ms)':>10}")
print(f"  {'':─<6} {'':─<50} {'':─>5} {'':─>10} {'':─>10}")
print(f"  (dur = CPU-side NCCL submission time; actual GPU transfer is async)")
print()
for (t, n), durs in sorted(by_name.items()):
    s = stats(durs)
    print(f"  {t:<6} {n:<50} {s['n']:>5} {fmt(s['mean']):>10} {fmt(s['median']):>10}")

# ── Negative gaps (overlapping collectives) ──────────────────────────────────
print()
print(SEP)
print("  Overlapping collectives (negative gaps) — rank 0")
print(SEP)
negs = [g for g in all_rank_gaps[0] if g["gap_ms"] < 0]
print(f"\n  {len(negs)} overlapping windows out of {len(all_rank_gaps[0])} total")
if negs:
    by_pair_neg = defaultdict(list)
    for g in negs:
        by_pair_neg[g["pair"]].append(g["gap_ms"])
    for pair, vals in sorted(by_pair_neg.items()):
        s = stats(vals)
        print(f"  {pair[0]+'→'+pair[1]:<14} {s['n']:>4} windows  "
              f"mean overlap={fmt(abs(s['mean']))}  max overlap={fmt(abs(s['min']))}")
