#!/usr/bin/env python3
"""
CDF plot of timing gaps for each cross-parallelism transition type,
extracted from DeepSeek-V2 236B training traces (EP=32, DP=8, PP=8, TP=4).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict
from pathlib import Path

TRACE_DIR = Path(__file__).parent
US_TO_MS  = 1e-3

# ── Classifier (same as trace_timing_gap.py) ─────────────────────────────────
def classify(event):
    name = event.get("name", "")
    cat  = event.get("cat", "")
    dims = event.get("args", {}).get("Input Dims", None)
    if name in ("c10d::send", "c10d::recv_") and cat == "cpu_op":
        return "PP"
    if cat != "user_annotation":
        return None
    if name == "nccl:all_to_all":
        flat = dims[0] if dims and isinstance(dims[0], list) else dims
        if flat and len(flat) == 3 and flat[0] == 4:
            return "TP"
        if flat and ((len(flat) == 2 and flat[-1] == 5120) or
                     (len(flat) == 1 and flat[0] == 160)):
            return "EP"
        return None
    if name in ("nccl:reduce_scatter_tensor_coalesced",
                "nccl:all_gather_into_tensor_coalesced",
                "nccl:_all_gather_base",
                "nccl:_reduce_scatter_base",
                "nccl:all_reduce"):
        return "DP"
    return None

# ── Load traces ───────────────────────────────────────────────────────────────
gaps_by_pair = defaultdict(list)

for trace_file in sorted(TRACE_DIR.glob("rank*_trace.json")):
    with open(trace_file) as f:
        data = json.load(f)
    rank = data["distributedInfo"]["rank"]

    ops = []
    for e in data["traceEvents"]:
        c = classify(e)
        if c and c != "TP":
            ops.append({"type": c, "ts": e["ts"], "end": e["ts"] + e.get("dur", 0)})
    ops.sort(key=lambda x: x["ts"])

    prev = None
    for curr in ops:
        if prev is not None and prev["type"] != curr["type"]:
            gap_ms = (curr["ts"] - prev["end"]) * US_TO_MS
            gaps_by_pair[(prev["type"], curr["type"])].append(gap_ms)
        prev = curr

# ── Plot ──────────────────────────────────────────────────────────────────────
PAIRS   = [("DP", "EP"), ("EP", "DP"), ("PP", "DP"), ("DP", "PP")]
COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
LABELS  = {
    ("DP", "EP"): "DP→EP  (ZeRO all-gather → EP dispatch)",
    ("EP", "DP"): "EP→DP  (EP gather → ZeRO reduce-scatter)",
    ("PP", "DP"): "PP→DP  (PP send → ZeRO reduce-scatter)",
    ("DP", "PP"): "DP→PP  (ZeRO → PP send/recv)",
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── Left: all pairs on one plot ───────────────────────────────────────────────
ax = axes[0]
for pair, color in zip(PAIRS, COLORS):
    vals = np.array(gaps_by_pair[pair])
    if len(vals) == 0:
        continue
    vals_sorted = np.sort(vals)
    cdf = np.arange(1, len(vals_sorted) + 1) / len(vals_sorted)
    ax.step(vals_sorted, cdf, where="post", color=color,
            label=f"{LABELS[pair]}  (N={len(vals)})", linewidth=1.8)
    # mark median
    med = np.median(vals_sorted)
    ax.axvline(med, color=color, linestyle="--", linewidth=0.8, alpha=0.6)

ax.set_xlabel("Window (ms)", fontsize=11)
ax.set_ylabel("CDF", fontsize=11)
ax.set_title("CDF of Cross-Parallelism Transition Gaps\n(all ranks combined)", fontsize=11)
ax.legend(fontsize=8, loc="lower right")
ax.set_ylim(0, 1.02)
ax.set_xlim(left=0)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

# ── Right: individual subplots per pair ──────────────────────────────────────
ax2 = axes[1]
ax2.set_visible(False)

fig2, sub_axes = plt.subplots(2, 2, figsize=(12, 8))
sub_axes = sub_axes.flatten()

for i, (pair, color) in enumerate(zip(PAIRS, COLORS)):
    ax = sub_axes[i]
    vals = np.array(gaps_by_pair[pair])
    if len(vals) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        continue
    vals_sorted = np.sort(vals)
    cdf = np.arange(1, len(vals_sorted) + 1) / len(vals_sorted)

    ax.step(vals_sorted, cdf, where="post", color=color, linewidth=2)
    ax.fill_between(vals_sorted, cdf, step="post", alpha=0.15, color=color)

    med = np.median(vals_sorted)
    p25 = np.percentile(vals_sorted, 25)
    p75 = np.percentile(vals_sorted, 75)
    p95 = np.percentile(vals_sorted, 95)

    ax.axvline(med, color=color, linestyle="--", linewidth=1.2, label=f"Median={med:.3f} ms")
    ax.axvline(p95, color="gray",  linestyle=":",  linewidth=1.0, label=f"P95={p95:.3f} ms")

    ax.set_xlabel("Window (ms)", fontsize=10)
    ax.set_ylabel("CDF", fontsize=10)
    ax.set_title(f"{pair[0]}→{pair[1]}  (N={len(vals)}, mean={np.mean(vals):.3f} ms)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    # Annotate IQR
    ax.annotate(f"IQR: [{p25:.2f}, {p75:.2f}] ms",
                xy=(0.97, 0.15), xycoords="axes fraction",
                ha="right", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

fig2.suptitle("CDF of Cross-Parallelism Transition Gaps — DeepSeek-V2 236B\n"
              "EP=32  DP=8  PP=8  TP=4  |  all 4 ranks combined",
              fontsize=12, fontweight="bold")
fig2.tight_layout()

out1 = TRACE_DIR / "transition_cdf_combined.png"
out2 = TRACE_DIR / "transition_cdf_per_pair.png"
fig.tight_layout()
fig.savefig(out1, dpi=150, bbox_inches="tight")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")
print(f"Saved: {out2}")

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print(f"{'Pair':<10} {'N':>5} {'Mean':>8} {'Median':>8} {'P25':>8} {'P75':>8} {'P95':>8} {'Max':>8}")
print("-" * 65)
for pair in PAIRS:
    vals = np.array(gaps_by_pair[pair])
    if len(vals) == 0:
        continue
    print(f"{pair[0]+'→'+pair[1]:<10} {len(vals):>5} "
          f"{np.mean(vals):>7.3f}ms {np.median(vals):>7.3f}ms "
          f"{np.percentile(vals,25):>7.3f}ms {np.percentile(vals,75):>7.3f}ms "
          f"{np.percentile(vals,95):>7.3f}ms {np.max(vals):>7.3f}ms")

overall = np.concatenate([gaps_by_pair[p] for p in PAIRS])
print("-" * 65)
print(f"{'Overall':<10} {len(overall):>5} "
      f"{np.mean(overall):>7.3f}ms {np.median(overall):>7.3f}ms")
