#!/usr/bin/env bash
set -euo pipefail

# Reconstruct paper Figure 14 with fixed GPU counts and explicit EP placement.
# Generated ET traces are ignored; per-case metadata and results are retained.

SCRIPT_DIR=$(dirname "$(realpath "$0")")
OPUS_ROOT=$(realpath "$SCRIPT_DIR/../..")
RECONFIG_BACKEND="$OPUS_ROOT/simulation/reconfig_backend"
EXAMPLE_DIR="$RECONFIG_BACKEND/examples"
PYTHON=${PYTHON:-python3}
RECONFIG_EXE="$RECONFIG_BACKEND/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"
OPUS_SKIP_LEGACY_BUILD=${OPUS_SKIP_LEGACY_BUILD:-1}

CLUSTER_SIZES=${CLUSTER_SIZES:-"256 512"}
PLACEMENTS=${PLACEMENTS:-"tpdp epdp"}
LATENCIES_MS=${LATENCIES_MS:-"0 0.5 1 10 50 100"}
EP_LAYERS=${EP_LAYERS:-58}
EP_SIZES_256_TPDP=${EP_SIZES_256_TPDP:-"1 2 4 8 16 32 64"}
EP_SIZES_256_EPDP=${EP_SIZES_256_EPDP:-"1 2 4 8"}
EP_SIZES_512_TPDP=${EP_SIZES_512_TPDP:-"1 2 4 8 16 32 64 128 256"}
EP_SIZES_512_EPDP=${EP_SIZES_512_EPDP:-"1 2 4 8"}
RESULTS_CSV=${RESULTS_CSV:-"$SCRIPT_DIR/fig14_ep_results.csv"}
PLOT_PDF=${PLOT_PDF:-"$SCRIPT_DIR/fig14_ep.pdf"}

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Error: Python interpreter not found: $PYTHON" >&2
    exit 2
fi
if [[ ! -x "$RECONFIG_EXE" ]]; then
    echo "Error: build the reconfigurable simulator first: $RECONFIG_EXE" >&2
    exit 2
fi

case_sizes() {
    case "$1:$2" in
        256:tpdp) echo "$EP_SIZES_256_TPDP" ;;
        256:epdp) echo "$EP_SIZES_256_EPDP" ;;
        512:tpdp) echo "$EP_SIZES_512_TPDP" ;;
        512:epdp) echo "$EP_SIZES_512_EPDP" ;;
        *) echo "Error: unsupported cluster/placement $1/$2" >&2; return 1 ;;
    esac
}

generate_case() {
    local total=$1
    local placement=$2
    local ep=$3
    local out_dir="$EXAMPLE_DIR/fig14_${total}_${placement}_ep${ep}"
    mkdir -p "$out_dir"

    "$PYTHON" - "$OPUS_ROOT" "$out_dir" "$total" "$placement" "$ep" "$EP_LAYERS" <<PY
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
total = int(sys.argv[3])
placement = sys.argv[4]
ep = int(sys.argv[5])
layers = int(sys.argv[6])

if total == 256:
    tp, pp, dp, domain, scaleout, scaleup = 8, 4, 8, 8, 50.0, 450.0
else:
    tp, pp, dp, domain, scaleout, scaleup = 32, 2, 8, 32, 100.0, 900.0
if total != dp * tp * pp or total % domain:
    raise SystemExit("invalid fixed-GPU setup")

sys.path.insert(0, str(root / "simulation" / "symbolic_tensor_graph"))
from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend.et_def.et_def_pb2 import (
    Node, NodeType, CollectiveCommType, GlobalMetadata, AttributeProto,
)
from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend.protolib import encodeMessage

def attr(name, **kwargs):
    return AttributeProto(name=name, **kwargs)

def rank_of(stage, data, tensor):
    return stage * tp * dp + data * tp + tensor

def build_groups():
    groups = {}
    gid = 1
    stage_size = tp * dp
    for stage in range(pp):
        base = stage * stage_size
        if placement == "tpdp":
            for start in range(0, stage_size, ep):
                groups[gid] = list(range(base + start, base + start + ep))
                gid += 1
        elif placement == "epdp":
            for tensor in range(tp):
                for start in range(0, dp, ep):
                    groups[gid] = [
                        rank_of(stage, data, tensor)
                        for data in range(start, start + ep)
                    ]
                    gid += 1
        else:
            raise SystemExit("unknown placement")
    return groups

groups = build_groups()
rank_group = {
    rank: gid for gid, members in groups.items() for rank in members
}

def topology(which):
    matrix = [[0.0 for _ in range(total)] for _ in range(total)]
    if which == "pp":
        stage_size = tp * dp
        for stage in range(0, pp, 2):
            if stage + 1 >= pp:
                continue
            for data in range(dp):
                for tensor in range(tp):
                    left = rank_of(stage, data, tensor)
                    right = rank_of(stage + 1, data, tensor)
                    matrix[left][right] = matrix[right][left] = scaleout
        return matrix

    for base in range(0, total, domain):
        for i in range(base, base + domain):
            for j in range(base, base + domain):
                if i != j:
                    matrix[i][j] = scaleup

    for members in groups.values():
        for i in members:
            for j in members:
                if i != j and matrix[i][j] == 0:
                    matrix[i][j] = scaleout
    return matrix

with (out / "schedules.txt").open("w") as stream:
    for topo_id, matrix in enumerate((topology("ep"), topology("pp"))):
        stream.write(f"BW {topo_id}\n")
        for row in matrix:
            stream.write(" ".join(f"{value:.3f}" for value in row) + "\n")
        stream.write("END\n\n")

payload = (256 // dp // pp) * 4096 * 8192 * 8 * 2
for rank in range(total):
    nodes = []
    next_id = 0

    def make(name, node_type, deps=(), duration=0, attrs=()):
        global next_id
        node = Node(id=next_id, name=name, type=node_type, duration_micros=duration)
        node.data_deps.extend(deps)
        node.attr.extend(attrs)
        nodes.append(node)
        next_id += 1
        return node

    previous = make(
        "attention_compute",
        NodeType.COMP_NODE,
        duration=20000,
        attrs=[
            attr("is_cpu_op", int32_val=0),
            attr("num_ops", int64_val=1),
            attr("tensor_size", uint64_val=1),
            attr("op_type", string_val="compute"),
        ],
    )
    stage = rank // (tp * dp)
    if stage % 2 == 0 and stage + 1 < pp:
        peer = rank + tp * dp
        boundary = make(
            "pp_send",
            NodeType.COMM_SEND_NODE,
            deps=[previous.id],
            attrs=[
                attr("comm_size", int64_val=payload // 32),
                attr("comm_tag", int32_val=17),
                attr("comm_dst", int32_val=peer),
                attr("is_cpu_op", int32_val=0),
            ],
        )
    elif stage % 2 == 1:
        peer = rank - tp * dp
        boundary = make(
            "pp_recv",
            NodeType.COMM_RECV_NODE,
            deps=[previous.id],
            attrs=[
                attr("comm_size", int64_val=payload // 32),
                attr("comm_tag", int32_val=17),
                attr("comm_src", int32_val=peer),
                attr("is_cpu_op", int32_val=0),
            ],
        )
    else:
        boundary = previous

    previous = boundary
    gid = rank_group[rank]
    for layer in range(layers):
        compute = make(
            f"moe_compute_{layer}",
            NodeType.COMP_NODE,
            deps=[previous.id],
            duration=30000,
            attrs=[
                attr("is_cpu_op", int32_val=0),
                attr("num_ops", int64_val=1),
                attr("tensor_size", uint64_val=1),
                attr("op_type", string_val="compute"),
            ],
        )
        if ep > 1:
            previous = make(
                f"ep_all_to_all_{layer}",
                NodeType.COMM_COLL_NODE,
                deps=[compute.id],
                attrs=[
                    attr("comm_size", int64_val=payload),
                    attr("comm_type", int64_val=CollectiveCommType.ALL_TO_ALL),
                    attr("pg_name", string_val=str(gid)),
                    attr("is_cpu_op", int32_val=0),
                ],
            )

    with (out / f"workload.{rank}.et").open("wb") as stream:
        encodeMessage(stream, GlobalMetadata(version="0.0.4"))
        for node in nodes:
            encodeMessage(stream, node)

(out / "workload.json").write_text(
    json.dumps({str(gid): members for gid, members in groups.items()})
)
(out / "network.yml").write_text(
    f"topology: [ FullyConnected ]\nnpus_count: [ {total} ]\n"
    "bandwidth: [ 25 ]\nlatency: [ 936.25 ]\nreconfig_time: [ 0.0 ]\n"
)
(out / "remote_memory.json").write_text(
    "{\"memory-type\": \"NO_MEMORY_EXPANSION\"}\n"
)
(out / "system.json").write_text(
    "{\"scheduling-policy\":\"LIFO\",\"endpoint-delay\":10,"
    "\"active-chunks-per-dimension\":1,\"preferred-dataset-splits\":4,"
    "\"all-reduce-implementation\":[\"ring\"],"
    "\"all-gather-implementation\":[\"ring\"],"
    "\"reduce-scatter-implementation\":[\"ring\"],"
    "\"all-to-all-implementation\":[\"ring\"],"
    "\"collective-optimization\":\"localBWAware\","
    "\"local-mem-bw\":4800,\"boost-mode\":0,\"replay-only\":0,"
    "\"roofline-enabled\":1,\"peak-perf\":989}\n"
)
PY
    cp "$RECONFIG_BACKEND/examples/stg-template/run_network_reconfig.sh" "$out_dir/run_network_reconfig.sh"
    chmod +x "$out_dir/run_network_reconfig.sh"
}

run_case() {
    local total=$1
    local placement=$2
    local ep=$3
    local out_dir="$EXAMPLE_DIR/fig14_${total}_${placement}_ep${ep}"
    generate_case "$total" "$placement" "$ep"

    for latency in $LATENCIES_MS; do
        local label
        label=$(echo "$latency" | tr "." "_")
        sed -E -i "s/^reconfig_time:.*/reconfig_time: [ ${latency}e6 ]/" "$out_dir/network.yml"
        echo "Running Figure 14: $total GPUs $placement EP=$ep OCS=$latency ms"
        (
            cd "$out_dir"
            OPUS_SKIP_PROVISION=1 OPUS_SKIP_LEGACY_BUILD="$OPUS_SKIP_LEGACY_BUILD" \
                bash ./run_network_reconfig.sh >"run_${label}ms.log" 2>&1
        )
        local cycles
        cycles=$(sed -n "s/.*sys\[0\] finished, \([0-9][0-9]*\) cycles.*/\1/p" \
            "$out_dir/debug_no_provision.txt" | tail -1)
        if [[ -z "$cycles" ]]; then
            echo "Error: no sys[0] result in $out_dir/debug_no_provision.txt" >&2
            tail -80 "$out_dir/debug_no_provision.txt" >&2
            exit 1
        fi
        printf "%s,%s,%s,%s,%s\n" "$total" "$placement" "$ep" "$latency" "$cycles" >>"$RESULTS_CSV"
    done
}

mkdir -p "$(dirname "$RESULTS_CSV")" "$(dirname "$PLOT_PDF")"
printf "gpus,placement,ep,latency_ms,cycles\n" >"$RESULTS_CSV"
for total in $CLUSTER_SIZES; do
    for placement in $PLACEMENTS; do
        for ep in $(case_sizes "$total" "$placement"); do
            run_case "$total" "$placement" "$ep"
        done
    done
done

"$PYTHON" - "$RESULTS_CSV" "$PLOT_PDF" <<PY
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(Path(sys.argv[1]).open()))
by_case = defaultdict(dict)
for row in rows:
    key = (int(row["gpus"]), row["placement"])
    by_case[key][(int(row["ep"]), float(row["latency_ms"]))] = int(row["cycles"]) / 1e9

fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
colors = {0.5: "#1f77b4", 1.0: "#ff7f0e", 10.0: "#2ca02c", 50.0: "#d62728", 100.0: "#9467bd"}
labels = {0.5: "Opus 500us", 1.0: "Opus 1ms", 10.0: "Opus 10ms", 50.0: "Opus 50ms", 100.0: "Opus 100ms"}
for row_idx, gpus in enumerate((256, 512)):
    for col_idx, placement in enumerate(("tpdp", "epdp")):
        ax = axes[row_idx][col_idx]
        data = by_case.get((gpus, placement), {})
        sizes = sorted({ep for ep, latency in data if latency == 0.0})
        if sizes:
            ax.plot(sizes, [data[(ep, 0.0)] for ep in sizes], color="#111111", marker="o", label="EPS")
        for latency in (0.5, 1.0, 10.0, 50.0, 100.0):
            points = [(ep, data[(ep, latency)]) for ep in sizes if (ep, latency) in data]
            if points:
                x, y = zip(*points)
                ax.plot(x, y, marker="o", linewidth=1.4, color=colors[latency], label=labels[latency])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("EP size")
        ax.set_ylabel("Step latency (s)")
        ax.set_title(f"{gpus} GPUs, {'EP+TP+DP' if placement == 'tpdp' else 'EP+DP'}")
        ax.grid(True, alpha=0.3)
        if sizes:
            ax.set_xticks(sizes)
            ax.set_xticklabels([str(ep) for ep in sizes])
handles, names = axes[0][0].get_legend_handles_labels()
fig.legend(handles, names, loc="upper center", ncol=3)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(sys.argv[2], dpi=150, bbox_inches="tight")
print(f"Saved {sys.argv[2]}")
PY
