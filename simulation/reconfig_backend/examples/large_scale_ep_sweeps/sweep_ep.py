#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import functools
import itertools
import json
import math
import os
import re
import select
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


EPS_VARIANTS = ("eps", "eps_fc")
OPUS_VARIANTS = ("opus_same", "opus_diverge")
VARIANTS = EPS_VARIANTS + OPUS_VARIANTS


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def label_for_ns(value: int) -> str:
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"{value // 1_000_000}ms"
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}us"
    return f"{value}ns"


def all_digit_id(digit: int, pp: int) -> int:
    return int(str(digit) * pp)


def topo_id(digits: tuple[int, ...] | list[int]) -> int:
    return int("".join(str(d) for d in digits))


def digits_for_topo(value: int, pp: int) -> list[int]:
    return [int(ch) for ch in str(value).zfill(pp)]


def load_config(path: Path) -> dict:
    with path.open() as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(path.resolve())
    cfg["_folder"] = str(path.resolve().parent)
    cfg.setdefault("latency_ns", 1000)
    cfg.setdefault("endpoint_delay", 10)
    cfg.setdefault("preferred_dataset_splits", 4)
    cfg.setdefault("layers_per_stage", 2)
    cfg.setdefault("global_batch_size", cfg["microbatches"] * 64)
    cfg.setdefault("seq_len", 4096)
    cfg.setdefault("d_model", 7168)
    cfg.setdefault("kv_lora_rank", 512)
    cfg.setdefault("q_lora_rank", 1536)
    cfg.setdefault("qk_head_dim", 128)
    cfg.setdefault("v_head_dim", 128)
    cfg.setdefault("qk_rope_dim", 64)
    cfg.setdefault("num_heads", 128)
    cfg.setdefault("num_experts", 256)
    cfg.setdefault("top_k", 8)
    cfg.setdefault("expert_d_ff", 2048)
    cfg.setdefault("ep_comm_pattern", "all_to_all")
    cfg.setdefault("moe_routing", "uniform")
    cfg.setdefault("all2allv_payload_scale", "comm_size")
    cfg.setdefault(
        "all2allv_collective_envelope",
        cfg.get("all2allv_collective_traffic", "max_rank_load"),
    )
    cfg.setdefault("dtype_bytes", 2)
    cfg.setdefault("h200_bf16_tflops", 989)
    cfg.setdefault("gemm_efficiency", 0.80)
    cfg.setdefault("ep_sizes", [1, 2, 4, 8, 16, 32, 64])
    cfg.setdefault("reconfig_latencies_ns", [500_000, 1_000_000, 10_000_000])
    cfg.setdefault("scale_out_GBps", cfg.get("scale_out_Gbps", 400.0) / 8.0)
    cfg.setdefault("scale_up_GBps", cfg.get("scale_up_GBps", 450.0))
    cfg.setdefault("plot_latencies_ns", cfg["reconfig_latencies_ns"])
    # "bounded_base": one real run at the smallest latency, others derived as
    # base + switches * delta. "real": simulate every latency end to end.
    cfg.setdefault("opus_run_mode", "bounded_base")
    # Restrict which variants the sweep runs (e.g. ["opus_diverge"]).
    cfg.setdefault("sweep_variants", list(VARIANTS))
    return cfg


def validate_config(cfg: dict) -> list[str]:
    errors: list[str] = []
    total = int(cfg["total_gpus"])
    gpn = int(cfg["gpus_per_node"])
    pp = int(cfg["pp"])
    microbatches = int(cfg["microbatches"])
    global_batch = int(cfg["global_batch_size"])
    num_heads = int(cfg["num_heads"])

    if total % gpn != 0:
        errors.append(
            f"total_gpus={total} is not divisible by gpus_per_node={gpn}; "
            "the analytical topology requires uniform complete nodes."
        )
    if total % pp != 0:
        errors.append(f"total_gpus={total} is not divisible by PP={pp}.")
    if total % pp == 0 and (total // pp) % gpn != 0:
        errors.append(
            f"ranks_per_stage={total // pp} is not divisible by "
            f"gpus_per_node={gpn}; PP stage boundaries must align to nodes."
        )
    if global_batch % microbatches != 0:
        errors.append(
            f"global_batch_size={global_batch} is not divisible by "
            f"microbatches={microbatches}."
        )
    if num_heads % gpn != 0:
        errors.append(
            f"num_heads={num_heads} is not divisible by gpus_per_node={gpn}; "
            "this generator uses one TP group per scale-up domain."
        )

    if total % pp == 0:
        ranks_per_stage = total // pp
        for ep_size in cfg["ep_sizes"]:
            ep = int(ep_size)
            if ep < 1 or ep > ranks_per_stage:
                errors.append(
                    f"EP size {ep} is outside the valid range "
                    f"1..{ranks_per_stage} for this PP geometry."
                )
            elif ranks_per_stage % ep != 0:
                errors.append(
                    f"EP size {ep} does not divide ranks_per_stage={ranks_per_stage}; "
                    "this harness builds uniform EP groups."
                )

    ep_comm_pattern = str(cfg.get("ep_comm_pattern", "all_to_all"))
    if ep_comm_pattern not in ("all_to_all", "all_to_allv"):
        errors.append(
            f"ep_comm_pattern={ep_comm_pattern!r} is invalid; expected "
            "'all_to_all' or 'all_to_allv'."
        )

    moe_routing = str(cfg.get("moe_routing", "uniform"))
    if moe_routing not in ("uniform", "topk_context"):
        errors.append(
            f"moe_routing={moe_routing!r} is invalid; expected "
            "'uniform' or 'topk_context'."
        )

    payload_scale = str(cfg.get("all2allv_payload_scale", "comm_size"))
    if payload_scale not in ("comm_size", "top_k"):
        errors.append(
            f"all2allv_payload_scale={payload_scale!r} is invalid; expected "
            "'comm_size' or 'top_k'."
        )

    collective_envelope = str(cfg.get("all2allv_collective_envelope", "max_rank_load"))
    if collective_envelope not in ("average_remote", "max_rank_load"):
        errors.append(
            f"all2allv_collective_envelope={collective_envelope!r} is invalid; "
            "expected 'average_remote' or 'max_rank_load'."
        )

    if moe_routing == "topk_context" and ep_comm_pattern != "all_to_allv":
        errors.append("moe_routing='topk_context' requires ep_comm_pattern='all_to_allv'.")

    return errors


def checked_config(cfg: dict) -> None:
    errors = validate_config(cfg)
    if errors:
        for error in errors:
            print(f"[validate] ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)


def nodes(cfg: dict) -> int:
    return int(cfg["total_gpus"]) // int(cfg["gpus_per_node"])


def ranks_per_stage(cfg: dict) -> int:
    return int(cfg["total_gpus"]) // int(cfg["pp"])


def nodes_per_stage(cfg: dict) -> int:
    return ranks_per_stage(cfg) // int(cfg["gpus_per_node"])


def system_config(cfg: dict) -> dict:
    return {
        "scheduling-policy": "LIFO",
        "endpoint-delay": int(cfg["endpoint_delay"]),
        "active-chunks-per-dimension": 1,
        "preferred-dataset-splits": int(cfg["preferred_dataset_splits"]),
        "all-reduce-implementation": ["ring"],
        "all-gather-implementation": ["ring"],
        "reduce-scatter-implementation": ["ring"],
        "all-to-all-implementation": ["direct"],
        "collective-optimization": "localBWAware",
        "local-mem-bw": 3350,
        "boost-mode": 0,
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, indent=4)
        f.write("\n")


def write_network(path: Path, cfg: dict, variant: str, reconfig_ns: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gpn = int(cfg["gpus_per_node"])
    total_nodes = nodes(cfg)
    scale_up = float(cfg["scale_up_GBps"])
    scale_out = float(cfg["scale_out_GBps"])
    latency = int(cfg["latency_ns"])

    if variant in EPS_VARIANTS:
        scale_out_topology = "FullyConnected" if variant == "eps_fc" else "Switch"
        scale_out_note = (
            "fully connected scale-out between nodes"
            if variant == "eps_fc"
            else "per-rail switched scale-out between nodes"
        )
        text = "\n".join(
            [
                "# EP baseline topology for the congestion-aware analytical backend.",
                "# Dimension 0: fully connected scale-up inside each node.",
                f"# Dimension 1: {scale_out_note}.",
                f"topology: [ FullyConnected, {scale_out_topology} ]",
                f"npus_count: [ {gpn}, {total_nodes} ]",
                f"bandwidth: [ {scale_up}, {scale_out} ] # GB/s",
                f"latency: [ {latency}, {latency} ] # ns",
                "",
            ]
        )
    else:
        value = int(reconfig_ns if reconfig_ns is not None else cfg["reconfig_latencies_ns"][0])
        text = "\n".join(
            [
                "# Reconfigurable OPUS topology.",
                "# Actual per-link bandwidth comes from schedules.txt.",
                "topology: [ Ring ]",
                f"npus_count: [ {int(cfg['total_gpus'])} ]",
                f"bandwidth: [ {scale_out} ] # GB/s",
                f"latency: [ {latency} ] # ns",
                f"reconfig_time: [ {value} ] # ns",
                "",
            ]
        )
    path.write_text(text)


def ring_edges(count: int) -> set[frozenset[int]]:
    return {frozenset({i, (i + 1) % count}) for i in range(count)}


def cycle_edges(cycle: list[int] | tuple[int, ...]) -> set[frozenset[int]]:
    return {
        frozenset({cycle[i], cycle[(i + 1) % len(cycle)]})
        for i in range(len(cycle))
    }


def cycle_edge_pairs(cycle: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            tuple(sorted((cycle[i], cycle[(i + 1) % len(cycle)])))
            for i in range(len(cycle))
        )
    )


@functools.lru_cache(maxsize=None)
def diverged_cycles_for_group(nodes: tuple[int, ...], rails: int) -> tuple[tuple[int, ...], ...]:
    if len(nodes) <= 2:
        return tuple(nodes for _ in range(rails))

    candidates: list[tuple[tuple[int, ...], tuple[tuple[int, int], ...]]] = []
    if len(nodes) <= 8:
        first = nodes[0]
        for suffix in itertools.permutations(nodes[1:]):
            cycle = (first,) + suffix
            candidates.append((cycle, cycle_edge_pairs(cycle)))
    else:
        for step in range(1, len(nodes)):
            if math.gcd(step, len(nodes)) == 1:
                cycle = tuple(nodes[(i * step) % len(nodes)] for i in range(len(nodes)))
                candidates.append((cycle, cycle_edge_pairs(cycle)))

    edge_counts: dict[tuple[int, int], int] = {}
    cycles: list[tuple[int, ...]] = []
    for _rail in range(rails):
        def score(item: tuple[tuple[int, ...], tuple[tuple[int, int], ...]]) -> tuple[int, int, int, tuple[int, ...]]:
            cycle, edges = item
            new_edges = sum(1 for edge in edges if edge_counts.get(edge, 0) == 0)
            max_after = max(edge_counts.get(edge, 0) + 1 for edge in edges)
            total_after = sum(edge_counts.get(edge, 0) + 1 for edge in edges)
            return (new_edges, -max_after, -total_after, tuple(-node for node in cycle))

        cycle, edges = max(candidates, key=score)
        cycles.append(cycle)
        for edge in edges:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    return tuple(cycles)


def ep_topo_enabled(cfg: dict, ep_size: int) -> bool:
    gpn = int(cfg["gpus_per_node"])
    return ep_size > gpn and ep_size % gpn == 0 and ep_size // gpn > 2


def diverged_topo_enabled(cfg: dict, ep_size: int) -> bool:
    return ep_topo_enabled(cfg, ep_size)


def diverged_edge_sets(cfg: dict, ep_size: int) -> list[set[frozenset[int]]]:
    gpn = int(cfg["gpus_per_node"])
    nps = nodes_per_stage(cfg)
    nodes_per_group = ep_size // gpn
    groups_per_stage = nps // nodes_per_group
    edge_sets: list[set[frozenset[int]]] = [set() for _ in range(gpn)]

    for group_idx in range(groups_per_stage):
        start = group_idx * nodes_per_group
        nodes = tuple(range(start, start + nodes_per_group))
        for rail, cycle in enumerate(diverged_cycles_for_group(nodes, gpn)):
            edge_sets[rail].update(cycle_edges(cycle))

    return edge_sets


def topology_matrix(cfg: dict, digits: list[int], diverged: bool, ep_size: int | None = None) -> list[list[float]]:
    total = int(cfg["total_gpus"])
    gpn = int(cfg["gpus_per_node"])
    rps = ranks_per_stage(cfg)
    nps = nodes_per_stage(cfg)
    scale_up = float(cfg["scale_up_GBps"])
    scale_out = float(cfg["scale_out_GBps"])
    ring = ring_edges(nps)
    diverged_edges = diverged_edge_sets(cfg, ep_size or rps) if diverged else [set() for _ in range(gpn)]

    matrix = [[0.0] * total for _ in range(total)]
    for src in range(total):
        src_stage = src // rps
        src_stage_rank = src % rps
        src_node = src_stage_rank // gpn
        src_rail = src % gpn
        src_physical_node = src // gpn

        for dst in range(total):
            if src == dst:
                continue

            dst_stage = dst // rps
            dst_stage_rank = dst % rps
            dst_node = dst_stage_rank // gpn
            dst_rail = dst % gpn
            dst_physical_node = dst // gpn

            if src_physical_node == dst_physical_node:
                matrix[src][dst] = scale_up
                continue

            if src_stage == dst_stage and src_rail == dst_rail:
                digit = digits[src_stage]
                edge = frozenset({src_node, dst_node})
                if digit == 1 and edge in ring:
                    matrix[src][dst] = scale_out
                    continue
                if digit == 3:
                    if diverged and edge in diverged_edges[src_rail]:
                        matrix[src][dst] = scale_out
                        continue
                    if not diverged and edge in ring:
                        matrix[src][dst] = scale_out
                        continue

            if (
                src_stage != dst_stage
                and src_stage_rank == dst_stage_rank
                and (digits[src_stage] == 0 or digits[dst_stage] == 0)
            ):
                matrix[src][dst] = scale_out

    return matrix


def sentinel_dp_matrix(cfg: dict) -> list[list[float]]:
    total = int(cfg["total_gpus"])
    gpn = int(cfg["gpus_per_node"])
    scale_up = float(cfg["scale_up_GBps"])
    matrix = [[0.0] * total for _ in range(total)]
    for src in range(total):
        src_physical_node = src // gpn
        for dst in range(total):
            if src != dst and src_physical_node == dst // gpn:
                matrix[src][dst] = scale_up
    return matrix


def write_schedules(path: Path, cfg: dict, variant: str, ep_size: int) -> None:
    pp = int(cfg["pp"])
    has_ep_topo = ep_topo_enabled(cfg, ep_size)
    diverged = variant == "opus_diverge" and has_ep_topo
    choices = [0, 1, 3] if has_ep_topo else [0, 1]
    ids = sorted({topo_id(tuple(digits)) for digits in itertools.product(choices, repeat=pp)})

    lines = [
        f"// Generated by {Path(__file__).name} for {cfg['name']} ({variant}).",
        "// Decimal topology digits: 0=PP links, 1=uniform DP ring, 3=EP rail topology.",
        f"// Scale-up: {cfg['scale_up_GBps']} GB/s fully connected within each node.",
        f"// Scale-out: {cfg['scale_out_GBps']} GB/s per rail/link.",
        "",
    ]
    for tid in ids:
        digits = digits_for_topo(tid, pp)
        lines.append(f"// Topo {tid}: digits {digits}")
        lines.append(f"BW {tid}")
        matrix = topology_matrix(cfg, digits, diverged, ep_size)
        lines.extend(" ".join(str(value) for value in row) for row in matrix)
        lines.append("END")
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n")


def import_chakra() -> tuple:
    root = repo_root()
    sys.path.insert(0, str(root / "extern" / "graph_frontend"))
    try:
        from chakra.schema.protobuf.et_def_pb2 import (  # type: ignore
            ALL_GATHER,
            ALL_REDUCE,
            ALL_TO_ALL,
            COMM_COLL_NODE,
            COMM_RECV_NODE,
            COMM_SEND_NODE,
            COMP_NODE,
            REDUCE_SCATTER,
            AttributeProto as ChakraAttr,
            GlobalMetadata,
            Node as ChakraNode,
        )
        from chakra.src.third_party.utils.protolib import encodeMessage as encode_message  # type: ignore
    except ModuleNotFoundError:
        # The artifact vendors the Chakra 0.0.4 schema with its symbolic graph
        # generator, so a separate pip-installed Chakra package is unnecessary.
        sys.path.insert(0, str(root.parent / "symbolic_tensor_graph"))
        from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend.et_def import et_def_pb2 as et  # type: ignore
        from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend.protolib import encodeMessage as encode_message  # type: ignore

        ChakraAttr = et.AttributeProto
        GlobalMetadata = et.GlobalMetadata
        ChakraNode = et.Node
        COMP_NODE = et.NodeType.COMP_NODE
        COMM_COLL_NODE = et.NodeType.COMM_COLL_NODE
        COMM_SEND_NODE = et.NodeType.COMM_SEND_NODE
        COMM_RECV_NODE = et.NodeType.COMM_RECV_NODE
        ALL_REDUCE = et.CollectiveCommType.ALL_REDUCE
        ALL_TO_ALL = et.CollectiveCommType.ALL_TO_ALL
        ALL_GATHER = et.CollectiveCommType.ALL_GATHER
        REDUCE_SCATTER = et.CollectiveCommType.REDUCE_SCATTER

    return (
        encode_message,
        ChakraNode,
        GlobalMetadata,
        ChakraAttr,
        COMP_NODE,
        COMM_COLL_NODE,
        COMM_SEND_NODE,
        COMM_RECV_NODE,
        ALL_REDUCE,
        ALL_TO_ALL,
        ALL_GATHER,
        REDUCE_SCATTER,
    )


def generate_comm_groups(cfg: dict, ep_size: int, variant: str) -> tuple[dict[str, list[int]], dict[str, int]]:
    pp = int(cfg["pp"])
    total = int(cfg["total_gpus"])
    gpn = int(cfg["gpus_per_node"])
    rps = ranks_per_stage(cfg)
    nps = nodes_per_stage(cfg)
    ep_groups_per_stage = rps // ep_size
    total_nodes = total // gpn

    groups: dict[str, list[int]] = {}

    def ep_members(stage: int, group_idx: int) -> list[int]:
        stage_base = stage * rps
        start = stage_base + group_idx * ep_size
        return list(range(start, start + ep_size))

    for stage in range(pp):
        for group_idx in range(ep_groups_per_stage):
            gid = stage * ep_groups_per_stage + group_idx
            groups[str(gid)] = ep_members(stage, group_idx)

    tp_group_base = pp * ep_groups_per_stage
    for node in range(total_nodes):
        start = node * gpn
        groups[str(tp_group_base + node)] = list(range(start, start + gpn))

    dp_group_base = tp_group_base + total_nodes
    for stage in range(pp):
        stage_base = stage * rps
        for rail in range(gpn):
            gid = dp_group_base + stage * gpn + rail
            groups[str(gid)] = [stage_base + node * gpn + rail for node in range(nps)]

    stage_group_base = dp_group_base + pp * gpn
    for stage in range(pp):
        stage_base = stage * rps
        groups[str(stage_group_base + stage)] = list(range(stage_base, stage_base + rps))

    bases = {
        "ep_groups_per_stage": ep_groups_per_stage,
        "tp_group_base": tp_group_base,
        "dp_group_base": dp_group_base,
        "stage_group_base": stage_group_base,
    }
    return groups, bases


class WorkloadShape:
    def __init__(self, cfg: dict, ep_size: int) -> None:
        self.cfg = cfg
        self.ep_size = ep_size
        self.pp = int(cfg["pp"])
        self.microbatches = int(cfg["microbatches"])
        self.total = int(cfg["total_gpus"])
        self.gpn = int(cfg["gpus_per_node"])
        self.rps = ranks_per_stage(cfg)
        self.layers_per_stage = int(cfg["layers_per_stage"])
        self.total_layers = self.pp * self.layers_per_stage
        self.global_batch = int(cfg["global_batch_size"])
        self.microbatch_size = self.global_batch // self.microbatches
        self.seq_len = int(cfg["seq_len"])
        self.global_microbatch_tokens = self.microbatch_size * self.seq_len
        self.local_seq = self.global_microbatch_tokens // self.rps
        self.num_heads = int(cfg["num_heads"])
        self.tp_heads = self.num_heads // self.gpn
        self.num_experts = int(cfg["num_experts"])
        self.ep_experts = max(1, self.num_experts // ep_size)
        self.d_model = int(cfg["d_model"])
        self.kv_lora_rank = int(cfg["kv_lora_rank"])
        self.q_lora_rank = int(cfg["q_lora_rank"])
        self.qk_head_dim = int(cfg["qk_head_dim"])
        self.v_head_dim = int(cfg["v_head_dim"])
        self.qk_rope_dim = int(cfg["qk_rope_dim"])
        self.top_k = int(cfg["top_k"])
        self.expert_d_ff = int(cfg["expert_d_ff"])
        self.dtype_bytes = int(cfg["dtype_bytes"])
        self.effective_tflops = float(cfg["h200_bf16_tflops"]) * float(cfg["gemm_efficiency"])

        self.tokens_post_dispatch = (
            self.global_microbatch_tokens * self.top_k // self.num_experts * self.ep_experts
        )
        self.attn_compute_us = self._flops_to_us(self._attn_flops())
        self.router_compute_us = self._flops_to_us(self._router_flops())
        self.expert_compute_us = self._flops_to_us(self._expert_flops())
        self.attn_bwd_us = 2 * self.attn_compute_us
        self.expert_bwd_us = 2 * self.expert_compute_us
        self.comm_size = self.local_seq * self.d_model * self.dtype_bytes
        self.pp_activation_size = self.comm_size
        self.attn_grad_size = self._mla_params_total() // self.gpn * self.dtype_bytes
        self.expert_grad_size = self.ep_experts * 3 * self.d_model * self.expert_d_ff * self.dtype_bytes

    def _attn_flops(self) -> int:
        f = 2 * self.local_seq * self.d_model * (self.kv_lora_rank + self.qk_rope_dim)
        f += 2 * self.local_seq * self.kv_lora_rank * self.tp_heads * (
            self.v_head_dim + self.qk_head_dim + self.qk_rope_dim
        )
        f += 2 * self.local_seq * self.d_model * (self.q_lora_rank // self.gpn)
        f += 2 * self.local_seq * (self.q_lora_rank // self.gpn) * (self.tp_heads * self.qk_head_dim)
        f += 2 * self.local_seq * self.seq_len * self.tp_heads * self.qk_head_dim
        f += 2 * self.local_seq * self.seq_len * self.tp_heads * self.v_head_dim
        f += 2 * self.local_seq * (self.tp_heads * self.v_head_dim) * (self.d_model // self.gpn)
        return f

    def _router_flops(self) -> int:
        return 2 * self.local_seq * self.d_model * self.ep_experts

    def _expert_flops(self) -> int:
        return 2 * 2 * self.tokens_post_dispatch * self.d_model * self.expert_d_ff

    def _flops_to_us(self, flops: int) -> int:
        return max(1, int(flops / (self.effective_tflops * 1e12) * 1e6))

    def _mla_params_total(self) -> int:
        return (
            self.d_model * (self.kv_lora_rank + self.qk_rope_dim)
            + self.kv_lora_rank * self.num_heads * (self.v_head_dim + self.qk_head_dim + self.qk_rope_dim)
            + self.d_model * self.q_lora_rank
            + self.q_lora_rank * self.num_heads * (self.qk_head_dim + self.qk_rope_dim)
            + self.num_heads * self.v_head_dim * self.d_model
        )


def generate_workload(run_dir: Path, cfg: dict, ep_size: int, variant: str) -> None:
    (
        encode_message,
        ChakraNode,
        GlobalMetadata,
        ChakraAttr,
        COMP_NODE,
        COMM_COLL_NODE,
        COMM_SEND_NODE,
        COMM_RECV_NODE,
        ALL_REDUCE,
        ALL_TO_ALL,
        ALL_GATHER,
        REDUCE_SCATTER,
    ) = import_chakra()

    shape = WorkloadShape(cfg, ep_size)
    workload_dir = run_dir / "workload"
    workload_dir.mkdir(parents=True, exist_ok=True)
    groups, bases = generate_comm_groups(cfg, ep_size, variant)
    write_json(run_dir / "comm_group.json", groups)
    rank_to_ep_group: dict[int, str] = {}
    for gid, members in groups.items():
        if int(gid) >= bases["tp_group_base"]:
            continue
        for member in members:
            rank_to_ep_group[member] = gid

    ep_crosses_scale_out = ep_size > shape.gpn
    uses_ep_topo = variant in OPUS_VARIANTS and ep_topo_enabled(cfg, ep_size)
    uses_diverged_topo = variant == "opus_diverge" and uses_ep_topo
    dp_topo = all_digit_id(1, shape.pp)
    ep_topo = all_digit_id(3, shape.pp) if uses_ep_topo else dp_topo

    def comp(nid: int, name: str, us: int, deps: list[int],
             reconfig_topo_id: int | None = None,
             reconfig_dim: int | None = None):
        node = ChakraNode()
        node.id = nid
        node.name = name
        node.type = COMP_NODE
        node.duration_micros = us
        for dep in deps:
            node.data_deps.append(dep)
        node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        if reconfig_topo_id is not None:
            node.attr.append(ChakraAttr(name="reconfig_topo_id", int64_val=reconfig_topo_id))
        if reconfig_dim is not None:
            node.attr.append(ChakraAttr(name="reconfig_dim", int64_val=reconfig_dim))
        return node

    def coll(nid: int, name: str, ctype: int, pg: str, size: int,
             deps: list[int], reconfig_topo_id: int | None = None,
             reconfig_dim: int | None = None,
             skip_scheduler_reconfig: bool = False,
             all2allv_load_key: str | None = None):
        node = ChakraNode()
        node.id = nid
        node.name = name
        node.type = COMM_COLL_NODE
        for dep in deps:
            node.data_deps.append(dep)
        node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        node.attr.append(ChakraAttr(name="comm_type", int64_val=ctype))
        node.attr.append(ChakraAttr(name="comm_size", int64_val=size))
        node.attr.append(ChakraAttr(name="pg_name", string_val=pg))
        if reconfig_topo_id is not None:
            node.attr.append(ChakraAttr(name="reconfig_topo_id", int64_val=reconfig_topo_id))
        if reconfig_dim is not None:
            node.attr.append(ChakraAttr(name="reconfig_dim", int64_val=reconfig_dim))
        if skip_scheduler_reconfig:
            node.attr.append(ChakraAttr(name="skip_scheduler_reconfig", int64_val=1))
        if all2allv_load_key is not None:
            node.attr.append(ChakraAttr(name="all2allv_load_key", string_val=all2allv_load_key))
        return node

    ep_comm_pattern = str(cfg.get("ep_comm_pattern", "all_to_all"))
    moe_routing = str(cfg.get("moe_routing", "uniform"))
    all2allv_payload_scale = str(cfg.get("all2allv_payload_scale", "comm_size"))
    all2allv_collective_envelope = str(cfg.get("all2allv_collective_envelope", "max_rank_load"))
    all2allv_load_cache: dict[str, dict] = {}

    def send(nid: int, name: str, src: int, dst: int, size: int, tag: int,
             deps: list[int], reconfig_dim: int | None = None,
             skip_scheduler_reconfig: bool = False):
        node = ChakraNode()
        node.id = nid
        node.name = name
        node.type = COMM_SEND_NODE
        for dep in deps:
            node.data_deps.append(dep)
        node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        node.attr.append(ChakraAttr(name="comm_src", int32_val=src))
        node.attr.append(ChakraAttr(name="comm_dst", int32_val=dst))
        node.attr.append(ChakraAttr(name="comm_size", int64_val=size))
        node.attr.append(ChakraAttr(name="comm_tag", int32_val=tag))
        if reconfig_dim is not None:
            node.attr.append(ChakraAttr(name="reconfig_dim", int64_val=reconfig_dim))
        if skip_scheduler_reconfig:
            node.attr.append(ChakraAttr(name="skip_scheduler_reconfig", int64_val=1))
        return node

    def recv(nid: int, name: str, src: int, dst: int, size: int, tag: int,
             deps: list[int], skip_scheduler_reconfig: bool = False):
        node = ChakraNode()
        node.id = nid
        node.name = name
        node.type = COMM_RECV_NODE
        for dep in deps:
            node.data_deps.append(dep)
        node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        node.attr.append(ChakraAttr(name="comm_src", int32_val=src))
        node.attr.append(ChakraAttr(name="comm_dst", int32_val=dst))
        node.attr.append(ChakraAttr(name="comm_size", int64_val=size))
        node.attr.append(ChakraAttr(name="comm_tag", int32_val=tag))
        if skip_scheduler_reconfig:
            node.attr.append(ChakraAttr(name="skip_scheduler_reconfig", int64_val=1))
        return node

    def reconfig_for_dp() -> int | None:
        return dp_topo if variant in OPUS_VARIANTS else None

    def reconfig_for_ep() -> int | None:
        return ep_topo if variant in OPUS_VARIANTS else None

    @functools.lru_cache(maxsize=None)
    def topk_context_owner_counts(src: int, members: tuple[int, ...]) -> tuple[int, ...]:
        counts = [0] * len(members)
        if not members:
            return tuple(counts)
        experts_per_member = max(1, shape.num_experts // len(members))
        stage_rank = src % shape.rps
        token_base = stage_rank * shape.local_seq
        alpha = float(shape.cfg.get("moe_popularity_exponent", 1.5))
        for token in range(shape.local_seq):
            # Each rank holds whole sequences (batch sharding): its tokens
            # span the full context range, so aggregated top-k routing
            # reaches every EP peer, as in real MoE training.
            context_frac = ((token_base + token) % shape.seq_len + 0.5) / shape.seq_len
            for offset in range(shape.top_k):
                # Strided picks scattered over the expert space (real routers
                # score experts independently), warped by a power law so some
                # experts are hotter than others: full peer coverage with
                # heterogeneous per-path loads.
                u = (context_frac + offset / shape.top_k) % 1.0
                expert = min(shape.num_experts - 1, int((u ** alpha) * shape.num_experts))
                owner_idx = min(expert // experts_per_member, len(members) - 1)
                counts[owner_idx] += 1
        return tuple(counts)

    def ceil_div(numerator: int, denominator: int) -> int:
        return (numerator + denominator - 1) // denominator

    def ep_all2allv_size(src: int, dst: int, members: tuple[int, ...]) -> int:
        if src == dst:
            return 0
        if moe_routing == "topk_context":
            # Every rank transfers a fixed total payload to its remote peers
            # regardless of the EP group mapping; only the per-destination
            # split is heterogeneous (proportional to top-k context counts).
            # This keeps total network bytes identical across original and
            # scaleout mappings so the curves compare topology, not volume.
            payload = shape.comm_size
            if all2allv_payload_scale == "top_k":
                payload *= shape.top_k
            counts = topk_context_owner_counts(src, members)
            src_idx = members.index(src)
            dst_idx = members.index(dst)
            remote_total = sum(counts) - counts[src_idx]
            if remote_total == 0:
                # All tokens are owner-local: no heterogeneity information,
                # fall back to a uniform split of the same fixed payload.
                return payload // (len(members) - 1)
            return int(round(counts[dst_idx] * payload / remote_total))
        return shape.comm_size // len(members)

    @functools.lru_cache(maxsize=None)
    def ep_all2allv_collective_size(members: tuple[int, ...]) -> int:
        group_size = len(members)
        if group_size <= 1:
            return 0

        outgoing = [0] * group_size
        incoming = [0] * group_size
        for src_idx, src in enumerate(members):
            for dst_idx, dst in enumerate(members):
                size = ep_all2allv_size(src, dst, members)
                outgoing[src_idx] += size
                incoming[dst_idx] += size

        if all2allv_collective_envelope == "average_remote":
            target_remote_per_rank = ceil_div(sum(outgoing), group_size)
        else:
            target_remote_per_rank = max(max(outgoing), max(incoming))

        # This is only the collective envelope used for chunking. Actual
        # per-pair traffic is loaded by the simulator from all2allv_loads.json.
        per_peer = ceil_div(target_remote_per_rank, group_size - 1)
        return max(group_size, per_peer * group_size)

    def ep_all2allv_load_key(pg: str, members: tuple[int, ...]) -> str:
        key = f"pg_{pg}"
        if key not in all2allv_load_cache:
            loads = []
            for src in members:
                for dst in members:
                    if src == dst:
                        continue
                    loads.append({
                        "src": src,
                        "dst": dst,
                        "bytes": ep_all2allv_size(src, dst, members),
                    })
            all2allv_load_cache[key] = {
                "members": list(members),
                "loads": loads,
            }
        return key

    def ep_all_to_all(et, next_id: int, name: str, rank: int, pg: str,
                      deps: list[int]) -> tuple[int, list[int]]:
        if ep_size == 1:
            encode_message(et, comp(next_id, f"{name}:noop", 0, deps))
            return next_id + 1, [next_id]

        ep_dim = 3 if uses_ep_topo else None
        all2allv_load_key = None
        if ep_comm_pattern == "all_to_all":
            comm_size = shape.comm_size
        else:
            members = tuple(groups[pg])
            comm_size = ep_all2allv_collective_size(members)
            all2allv_load_key = ep_all2allv_load_key(pg, members)
        encode_message(et, coll(next_id, name, ALL_TO_ALL, pg, comm_size, deps,
                                reconfig_dim=ep_dim,
                                all2allv_load_key=all2allv_load_key))
        return next_id + 1, [next_id]

    def write_forward_layer(et, next_id: int, rank: int, mb: int, layer: int,
                            dp_group: str, tp_group: str, ep_group: str,
                            deps: list[int]) -> tuple[int, list[int]]:
        prefix = f"MB{mb}:L{layer}"
        encode_message(et, coll(next_id, f"{prefix}:DP-AG-attn", ALL_GATHER,
                                dp_group, shape.attn_grad_size, deps))
        deps = [next_id]
        next_id += 1
        encode_message(et, comp(next_id, f"{prefix}:attn-fwd", shape.attn_compute_us, deps))
        deps = [next_id]
        next_id += 1
        encode_message(et, coll(next_id, f"{prefix}:TP-AR-fwd", ALL_REDUCE,
                                tp_group, shape.comm_size, deps))
        deps = [next_id]
        next_id += 1
        encode_message(et, comp(next_id, f"{prefix}:router-fwd", shape.router_compute_us, deps))
        deps = [next_id]
        next_id += 1
        next_id, deps = ep_all_to_all(et, next_id, f"{prefix}:EP-dispatch-fwd", rank, ep_group, deps)
        encode_message(et, comp(next_id, f"{prefix}:expert-fwd", shape.expert_compute_us, deps))
        deps = [next_id]
        next_id += 1
        return ep_all_to_all(et, next_id, f"{prefix}:EP-combine-fwd", rank, ep_group, deps)

    def write_backward_layer(et, next_id: int, rank: int, mb: int, layer: int,
                             dp_group: str, tp_group: str, ep_group: str,
                             deps: list[int]) -> tuple[int, list[int]]:
        prefix = f"MB{mb}:L{layer}"
        next_id, deps = ep_all_to_all(et, next_id, f"{prefix}:EP-combine-bwd", rank, ep_group, deps)
        encode_message(et, comp(next_id, f"{prefix}:expert-bwd", shape.expert_bwd_us, deps))
        deps = [next_id]
        next_id += 1
        next_id, deps = ep_all_to_all(et, next_id, f"{prefix}:EP-dispatch-bwd", rank, ep_group, deps)
        ep_done = list(deps)
        encode_message(et, comp(next_id, f"{prefix}:topo-switch-to-DP", 0, ep_done,
                                reconfig_topo_id=reconfig_for_dp()))
        dp_ready = [next_id]
        next_id += 1
        encode_message(et, comp(next_id, f"{prefix}:router-bwd", shape.router_compute_us, ep_done))
        deps = [next_id]
        next_id += 1
        encode_message(et, comp(next_id, f"{prefix}:attn-bwd", shape.attn_bwd_us, deps))
        deps = [next_id]
        next_id += 1
        encode_message(et, coll(next_id, f"{prefix}:TP-AR-bwd", ALL_REDUCE,
                                tp_group, shape.comm_size, deps))
        deps = [next_id]
        next_id += 1
        encode_message(et, coll(next_id, f"{prefix}:DP-RS-attn", REDUCE_SCATTER,
                                dp_group, shape.attn_grad_size, dp_ready + deps))
        deps = [next_id]
        next_id += 1
        encode_message(et, coll(next_id, f"{prefix}:DP-RS-expert", REDUCE_SCATTER,
                                dp_group, shape.expert_grad_size, deps))
        deps = [next_id]
        next_id += 1
        encode_message(et, coll(next_id, f"{prefix}:DP-AG-expert", ALL_GATHER,
                                dp_group, shape.expert_grad_size, deps))
        return next_id + 1, [next_id]

    def write_forward_stack(et, next_id: int, rank: int, mb: int, local_layers: list[int],
                            dp_group: str, tp_group: str, ep_group: str,
                            deps: list[int]) -> tuple[int, list[int]]:
        for layer in local_layers:
            next_id, deps = write_forward_layer(et, next_id, rank, mb, layer,
                                                dp_group, tp_group, ep_group, deps)
        return next_id, deps

    def write_backward_stack(et, next_id: int, rank: int, mb: int, local_layers: list[int],
                             dp_group: str, tp_group: str, ep_group: str,
                             deps: list[int]) -> tuple[int, list[int]]:
        for layer in reversed(local_layers):
            next_id, deps = write_backward_layer(et, next_id, rank, mb, layer,
                                                 dp_group, tp_group, ep_group, deps)
        return next_id, deps

    fwd_tag_base = 1_000_000
    bwd_tag_base = 2_000_000
    for rank in range(shape.total):
        stage = rank // shape.rps
        stage_rank = rank % shape.rps
        physical_node = rank // shape.gpn
        rail = rank % shape.gpn
        ep_group = rank_to_ep_group[rank]
        tp_group = str(bases["tp_group_base"] + physical_node)
        dp_group = str(bases["dp_group_base"] + stage * shape.gpn + rail)
        local_layers = list(range(stage * shape.layers_per_stage, (stage + 1) * shape.layers_per_stage))

        with (workload_dir / f"workload.{rank}.et").open("wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))
            next_id = 1

            activation_recv: dict[int, int] = {}
            if stage > 0:
                for mb in range(shape.microbatches):
                    src = rank - shape.rps
                    tag = fwd_tag_base + mb * shape.total + src
                    activation_recv[mb] = next_id
                    encode_message(et, recv(next_id, f"MB{mb}:PP-recv-activation {src}->{rank}",
                                            src, rank, shape.pp_activation_size, tag, []))
                    next_id += 1

            fwd_done: dict[int, list[int]] = {}
            fwd_chain: list[int] = []
            for mb in range(shape.microbatches):
                deps = list(fwd_chain)
                if stage > 0:
                    deps = [activation_recv[mb]] + deps
                next_id, fwd_chain = write_forward_stack(
                    et, next_id, rank, mb, local_layers, dp_group, tp_group, ep_group, deps
                )
                fwd_done[mb] = fwd_chain
                if stage < shape.pp - 1:
                    dst = rank + shape.rps
                    tag = fwd_tag_base + mb * shape.total + rank
                    encode_message(et, send(next_id, f"MB{mb}:PP-send-activation {rank}->{dst}",
                                            rank, dst, shape.pp_activation_size, tag, fwd_chain))
                    fwd_done[mb] = [next_id]
                    fwd_chain = [next_id]
                    next_id += 1

            grad_recv: dict[int, int] = {}
            if stage < shape.pp - 1:
                for mb in range(shape.microbatches):
                    src = rank + shape.rps
                    tag = bwd_tag_base + mb * shape.total + src
                    grad_recv[mb] = next_id
                    encode_message(et, recv(next_id, f"MB{mb}:PP-recv-grad {src}->{rank}",
                                            src, rank, shape.pp_activation_size, tag, []))
                    next_id += 1

            backward_chain: list[int] = []
            for mb in reversed(range(shape.microbatches)):
                deps = list(fwd_done[mb]) + list(backward_chain)
                if stage < shape.pp - 1:
                    deps = [grad_recv[mb]] + deps
                next_id, backward_chain = write_backward_stack(
                    et, next_id, rank, mb, local_layers, dp_group, tp_group, ep_group, deps
                )
                if stage > 0:
                    dst = rank - shape.rps
                    tag = bwd_tag_base + mb * shape.total + rank
                    encode_message(et, send(next_id, f"MB{mb}:PP-send-grad {rank}->{dst}",
                                            rank, dst, shape.pp_activation_size, tag, backward_chain))
                    backward_chain = [next_id]
                    next_id += 1

    meta = {
        "variant": variant,
        "ep_size": ep_size,
        "pp": shape.pp,
        "microbatches": shape.microbatches,
        "total_gpus": shape.total,
        "gpus_per_node": shape.gpn,
        "ranks_per_stage": shape.rps,
        "layers_per_stage": shape.layers_per_stage,
        "comm_size_bytes": shape.comm_size,
        "ep_comm_pattern": ep_comm_pattern,
        "moe_routing": moe_routing,
        "all2allv_payload_scale": all2allv_payload_scale,
        "all2allv_collective_envelope": all2allv_collective_envelope,
        "all2allv_load_cache": "all2allv_loads.json" if ep_comm_pattern == "all_to_allv" else None,
        "attn_grad_size_bytes": shape.attn_grad_size,
        "expert_grad_size_bytes": shape.expert_grad_size,
        "dp_topo_id": dp_topo if variant in OPUS_VARIANTS else None,
        "ep_topo_id": ep_topo if variant in OPUS_VARIANTS else None,
        "ep_crosses_scale_out": ep_crosses_scale_out,
        "uses_ep_topo": uses_ep_topo,
        "uses_diverged_topo": uses_diverged_topo,
    }
    write_json(run_dir / "workload_meta.json", meta)
    if ep_comm_pattern == "all_to_allv":
        write_json(run_dir / "all2allv_loads.json", all2allv_load_cache)


def run_dir_for(cfg: dict, ep_size: int, variant: str) -> Path:
    return Path(cfg["_folder"]) / "runs" / f"ep_{ep_size:03d}" / variant


def prepare_run_dir(cfg: dict, ep_size: int, variant: str, with_traces: bool) -> Path:
    run_dir = run_dir_for(cfg, ep_size, variant)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "system.json", system_config(cfg))
    write_json(run_dir / "remote_memory.json", {"memory-type": "NO_MEMORY_EXPANSION"})
    write_network(run_dir / "network.yml", cfg, variant)
    if variant in OPUS_VARIANTS:
        write_schedules(run_dir / "schedules.txt", cfg, variant, ep_size)
    if with_traces:
        generate_workload(run_dir, cfg, ep_size, variant)
    return run_dir


def find_binary(name: str) -> Path:
    root = repo_root()
    candidates = []
    if override := os.environ.get("ASTRA_SIM_BIN_DIR"):
        candidates.append(Path(override).expanduser() / name)
    candidates.extend([
        root / "build" / "astra_analytical" / "build" / "bin" / name,
        root / "build" / "astra_analytical" / "build_local" / "bin" / name,
        Path("/app/astra-sim/build/astra_analytical/build/bin") / name,
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{name} not found; build the analytical backend first")


def parse_wall_times(path: Path) -> tuple[str, str, str, str]:
    wall_times: list[int] = []
    wall_pattern = re.compile(r"Wall time:\s*(\d+)")
    finish_pattern = re.compile(r"sys\[\d+\] finished,\s*(\d+)\s+cycles")
    with path.open(errors="replace") as f:
        for line in f:
            match = wall_pattern.search(line)
            if match:
                wall_times.append(int(match.group(1)))
                continue
            match = finish_pattern.search(line)
            if match:
                wall_times.append(int(match.group(1)))
    if not wall_times:
        return "NA", "NA", "NA", "no_wall_time"
    return (
        str(min(wall_times)),
        str(max(wall_times)),
        str(round(statistics.fmean(wall_times))),
        "ok",
    )


def bounded_opus_reconfig_switches(cfg: dict) -> int:
    return int(cfg.get("bounded_opus_reconfig_switches", 30))


def add_ns(value: str, delta: int) -> str:
    if value == "NA":
        return value
    return str(int(value) + int(delta))


def load_summary(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {(row["ep_size"], row["variant"], row["latency_label"]): row for row in rows}


def write_summary(path: Path, rows_by_key: dict[tuple[str, str, str], dict[str, str]]) -> None:
    fieldnames = [
        "scenario",
        "total_gpus",
        "gpus_per_node",
        "pp",
        "microbatches",
        "ep_size",
        "variant",
        "latency_label",
        "reconfig_time_ns",
        "wall_min_ns",
        "wall_max_ns",
        "wall_avg_ns",
        "status",
        "output_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        rows_by_key.values(),
        key=lambda row: (int(row["ep_size"]), row["variant"], int(row["reconfig_time_ns"])),
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def record_result(cfg: dict, ep_size: int, variant: str, latency_label: str,
                  reconfig_ns: int, output_path: Path, summary_rows: dict,
                  extra_wall_ns: int = 0) -> None:
    wall_min, wall_max, wall_avg, status = parse_wall_times(output_path)
    if status == "ok" and extra_wall_ns:
        wall_min = add_ns(wall_min, extra_wall_ns)
        wall_max = add_ns(wall_max, extra_wall_ns)
        wall_avg = add_ns(wall_avg, extra_wall_ns)
    row = {
        "scenario": cfg["name"],
        "total_gpus": str(cfg["total_gpus"]),
        "gpus_per_node": str(cfg["gpus_per_node"]),
        "pp": str(cfg["pp"]),
        "microbatches": str(cfg["microbatches"]),
        "ep_size": str(ep_size),
        "variant": variant,
        "latency_label": latency_label,
        "reconfig_time_ns": str(reconfig_ns),
        "wall_min_ns": wall_min,
        "wall_max_ns": wall_max,
        "wall_avg_ns": wall_avg,
        "status": status,
        "output_path": str(output_path),
    }
    summary_rows[(str(ep_size), variant, latency_label)] = row


def keep_sim_output_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if re.fullmatch(r"(?:\d+(?:\.\d+)?\s*)+", stripped):
        return False
    drop_tokens = (
        "Already in the requested topology",
        "checking votes for comm group",
        "previous comm has not finished yet",
    )
    if any(token in line for token in drop_tokens):
        return False
    keep_tokens = (
        "Wall time:",
        "sys[",
        "critical",
        "warning",
        "ERROR",
        "Error:",
        "Traceback",
        "Assertion",
        "Topology ID",
        "Exiting",
        "finished",
        "Routing algorithm:",
        "Parsed BW matrix",
        "Mapped Comm Group",
        "PP Stages",
        "Num NPU per PP",
        "TM:",
        "Drained Network",
    )
    return any(token in line for token in keep_tokens)


def summary_has_ok(summary_rows: dict, ep_size: int, variant: str, latency_label: str) -> bool:
    row = summary_rows.get((str(ep_size), variant, latency_label))
    return row is not None and row.get("status") == "ok"


def run_command(cmd: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_path.parent.parent if output_path.parent.name == "log" else output_path.parent
    (work_dir / "log").mkdir(parents=True, exist_ok=True)
    keep_pattern = (
        r"Wall time:"
        r"|\[workload\].*sys\[[0-9]+\] finished"
        r"|critical|warning|ERROR|Error:|Traceback|Assertion"
        r"|Topology ID|Exiting|Routing algorithm:"
        r"|Parsed BW matrix|Mapped Comm Group|PP Stages|Num NPU per PP"
        r"|TM:|Drained Network"
    )
    with output_path.open("w") as f:
        sim = subprocess.Popen(cmd, cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert sim.stdout is not None
        filt = subprocess.Popen(
            ["grep", "-aE", keep_pattern],
            stdin=sim.stdout,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        sim.stdout.close()
        rc = sim.wait()
        filter_rc = filt.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    if filter_rc not in (0, 1):
        raise subprocess.CalledProcessError(filter_rc, ["grep", "-aE", keep_pattern])


def run_eps(cfg: dict, ep_size: int, variant: str, summary_rows: dict, skip_existing: bool) -> None:
    run_dir = prepare_run_dir(cfg, ep_size, variant, with_traces=True)
    output_path = run_dir / "output.txt"
    if not (skip_existing and summary_has_ok(summary_rows, ep_size, variant, "none")):
        binary = find_binary("AstraSim_Analytical_Congestion_Aware")
        cmd = [
            str(binary),
            f"--workload-configuration={run_dir / 'workload' / 'workload'}",
            f"--system-configuration={run_dir / 'system.json'}",
            f"--network-configuration={run_dir / 'network.yml'}",
            f"--remote-memory-configuration={run_dir / 'remote_memory.json'}",
            f"--comm-group-configuration={run_dir / 'comm_group.json'}",
        ]
        print(f"[run] {variant} ep={ep_size}")
        run_command(cmd, output_path)
    record_result(cfg, ep_size, variant, "none", 0, output_path, summary_rows)


def run_opus(cfg: dict, ep_size: int, variant: str, reconfig_ns: int,
             summary_rows: dict, skip_existing: bool) -> None:
    run_dir = prepare_run_dir(cfg, ep_size, variant, with_traces=True)
    label = label_for_ns(reconfig_ns)
    if str(cfg.get("opus_run_mode", "bounded_base")) == "real":
        # Simulate this reconfiguration latency end to end (no derivation).
        network_path = run_dir / "log" / f"network_{label}.yml"
        output_path = run_dir / "log" / f"output_{label}.txt"
        write_network(network_path, cfg, variant, reconfig_ns)
        if not (skip_existing and summary_has_ok(summary_rows, ep_size, variant, label)):
            binary = find_binary("AstraSim_Analytical_Reconfigurable")
            cmd = [
                str(binary),
                f"--workload-configuration={run_dir / 'workload' / 'workload'}",
                f"--system-configuration={run_dir / 'system.json'}",
                f"--network-configuration={network_path}",
                f"--remote-memory-configuration={run_dir / 'remote_memory.json'}",
                f"--comm-group-configuration={run_dir / 'comm_group.json'}",
                f"--circuit-schedules={run_dir / 'schedules.txt'}",
                "--routing-algorithm=opus",
            ]
            print(f"[run] {variant} ep={ep_size} reconfig={label} real")
            run_command(cmd, output_path)
        record_result(cfg, ep_size, variant, label, reconfig_ns, output_path, summary_rows)
        return
    base_latency = min(int(v) for v in cfg["reconfig_latencies_ns"])
    base_label = label_for_ns(base_latency)
    network_path = run_dir / "log" / f"network_bounded_base_{base_label}.yml"
    output_path = run_dir / "log" / f"output_bounded_base_{base_label}.txt"
    write_network(network_path, cfg, variant, base_latency)
    should_run = (not output_path.exists()) or (not skip_existing and reconfig_ns == base_latency)
    if should_run:
        binary = find_binary("AstraSim_Analytical_Reconfigurable")
        cmd = [
            str(binary),
            f"--workload-configuration={run_dir / 'workload' / 'workload'}",
            f"--system-configuration={run_dir / 'system.json'}",
            f"--network-configuration={network_path}",
            f"--remote-memory-configuration={run_dir / 'remote_memory.json'}",
            f"--comm-group-configuration={run_dir / 'comm_group.json'}",
            f"--circuit-schedules={run_dir / 'schedules.txt'}",
            "--routing-algorithm=opus",
        ]
        print(f"[run] {variant} ep={ep_size} reconfig={base_label} bounded-base")
        run_command(cmd, output_path)
    penalty_ns = bounded_opus_reconfig_switches(cfg) * max(0, int(reconfig_ns) - base_latency)
    record_result(cfg, ep_size, variant, label, reconfig_ns, output_path, summary_rows, penalty_ns)


def prepare(cfg: dict, with_traces: bool) -> None:
    checked_config(cfg)
    for ep_size in cfg["ep_sizes"]:
        for variant in VARIANTS:
            print(f"[prepare] ep={ep_size} variant={variant}")
            prepare_run_dir(cfg, int(ep_size), variant, with_traces=with_traces)


def run(cfg: dict, skip_existing: bool) -> None:
    checked_config(cfg)
    summary_path = Path(cfg["_folder"]) / "results" / "summary.csv"
    summary_rows = load_summary(summary_path)
    allowed = set(cfg.get("sweep_variants", VARIANTS))
    for ep_size_raw in cfg["ep_sizes"]:
        ep_size = int(ep_size_raw)
        for variant in EPS_VARIANTS:
            if variant not in allowed:
                continue
            run_eps(cfg, ep_size, variant, summary_rows, skip_existing)
            write_summary(summary_path, summary_rows)
        for variant in OPUS_VARIANTS:
            if variant not in allowed:
                continue
            for reconfig_ns in cfg["reconfig_latencies_ns"]:
                run_opus(cfg, ep_size, variant, int(reconfig_ns), summary_rows, skip_existing)
                write_summary(summary_path, summary_rows)
        write_summary(summary_path, summary_rows)
    write_summary(summary_path, summary_rows)
    print(f"[run] summary: {summary_path}")


def value_ns(row: dict[str, str], stat: str) -> float | None:
    raw = row.get(f"wall_{stat}_ns", "NA")
    if raw == "NA":
        return None
    return float(raw)


def plot(cfg: dict, stat: str) -> None:
    checked_config(cfg)
    import matplotlib.pyplot as plt
    from matplotlib import ticker as mticker

    summary_path = Path(cfg["_folder"]) / "results" / "summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"No summary exists yet: {summary_path}")
    with summary_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    rows_by_key = {(int(row["ep_size"]), row["variant"], row["latency_label"]): row for row in rows}
    ep_sizes = [int(ep) for ep in cfg["ep_sizes"]]
    plot_dir = Path(cfg["_folder"]) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    series = [
        ("eps", "EPS (FullyConnected, Switch)", "#2ca02c", "^", "none", "-", False, 2),
        ("eps_fc", "EPS (FullyConnected, FullyConnected)", "#7f7f7f", "D", "none", "-", False, 2),
        ("opus_diverge", "OPUS diverge topo", "#d62728", "s", None, "-", False, 3),
        ("opus_same", "OPUS same topo", "#1f77b4", "o", None, "--", True, 4),
    ]

    for reconfig_ns in cfg["plot_latencies_ns"]:
        label = label_for_ns(int(reconfig_ns))
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        x = list(range(len(ep_sizes)))
        for variant, display, color, marker, fixed_label, linestyle, hollow, zorder in series:
            latency_label = fixed_label or label
            y: list[float | None] = []
            for ep_size in ep_sizes:
                row = rows_by_key.get((ep_size, variant, latency_label))
                ns = value_ns(row, stat) if row else None
                y.append(None if ns is None else ns / 1e6)
            if any(v is not None for v in y):
                ax.plot(
                    x,
                    y,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2.0,
                    markersize=5.5,
                    color=color,
                    markerfacecolor="white" if hollow else color,
                    markeredgecolor=color,
                    markeredgewidth=1.4 if hollow else 1.0,
                    label=display,
                    zorder=zorder,
                )
            else:
                print(f"[plot] missing all points for {display} at {label}")

        ax.set_xticks(x)
        ax.set_xticklabels([str(ep) for ep in ep_sizes])
        ax.set_xlabel("EP size")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _pos: f"{value:,.0f}" if value >= 10 else f"{value:g}")
        )
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(4.0, 6.0, 7.0, 8.0, 9.0)))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_ylabel(f"E2E wall time ({stat}, ms, log scale)")
        ax.set_title(f"{cfg['name']} - EP sweep at {label} switching latency")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.legend(frameon=False)
        fig.tight_layout()
        out_png = plot_dir / f"ep_sweep_{label}.png"
        out_pdf = plot_dir / f"ep_sweep_{label}.pdf"
        fig.savefig(out_png, dpi=200)
        fig.savefig(out_pdf)
        plt.close(fig)
        print(f"[plot] wrote {out_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run large-scale EP sweeps.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment config.json.")
    parser.add_argument(
        "action",
        choices=("validate", "prepare", "run", "plot", "all"),
        nargs="?",
        default="all",
        help="Action to execute. Default: all.",
    )
    parser.add_argument(
        "--with-traces",
        action="store_true",
        help="Generate Chakra traces during prepare. Run/all always generate traces as needed.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Rerun simulations even when output files already exist.",
    )
    parser.add_argument(
        "--stat",
        choices=("min", "max", "avg"),
        default="max",
        help="Wall-time statistic to plot. Default: max.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.action == "validate":
        errors = validate_config(cfg)
        if errors:
            for error in errors:
                print(f"[validate] ERROR: {error}", file=sys.stderr)
            raise SystemExit(2)
        print(f"[validate] ok: {cfg['name']}")
        return
    if args.action == "prepare":
        prepare(cfg, with_traces=args.with_traces)
    elif args.action == "run":
        run(cfg, skip_existing=not args.rerun)
    elif args.action == "plot":
        plot(cfg, stat=args.stat)
    elif args.action == "all":
        prepare(cfg, with_traces=False)
        run(cfg, skip_existing=not args.rerun)
        plot(cfg, stat=args.stat)


if __name__ == "__main__":
    main()
