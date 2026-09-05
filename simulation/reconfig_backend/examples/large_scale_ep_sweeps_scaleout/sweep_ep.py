#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples.large_scale_ep_sweeps import sweep_ep as base  # noqa: E402


_base_validate_config = base.validate_config


def validate_config(cfg: dict) -> list[str]:
    errors = _base_validate_config(cfg)

    total = int(cfg["total_gpus"])
    pp = int(cfg["pp"])
    gpn = int(cfg["gpus_per_node"])
    if total % pp != 0 or (total // pp) % gpn != 0:
        return errors

    nps = base.nodes_per_stage(cfg)
    for ep_size in cfg["ep_sizes"]:
        ep = int(ep_size)
        if ep < 1 or ep > nps:
            errors.append(
                f"EP size {ep} is outside the valid scale-out range "
                f"1..{nps} for this PP geometry. This generator maps each EP "
                "group to one TP rail across nodes, so EP size cannot exceed "
                "nodes_per_stage."
            )
        elif nps % ep != 0:
            errors.append(
                f"EP size {ep} does not divide nodes_per_stage={nps}; "
                "this harness builds uniform scale-out EP groups."
            )

    return errors


def ep_topo_enabled(cfg: dict, ep_size: int) -> bool:
    nps = base.nodes_per_stage(cfg)
    return ep_size > 1 and ep_size <= nps and nps % ep_size == 0


def diverged_topo_enabled(cfg: dict, ep_size: int) -> bool:
    return ep_topo_enabled(cfg, ep_size)


def diverged_edge_sets(cfg: dict, ep_size: int) -> list[set[frozenset[int]]]:
    gpn = int(cfg["gpus_per_node"])
    nps = base.nodes_per_stage(cfg)
    groups_per_stage = nps // ep_size
    edge_sets: list[set[frozenset[int]]] = [set() for _ in range(gpn)]

    for group_idx in range(groups_per_stage):
        start = group_idx * ep_size
        nodes = tuple(range(start, start + ep_size))
        for rail, cycle in enumerate(base.diverged_cycles_for_group(nodes, gpn)):
            edge_sets[rail].update(base.cycle_edges(cycle))

    return edge_sets


def generate_comm_groups(cfg: dict, ep_size: int, variant: str) -> tuple[dict[str, list[int]], dict[str, int]]:
    pp = int(cfg["pp"])
    total = int(cfg["total_gpus"])
    gpn = int(cfg["gpus_per_node"])
    rps = base.ranks_per_stage(cfg)
    nps = base.nodes_per_stage(cfg)
    ep_node_groups_per_stage = nps // ep_size
    ep_groups_per_stage = ep_node_groups_per_stage * gpn
    total_nodes = total // gpn

    groups: dict[str, list[int]] = {}

    def ep_members(stage: int, rail: int, group_idx: int) -> list[int]:
        stage_base = stage * rps
        start_node = group_idx * ep_size
        return [
            stage_base + node * gpn + rail
            for node in range(start_node, start_node + ep_size)
        ]

    for stage in range(pp):
        stage_ep_base = stage * ep_groups_per_stage
        for rail in range(gpn):
            for group_idx in range(ep_node_groups_per_stage):
                gid = stage_ep_base + rail * ep_node_groups_per_stage + group_idx
                groups[str(gid)] = ep_members(stage, rail, group_idx)

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


_base_generate_workload = base.generate_workload


def generate_workload(run_dir: Path, cfg: dict, ep_size: int, variant: str) -> None:
    _base_generate_workload(run_dir, cfg, ep_size, variant)

    meta_path = run_dir / "workload_meta.json"
    with meta_path.open() as f:
        meta = json.load(f)
    meta["ep_group_scheme"] = "scaleout_dp_rail"
    meta["ep_crosses_scale_out"] = ep_size > 1
    meta["ep_groups_overlap_tp"] = False
    base.write_json(meta_path, meta)


base.validate_config = validate_config
base.ep_topo_enabled = ep_topo_enabled
base.diverged_topo_enabled = diverged_topo_enabled
base.diverged_edge_sets = diverged_edge_sets
base.generate_comm_groups = generate_comm_groups
base.generate_workload = generate_workload


if __name__ == "__main__":
    base.main()
