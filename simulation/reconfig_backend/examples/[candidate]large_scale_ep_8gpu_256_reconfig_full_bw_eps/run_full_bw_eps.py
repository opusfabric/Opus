#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
ROOT = EXAMPLES.parent
EPS_TRACE_VARIANT = "opus_same"

BASELINES = {
    "full": {
        "variant": "eps_baseline",
        "label": "EPS full baseline",
        "run_dir": "eps_baseline",
        "summary_prefix": "eps_baseline",
        "plot_prefix": "eps_baseline",
        "description": "full-bandwidth matrices",
        "routing": "bfs",
    },
    "ring": {
        "variant": "eps_ring_baseline",
        "label": "EPS ring baseline",
        "run_dir": "eps_ring_baseline",
        "summary_prefix": "eps_ring_baseline",
        "plot_prefix": "eps_ring_baseline",
        "description": "same-rail scale-out ring matrices",
        "routing": "bfs",
    },
    "constrained": {
        "variant": "eps_constrained_baseline",
        "label": "EPS",
        "run_dir": "eps_constrained_baseline",
        "summary_prefix": "eps",
        "plot_prefix": "eps",
        "description": "OPUS DP/PP/EP constrained matrices",
        "routing": "opus",
    },
}

SOURCES = [
    {
        "key": "original",
        "label": "EP + TP + DP",
        "folder": EXAMPLES / "large_scale_ep_8gpu_256",
        "color": "#1f77b4",
    },
    {
        "key": "scaleout",
        "label": "EP + DP",
        "folder": EXAMPLES / "large_scale_ep_scaleout_8gpu_256",
        "color": "#ff7f0e",
    },
]

OPUS_VARIANT = "opus_diverge"
OPUS_LABEL = "OPUS diverged topology"
LATENCY_STYLES = {
    "500us": {"linestyle": "--", "marker": "o"},
    "1ms": {"linestyle": "-.", "marker": "s"},
    "10ms": {"linestyle": ":", "marker": "^"},
}

SUMMARY_FIELDS = [
    "source",
    "source_label",
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

COMPARISON_FIELDS = [
    "source",
    "source_label",
    "ep_size",
    "latency_label",
    "stat",
    "eps_ms",
    "opus_diverge_ms",
    "eps_speedup_vs_opus_diverge",
    "eps_slowdown_vs_opus_diverge",
]


def label_for_ns(value: int) -> str:
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"{value // 1_000_000}ms"
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}us"
    return f"{value}ns"


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def source_config(source: dict) -> dict:
    return load_json(source["folder"] / "config.json")


def source_summary(source: dict) -> dict[tuple[int, str, str], dict[str, str]]:
    rows = load_rows(source["folder"] / "results" / "summary.csv")
    return {
        (int(row["ep_size"]), row["variant"], row["latency_label"]): row
        for row in rows
        if row.get("status") == "ok"
    }


def eps_summary(baseline: dict) -> dict[tuple[str, int, str], dict[str, str]]:
    path = HERE / "results" / "summary.csv"
    if not path.exists():
        return {}
    rows = load_rows(path)
    return {
        (row["source"], int(row["ep_size"]), row["latency_label"]): row
        for row in rows
        if row.get("status") == "ok" and row.get("variant") == baseline["variant"]
    }


def ep_sizes(source: dict) -> list[int]:
    return [int(ep) for ep in source_config(source)["ep_sizes"]]


def latency_labels() -> list[str]:
    values: set[int] = set()
    for source in SOURCES:
        values.update(int(v) for v in source_config(source)["reconfig_latencies_ns"])
    return [label_for_ns(v) for v in sorted(values)]


def source_run_dir(source: dict, ep_size: int, variant: str) -> Path:
    return source["folder"] / "runs" / f"ep_{ep_size:03d}" / variant


def target_run_dir(source: dict, ep_size: int, baseline: dict) -> Path:
    return HERE / "runs" / source["key"] / f"ep_{ep_size:03d}" / baseline["run_dir"]


def find_binary() -> Path:
    candidates = []
    if override := os.environ.get("ASTRA_SIM_BIN_DIR"):
        candidates.append(Path(override).expanduser() / "AstraSim_Analytical_Reconfigurable")
    candidates.extend([
        ROOT / "build" / "astra_analytical" / "build" / "bin" / "AstraSim_Analytical_Reconfigurable",
        ROOT / "build" / "astra_analytical" / "build_local" / "bin" / "AstraSim_Analytical_Reconfigurable",
        Path("/app/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"),
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("AstraSim_Analytical_Reconfigurable not found")


def parse_schedule_ids(path: Path) -> list[int]:
    ids: list[int] = []
    pattern = re.compile(r"^BW\s+(\d+)\s*$")
    with path.open() as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                ids.append(int(match.group(1)))
    return sorted(set(ids)) or [1]


def full_bw_matrix(total: int, gpus_per_node: int, scale_up: float, scale_out: float) -> list[str]:
    rows: list[str] = []
    for src in range(total):
        src_node = src // gpus_per_node
        values: list[str] = []
        for dst in range(total):
            if src == dst:
                values.append("0.0")
            elif dst // gpus_per_node == src_node:
                values.append(f"{scale_up:.1f}")
            else:
                values.append(f"{scale_out:.1f}")
        rows.append(" ".join(values))
    return rows


def scaleout_ring_matrix(total: int, gpus_per_node: int, scale_up: float, scale_out: float) -> list[str]:
    rows: list[str] = []
    total_nodes = total // gpus_per_node
    ring = {frozenset({node, (node + 1) % total_nodes}) for node in range(total_nodes)}
    for src in range(total):
        src_node = src // gpus_per_node
        src_rail = src % gpus_per_node
        values: list[str] = []
        for dst in range(total):
            dst_node = dst // gpus_per_node
            dst_rail = dst % gpus_per_node
            if src == dst:
                values.append("0.0")
            elif dst_node == src_node:
                values.append(f"{scale_up:.1f}")
            elif src_rail == dst_rail and frozenset({src_node, dst_node}) in ring:
                values.append(f"{scale_out:.1f}")
            else:
                values.append("0.0")
        rows.append(" ".join(values))
    return rows


def ring_edges(count: int) -> set[frozenset[int]]:
    return {frozenset({node, (node + 1) % count}) for node in range(count)}


def digits_for_topo(value: int, pp: int) -> list[int]:
    return [int(ch) for ch in str(value).zfill(pp)]


def constrained_matrix(cfg: dict, digits: list[int]) -> list[str]:
    total = int(cfg["total_gpus"])
    gpn = int(cfg["gpus_per_node"])
    pp = int(cfg["pp"])
    rps = total // pp
    nps = rps // gpn
    scale_up = float(cfg["scale_up_GBps"])
    scale_out = float(cfg["scale_out_GBps"])
    ring = ring_edges(nps)

    matrix = [["0.0"] * total for _ in range(total)]
    for src in range(total):
        src_stage = src // rps
        src_stage_rank = src % rps
        src_node = src_stage_rank // gpn
        src_rail = src_stage_rank % gpn
        src_physical_node = src // gpn

        for dst in range(total):
            if src == dst:
                continue

            dst_stage = dst // rps
            dst_stage_rank = dst % rps
            dst_node = dst_stage_rank // gpn
            dst_rail = dst_stage_rank % gpn
            dst_physical_node = dst // gpn

            if src_physical_node == dst_physical_node:
                matrix[src][dst] = f"{scale_up:.1f}"
                continue

            if src_stage == dst_stage and src_rail == dst_rail:
                digit = digits[src_stage]
                edge = frozenset({src_node, dst_node})
                if digit == 1 and edge in ring:
                    matrix[src][dst] = f"{scale_out:.1f}"
                    continue
                if digit == 3 and edge in ring:
                    matrix[src][dst] = f"{scale_out:.1f}"
                    continue

            if (
                src_stage != dst_stage
                and src_stage_rank == dst_stage_rank
                and (digits[src_stage] == 0 or digits[dst_stage] == 0)
            ):
                matrix[src][dst] = f"{scale_out:.1f}"

    return [" ".join(row) for row in matrix]


def baseline_matrix(cfg: dict, baseline: dict, topo_id: int | None = None) -> list[str]:
    total = int(cfg["total_gpus"])
    gpn = int(cfg["gpus_per_node"])
    scale_up = float(cfg["scale_up_GBps"])
    scale_out = float(cfg["scale_out_GBps"])
    if baseline["variant"] == "eps_ring_baseline":
        return scaleout_ring_matrix(total, gpn, scale_up, scale_out)
    if baseline["variant"] == "eps_constrained_baseline":
        if topo_id is None:
            raise ValueError("constrained EPS schedules require a topology ID")
        return constrained_matrix(cfg, digits_for_topo(topo_id, int(cfg["pp"])))
    return full_bw_matrix(total, gpn, scale_up, scale_out)


def matrix_note(baseline: dict, scale_out: float) -> str:
    if baseline["variant"] == "eps_ring_baseline":
        return f"// inter-node: {scale_out:.1f} GB/s only on same-rail physical-node ring edges"
    if baseline["variant"] == "eps_constrained_baseline":
        return (
            f"// DP and EP use same-stage same-rail node ring edges at {scale_out:.1f} GB/s; "
            "PP uses same-rank inter-stage links"
        )
    return f"// inter-node: {scale_out:.1f} GB/s for every cross-node GPU pair"


def write_schedules(path: Path, cfg: dict, schedule_ids: list[int], baseline: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scale_up = float(cfg["scale_up_GBps"])
    scale_out = float(cfg["scale_out_GBps"])
    shared_rows = None
    if baseline["variant"] != "eps_constrained_baseline":
        shared_rows = baseline_matrix(cfg, baseline)
    lines = [
        f"// {baseline['label']}: {baseline['description']} on the reconfiguration backend.",
        "// No reconfiguration delay is modeled; network reconfig_time is 0 ns.",
        f"// intra-node: {scale_up:.1f} GB/s for every same-node GPU pair",
        matrix_note(baseline, scale_out),
        "",
    ]
    for tid in schedule_ids:
        rows = shared_rows if shared_rows is not None else baseline_matrix(cfg, baseline, tid)
        lines.append(f"// Topo {tid}: {baseline['description']}")
        lines.append(f"BW {tid}")
        lines.extend(rows)
        lines.append("END")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n")


def write_network(path: Path, cfg: dict, baseline: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scale_out = float(cfg["scale_out_GBps"])
    latency = int(cfg["latency_ns"])
    text = "\n".join(
        [
            f"# {baseline['label']} on the reconfiguration backend.",
            "# Actual per-link bandwidth comes from schedules.txt.",
            "topology: [ Ring ]",
            f"npus_count: [ {int(cfg['total_gpus'])} ]",
            f"bandwidth: [ {scale_out:.1f} ] # GB/s",
            f"latency: [ {latency} ] # ns",
            "reconfig_time: [ 0 ] # ns",
            "",
        ]
    )
    path.write_text(text)


def prepare_run(source: dict, ep_size: int, baseline: dict) -> Path:
    cfg = source_config(source)
    src = source_run_dir(source, ep_size, EPS_TRACE_VARIANT)
    required = [
        src / "workload" / "workload.0.et",
        src / "system.json",
        src / "remote_memory.json",
        src / "comm_group.json",
        src / "schedules.txt",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source artifacts for {src}: {missing[:3]}")

    run_dir = target_run_dir(source, ep_size, baseline)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_schedules(run_dir / "schedules.txt", cfg, parse_schedule_ids(src / "schedules.txt"), baseline)
    write_network(run_dir / "log" / "network_zero.yml", cfg, baseline)
    return run_dir


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
        r"|TM:|Drained Network|reconfig"
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


def run_one(
    source: dict,
    ep_size: int,
    baseline: dict,
    rows: dict[tuple[str, int, str], dict[str, str]],
    rerun: bool,
) -> None:
    key = (source["key"], ep_size, baseline["variant"])
    if not rerun and rows.get(key, {}).get("status") == "ok":
        return

    cfg = source_config(source)
    src = source_run_dir(source, ep_size, EPS_TRACE_VARIANT)
    run_dir = prepare_run(source, ep_size, baseline)
    output_path = run_dir / "log" / "output_zero.txt"
    cmd = [
        str(find_binary()),
        f"--workload-configuration={src / 'workload' / 'workload'}",
        f"--system-configuration={src / 'system.json'}",
        f"--network-configuration={run_dir / 'log' / 'network_zero.yml'}",
        f"--remote-memory-configuration={src / 'remote_memory.json'}",
        f"--comm-group-configuration={src / 'comm_group.json'}",
        f"--circuit-schedules={run_dir / 'schedules.txt'}",
        f"--routing-algorithm={baseline.get('routing', 'bfs')}",
    ]
    print(f"[run] {source['key']} ep={ep_size} {baseline['variant']} reconfig=0ns")
    run_command(cmd, output_path)
    wall_min, wall_max, wall_avg, status = parse_wall_times(output_path)
    rows[key] = {
        "source": source["key"],
        "source_label": source["label"],
        "scenario": cfg["name"],
        "total_gpus": str(cfg["total_gpus"]),
        "gpus_per_node": str(cfg["gpus_per_node"]),
        "pp": str(cfg["pp"]),
        "microbatches": str(cfg["microbatches"]),
        "ep_size": str(ep_size),
        "variant": baseline["variant"],
        "latency_label": "none",
        "reconfig_time_ns": "0",
        "wall_min_ns": wall_min,
        "wall_max_ns": wall_max,
        "wall_avg_ns": wall_avg,
        "status": status,
        "output_path": str(output_path),
    }


def read_existing_eps_rows() -> dict[tuple[str, int, str], dict[str, str]]:
    path = HERE / "results" / "summary.csv"
    if not path.exists():
        return {}
    rows = {}
    for row in load_rows(path):
        rows[(row["source"], int(row["ep_size"]), row["variant"])] = row
    return rows


def write_summary(rows_by_key: dict[tuple[str, int, str], dict[str, str]]) -> None:
    out = HERE / "results" / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows_by_key.values(), key=lambda r: (r["source"], r["variant"], int(r["ep_size"])))
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] wrote {out}")


def run_all(baseline: dict, rerun: bool) -> None:
    rows = read_existing_eps_rows()
    for source in SOURCES:
        for ep in ep_sizes(source):
            run_one(source, ep, baseline, rows, rerun)
            write_summary(rows)
    write_summary(rows)


def value_ns(row: dict[str, str] | None, stat: str) -> float | None:
    if row is None:
        return None
    raw = row.get(f"wall_{stat}_ns", "NA")
    if raw == "NA":
        return None
    return float(raw)


def value_ms(row: dict[str, str] | None, stat: str) -> float | None:
    ns = value_ns(row, stat)
    if ns is None:
        return None
    return ns / 1e6


def fmt_ms(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def fmt_ratio(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def write_comparison(baseline: dict, stat: str) -> None:
    eps_rows = eps_summary(baseline)
    rows: list[dict[str, str]] = []
    for source in SOURCES:
        opus_rows = source_summary(source)
        for label in latency_labels():
            for ep in ep_sizes(source):
                eps_ms = value_ms(eps_rows.get((source["key"], ep, "none")), stat)
                opus_diverge_ms = value_ms(opus_rows.get((ep, OPUS_VARIANT, label)), stat)
                rows.append(
                    {
                        "source": source["key"],
                        "source_label": source["label"],
                        "ep_size": str(ep),
                        "latency_label": label,
                        "stat": stat,
                        "eps_ms": fmt_ms(eps_ms),
                        "opus_diverge_ms": fmt_ms(opus_diverge_ms),
                        "eps_speedup_vs_opus_diverge": fmt_ratio(ratio(opus_diverge_ms, eps_ms)),
                        "eps_slowdown_vs_opus_diverge": fmt_ratio(ratio(eps_ms, opus_diverge_ms)),
                    }
                )

    out = HERE / "results" / f"{baseline['summary_prefix']}_vs_opus_diverged_{stat}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[compare] wrote {out}")


def plot(baseline: dict, stat: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import ticker as mticker

    eps_rows = eps_summary(baseline)
    if not eps_rows:
        raise SystemExit(
            f"No {baseline['variant']} summary exists yet. "
            f"Run `python3 run_full_bw_eps.py --baseline {baseline['name']} run` first."
        )

    plot_dir = HERE / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    labels = latency_labels()
    latency_colors = {
        "500us": "#1f77b4",
        "1ms": "#ff7f0e",
        "10ms": "#d62728",
    }
    source_styles = {
        "original": {"linestyle": "-", "suffix": "EP + TP + DP", "hollow": False},
        "scaleout": {"linestyle": "--", "suffix": "EP + DP", "hollow": True},
    }

    all_eps = sorted({ep for source in SOURCES for ep in ep_sizes(source)})
    fig, ax = plt.subplots(figsize=(9.8, 5.8))

    for source in SOURCES:
        eps = ep_sizes(source)
        style = source_styles[source["key"]]
        eps_y = [value_ms(eps_rows.get((source["key"], ep, "none")), stat) for ep in eps]
        ax.plot(
            eps,
            eps_y,
            label=f"EPS / {style['suffix']}",
            color="#9467bd",
            linestyle=style["linestyle"],
            marker="^",
            linewidth=2.2,
            markersize=6.0,
            markerfacecolor="none" if style["hollow"] else "#9467bd",
            markeredgecolor="#9467bd",
            markeredgewidth=1.4,
        )

        opus_rows = source_summary(source)
        for label in labels:
            y = [value_ms(opus_rows.get((ep, OPUS_VARIANT, label)), stat) for ep in eps]
            if not any(v is not None for v in y):
                continue
            latency_style = LATENCY_STYLES.get(label, {"linestyle": "--", "marker": "o"})
            ax.plot(
                eps,
                y,
                label=f"OPUS {label} / {style['suffix']}",
                color=latency_colors.get(label, "#2ca02c"),
                linestyle=style["linestyle"],
                marker=latency_style["marker"],
                linewidth=1.9,
                markersize=5.2,
                markerfacecolor="none" if style["hollow"] else latency_colors.get(label, "#2ca02c"),
                markeredgecolor=latency_colors.get(label, "#2ca02c"),
                markeredgewidth=1.2,
            )

    ax.set_xscale("log", base=2)
    ax.set_xticks(all_eps)
    ax.set_xticklabels([str(ep) for ep in all_eps])
    ax.set_xlabel("EP size")
    ax.set_yscale("log")
    ax.set_ylabel(f"E2E wall time ({stat}, ms, log scale)")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _pos: f"{value:,.0f}" if value >= 10 else f"{value:g}")
    )
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(4.0, 6.0, 7.0, 8.0, 9.0)))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.45)

    ax.legend(frameon=False, fontsize=7.8, ncol=2, loc="upper right")
    fig.suptitle("EPS vs OPUS diverged topology", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_base = plot_dir / f"{baseline['plot_prefix']}_vs_opus_diverged_{stat}_log"
    fig.savefig(out_base.with_suffix(".png"), dpi=200)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[plot] wrote {out_base.with_suffix('.png')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and plot zero-delay EPS baselines on the reconfiguration backend."
    )
    parser.add_argument("action", choices=("run", "compare", "plot", "all"), nargs="?", default="all")
    parser.add_argument("--baseline", choices=tuple(BASELINES), default="constrained")
    parser.add_argument("--rerun", action="store_true", help="rerun selected EPS baseline simulations even if summary rows are ok")
    parser.add_argument("--stat", choices=("min", "max", "avg"), default="max")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = dict(BASELINES[args.baseline])
    baseline["name"] = args.baseline
    if args.action in ("run", "all"):
        run_all(baseline, args.rerun)
    if args.action in ("compare", "all"):
        write_comparison(baseline, args.stat)
    if args.action in ("plot", "all"):
        plot(baseline, args.stat)


if __name__ == "__main__":
    main()
