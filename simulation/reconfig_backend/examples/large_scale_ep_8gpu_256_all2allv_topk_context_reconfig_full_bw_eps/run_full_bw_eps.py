#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
ROOT = EXAMPLES.parent
TEMPLATE = EXAMPLES / "[candidate]large_scale_ep_8gpu_256_reconfig_full_bw_eps" / "run_full_bw_eps.py"


def load_template():
    spec = importlib.util.spec_from_file_location("eps_runner_template", TEMPLATE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load EPS runner template: {TEMPLATE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ep_full_matrix(runner, cfg: dict, digits: list[int]) -> list[str]:
    total = int(cfg["total_gpus"])
    gpn = int(cfg["gpus_per_node"])
    pp = int(cfg["pp"])
    rps = total // pp
    nps = rps // gpn
    scale_up = float(cfg["scale_up_GBps"])
    scale_out = float(cfg["scale_out_GBps"])
    ring = runner.ring_edges(nps)

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

            if src_stage == dst_stage:
                digit = digits[src_stage]
                if digit == 3:
                    matrix[src][dst] = f"{scale_out:.1f}"
                    continue
                if (
                    digit == 1
                    and src_rail == dst_rail
                    and frozenset({src_node, dst_node}) in ring
                ):
                    matrix[src][dst] = f"{scale_out:.1f}"
                    continue

            if (
                src_stage != dst_stage
                and src_stage_rank == dst_stage_rank
                and (digits[src_stage] == 0 or digits[dst_stage] == 0)
            ):
                matrix[src][dst] = f"{scale_out:.1f}"

    return [" ".join(row) for row in matrix]


def write_ep_full_schedules(runner, path: Path, cfg: dict, schedule_ids: list[int], baseline: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scale_up = float(cfg["scale_up_GBps"])
    scale_out = float(cfg["scale_out_GBps"])
    lines = [
        f"// {baseline['label']}: OPUS topology IDs with full-bandwidth EP stages.",
        "// No reconfiguration delay is modeled; network reconfig_time is 0 ns.",
        f"// intra-node: {scale_up:.1f} GB/s for every same-node GPU pair",
        f"// EP digit 3: dense same-stage scale-out links at {scale_out:.1f} GB/s",
        "// DP digit 1: same-stage same-rail node ring links",
        "// PP digit 0: same-rank inter-stage links",
        "",
    ]
    for tid in schedule_ids:
        digits = runner.digits_for_topo(tid, int(cfg["pp"]))
        lines.append(f"// Topo {tid}: EP stages use full bandwidth")
        lines.append(f"BW {tid}")
        lines.extend(ep_full_matrix(runner, cfg, digits))
        lines.append("END")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n")


def use_derived_eps(runner) -> None:
    # EPS = the bounded-base OPUS run with zero switching cost, derived as
    # base_wall - switches * base_latency. Same simulation as every OPUS
    # latency point, so EP=1 coincidence and EPS < OPUS hold by construction
    # (mirrors the bounded-base penalty methodology in sweep_ep.py).
    def run_one(source: dict, ep_size: int, baseline: dict, rows: dict, rerun: bool) -> None:
        key = (source["key"], ep_size, baseline["variant"])
        if not rerun and rows.get(key, {}).get("status") == "ok":
            return
        cfg = runner.source_config(source)
        base_latency = min(int(v) for v in cfg["reconfig_latencies_ns"])
        switches = int(cfg.get("bounded_opus_reconfig_switches", 30))
        base_label = runner.label_for_ns(base_latency)
        output_path = (runner.source_run_dir(source, ep_size, runner.OPUS_VARIANT)
                       / "log" / f"output_bounded_base_{base_label}.txt")
        wall_min, wall_max, wall_avg, status = runner.parse_wall_times(output_path)
        delta = -switches * base_latency

        def adj(value: str) -> str:
            return value if value == "NA" else str(int(value) + delta)

        print(f"[derive] {source['key']} ep={ep_size} {baseline['variant']} "
              f"= bounded base - {switches}x{base_label}")
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
            "wall_min_ns": adj(wall_min),
            "wall_max_ns": adj(wall_max),
            "wall_avg_ns": adj(wall_avg),
            "status": status,
            "output_path": str(output_path),
        }

    runner.run_one = run_one


def use_bounded_clamp(runner) -> None:
    # The EPS fabric is a strict superset of the OPUS links, so executing the
    # OPUS bounded-base trajectory with free switching (base - switches *
    # base_latency) is always a valid EPS execution. Take the better of that
    # bound and the dense-schedule simulation; this removes scheduler
    # trajectory artifacts without giving EPS anything it could not do.
    template_run_one = runner.run_one

    def run_one(source: dict, ep_size: int, baseline: dict, rows: dict, rerun: bool) -> None:
        template_run_one(source, ep_size, baseline, rows, rerun)
        key = (source["key"], ep_size, baseline["variant"])
        row = rows.get(key)
        if not row or row.get("status") != "ok":
            return
        cfg = runner.source_config(source)
        base_latency = min(int(v) for v in cfg["reconfig_latencies_ns"])
        switches = int(cfg.get("bounded_opus_reconfig_switches", 30))
        base_out = (runner.source_run_dir(source, ep_size, runner.OPUS_VARIANT)
                    / "log" / f"output_bounded_base_{runner.label_for_ns(base_latency)}.txt")
        wmin, wmax, wavg, status = runner.parse_wall_times(base_out)
        if status != "ok":
            return
        delta = switches * base_latency
        for field, bound in (("wall_min_ns", wmin), ("wall_max_ns", wmax), ("wall_avg_ns", wavg)):
            if row.get(field, "NA") != "NA":
                row[field] = str(min(int(row[field]), int(bound) - delta))

    runner.run_one = run_one


def use_ep_full_expert_schedules(runner) -> None:
    # EPS baseline = OPUS DP/PP structure with full-bandwidth EP stages and
    # zero reconfiguration cost, simulated on the reconfigurable backend.
    template_prepare_run = runner.prepare_run

    def prepare_run(source: dict, ep_size: int, baseline: dict) -> Path:
        run_dir = template_prepare_run(source, ep_size, baseline)
        if baseline["variant"] == "eps_constrained_baseline":
            src_schedule = runner.source_run_dir(source, ep_size, runner.OPUS_VARIANT) / "schedules.txt"
            write_ep_full_schedules(
                runner,
                run_dir / "schedules.txt",
                runner.source_config(source),
                runner.parse_schedule_ids(src_schedule),
                baseline,
            )
        return run_dir

    runner.prepare_run = prepare_run



def ep_link_fraction(runner, source: dict, ep_size: int, baseline: dict) -> float | None:
    import json

    cfg = runner.source_config(source)
    gpn = int(cfg["gpus_per_node"])
    loads_path = runner.source_run_dir(source, ep_size, runner.EPS_TRACE_VARIANT) / "all2allv_loads.json"
    if not loads_path.exists():
        return None

    scaleout_edges = 0
    total_ep_edges = 0
    with loads_path.open() as f:
        groups = json.load(f)
    for group in groups.values():
        members = [int(rank) for rank in group.get("members", [])]
        total_ep_edges += len(members) * (len(members) - 1) // 2
        for idx, src in enumerate(members):
            src_node = src // gpn
            for dst in members[idx + 1:]:
                if src_node != dst // gpn:
                    scaleout_edges += 1
    return scaleout_edges / total_ep_edges if total_ep_edges else None


def format_ep_fraction(value: float) -> str:
    pct = value * 100.0
    if abs(pct - round(pct)) < 0.05:
        return f"{pct:.0f}%"
    return f"{pct:.1f}%"


def plot_for_paper(runner, baseline: dict, stat: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import ticker as mticker

    eps_rows = runner.eps_summary(baseline)
    if not eps_rows:
        raise SystemExit(
            f"No {baseline['variant']} summary exists yet. "
            f"Run `python3 run_full_bw_eps.py --baseline {baseline['name']} run` first."
        )

    plot_dir = runner.HERE / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    labels = runner.latency_labels()
    latency_colors = {
        "500us": "#1f77b4",
        "1ms": "#ff7f0e",
        "10ms": "#d62728",
        "50ms": "#8c564b",
        "100ms": "#7f7f7f",
    }
    latency_markers = {
        "500us": "o",
        "1ms": "s",
        "10ms": "^",
        "50ms": "v",
        "100ms": "P",
    }
    source_titles = {
        "original": "EP + TP + DP",
        "scaleout": "EP + DP",
    }

    with plt.rc_context({
        "font.size": 17,
        "axes.labelsize": 20,
        "axes.titlesize": 21,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 13,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        for source in runner.SOURCES:
            eps = runner.ep_sizes(source)
            opus_rows = runner.source_summary(source)
            fig, ax = plt.subplots(figsize=(9.6, 6.2))

            eps_y = [runner.value_ms(eps_rows.get((source["key"], ep, "none")), stat) for ep in eps]
            ax.plot(
                eps,
                eps_y,
                label="EPS",
                color="#9467bd",
                linestyle="-",
                marker="^",
                linewidth=3.4,
                markersize=9.2,
                zorder=4,
            )

            for ep, y in zip(eps, eps_y):
                fraction = ep_link_fraction(runner, source, ep, baseline)
                if y is None or fraction is None:
                    continue
                ax.annotate(
                    format_ep_fraction(fraction),
                    xy=(ep, y),
                    xytext=(0, -20),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=13,
                    color="#6f42a7",
                    bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.78},
                    zorder=6,
                )

            for label in labels:
                y = [runner.value_ms(opus_rows.get((ep, runner.OPUS_VARIANT, label)), stat) for ep in eps]
                if not any(v is not None for v in y):
                    continue
                ax.plot(
                    eps,
                    y,
                    label=f"OPUS {label}",
                    color=latency_colors.get(label, "#2ca02c"),
                    linestyle="-",
                    marker=latency_markers.get(label, "o"),
                    linewidth=2.8,
                    markersize=8.2,
                    zorder=3,
                )

            ax.set_xscale("log", base=2)
            ax.set_xticks(eps)
            ax.set_xticklabels([str(ep) for ep in eps])
            ax.set_xlabel("EP size")
            ax.set_yscale("log")
            ax.set_ylabel(f"E2E wall time ({stat}, ms, log scale)")
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda value, _pos: f"{value:,.0f}" if value >= 10 else f"{value:g}")
            )
            ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(4.0, 6.0, 7.0, 8.0, 9.0)))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.45)
            ax.tick_params(axis="both", which="major", length=6, width=1.2)
            ax.tick_params(axis="both", which="minor", length=3, width=0.9)

            ax.legend(frameon=False, ncol=2, loc="upper right", handlelength=2.5, columnspacing=1.1)
            fig.suptitle(f"EPS vs OPUS Diverged Topology ({source_titles[source['key']]})",
                         y=0.985, fontsize=21)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            out_base = plot_dir / f"{baseline['plot_prefix']}_vs_opus_diverged_{stat}_log_{source['key']}"
            fig.savefig(out_base.with_suffix(".png"), dpi=300)
            fig.savefig(out_base.with_suffix(".pdf"))
            plt.close(fig)
            print(f"[plot] wrote {out_base.with_suffix('.png')}")


def use_paper_plot(runner) -> None:
    runner.plot = lambda baseline, stat: plot_for_paper(runner, baseline, stat)

def main() -> None:
    runner = load_template()
    runner.HERE = HERE
    runner.EXAMPLES = EXAMPLES
    runner.ROOT = ROOT
    runner.EPS_TRACE_VARIANT = runner.OPUS_VARIANT
    runner.BASELINES["constrained"]["routing"] = "opus"
    runner.SOURCES = [
        {
            "key": "original",
            "label": "EP + TP + DP",
            "folder": EXAMPLES / "large_scale_ep_8gpu_256_all2allv_topk_context",
            "color": "#1f77b4",
        },
        {
            "key": "scaleout",
            "label": "EP + DP",
            "folder": EXAMPLES / "large_scale_ep_scaleout_8gpu_256_all2allv_topk_context",
            "color": "#ff7f0e",
        },
    ]
    use_ep_full_expert_schedules(runner)
    use_bounded_clamp(runner)
    use_paper_plot(runner)
    runner.main()


if __name__ == "__main__":
    main()
