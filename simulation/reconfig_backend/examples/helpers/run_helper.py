"""Run the reconfigurable backend at several reconfiguration delays."""


import argparse
import glob
import json
import os
import resource
import subprocess

import yaml


def prepare_simulation(base_dir):
    """Validate inputs and reserve one trace descriptor per simulated rank."""
    required_files = (
        "network.yml", "system.json", "remote_memory.json", "workload.json",
        "schedules.txt", "run_network_reconfig.sh",
    )
    for filename in required_files:
        path = os.path.join(base_dir, filename)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise RuntimeError(f"missing or empty simulator input: {path}")

    for filename in ("system.json", "remote_memory.json", "workload.json"):
        path = os.path.join(base_dir, filename)
        try:
            with open(path, "r") as stream:
                json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid simulator JSON input {path}: {error}") from error

    network_path = os.path.join(base_dir, "network.yml")
    with open(network_path, "r") as stream:
        network = yaml.safe_load(stream)
    try:
        npus_count = int(network["npus_count"][0])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError(f"invalid npus_count in {network_path}") from error

    traces = glob.glob(os.path.join(base_dir, "workload.*.et"))
    nonempty_traces = sum(os.path.getsize(path) > 0 for path in traces)
    if nonempty_traces < npus_count:
        raise RuntimeError(
            f"incomplete workload generation in {base_dir}: expected "
            f"{npus_count} nonempty rank traces, found {nonempty_traces}"
        )

    # ETFeeder keeps every rank trace open. A 1024-descriptor Docker limit
    # otherwise causes a misleading nlohmann empty-JSON parse error.
    required_nofile = npus_count + 256
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit < required_nofile:
        if hard_limit < required_nofile:
            raise RuntimeError(
                f"open-file limit is too low for {npus_count} simulated GPUs "
                f"(soft={soft_limit}, hard={hard_limit}, need at least "
                f"{required_nofile}). Start Docker with "
                f"--ulimit nofile=65536:65536"
            )
        resource.setrlimit(resource.RLIMIT_NOFILE, (required_nofile, hard_limit))
        print(f"Raised open-file soft limit to {required_nofile}")


def extract_last_cycles(file_path):
    """Return the last and sys[0] cycle counts from an ASTRA-Sim log."""
    last_cycles = None
    last_exposed = None
    sys0_cycles = None
    sys0_exposed = None

    with open(file_path, "r") as stream:
        for line in stream:
            if "finished" not in line or "exposed communication" not in line:
                continue

            payload = line.split("[info]")[-1].strip()
            parts = payload.split(",")
            cycles = int(parts[1].strip().split()[0])
            exposed_cycles = int(parts[2].strip().split()[2])

            last_cycles = cycles
            last_exposed = exposed_cycles
            if "sys[0] finished" in payload:
                sys0_cycles = cycles
                sys0_exposed = exposed_cycles

    return last_cycles, last_exposed, sys0_cycles, sys0_exposed


def _result(log_path):
    cycles, exposed, sys0_cycles, sys0_exposed = extract_last_cycles(log_path)
    if cycles is None or sys0_cycles is None:
        raise RuntimeError(
            f"simulator did not emit complete finish records in {log_path}"
        )
    return {
        "cycles": cycles,
        "exposed_cycles": exposed,
        "sys0_cycles": sys0_cycles,
        "sys0_exposed_cycles": sys0_exposed,
    }


def _latency_label(seconds):
    if seconds < 1:
        return f"{int(round(seconds * 1000))} ms"
    return f"{seconds:g} s"


def run_with_reconfig_times(reconfig_times, base_dir):
    """Run the reconfigurable simulator for every delay in seconds.

    The zero-delay result is the EPS reference; provisioning is skipped at zero
    delay. All runs use the reconfigurable simulator and run_network_reconfig.sh.
    """
    prepare_simulation(base_dir)
    results = {}
    network_file = os.path.join(base_dir, "network.yml")

    for seconds in reconfig_times:
        with open(network_file, "r") as stream:
            network_config = yaml.safe_load(stream)

        network_config["reconfig_time"] = [seconds * 1e9]
        with open(network_file, "w") as stream:
            yaml.safe_dump(network_config, stream, sort_keys=False)

        is_zero_delay = abs(seconds) < 1e-12
        environment = os.environ.copy()
        if is_zero_delay:
            environment["OPUS_SKIP_PROVISION"] = "1"

        print(f"Running reconfigurable backend with reconfig_time: {seconds} s")
        subprocess.run(
            ["bash", "run_network_reconfig.sh"],
            cwd=base_dir,
            check=True,
            env=environment,
        )

        results[seconds] = {
            "no_provision": _result(os.path.join(base_dir, "debug_no_provision.txt")),
            "provision": (
                None
                if is_zero_delay
                else _result(os.path.join(base_dir, "debug_provision.txt"))
            ),
        }

    return results


def run_analytical(base_dir):
    """Run the analytical and bandwidth-balanced baseline simulations."""
    print("Running analytical and baseline backends")
    subprocess.run(
        ["bash", "run_network_analytical.sh"],
        cwd=base_dir,
        check=True,
        env=os.environ.copy(),
    )
    analytical_log = os.path.join(base_dir, "debug_analytical.txt")
    baseline_log = os.path.join(base_dir, "debug_baseline.txt")
    if not os.path.isfile(analytical_log) or not os.path.isfile(baseline_log):
        raise RuntimeError("Analytical launcher did not produce both result logs")
    return _result(analytical_log), _result(baseline_log)


def _write_result(stream, result):
    stream.write(
        "{}\t{}\t{}\n".format(
            result["sys0_cycles"],
            result["cycles"],
            result["exposed_cycles"],
        )
    )


def write_result_for_sheet_import(results, filename, analytical=None, baseline=None):
    zero_delay = next(
        (data for seconds, data in results.items() if abs(seconds) < 1e-12),
        None,
    )
    if zero_delay is None:
        raise ValueError("reconfig_times must include 0 for the EPS reference")

    with open(filename, "w") as stream:
        if analytical is None:
            stream.write("EPS\n")
            _write_result(stream, zero_delay["no_provision"])
        else:
            stream.write("Analytical\n")
            _write_result(stream, analytical)
            if baseline is not None:
                stream.write("\nBaseline\n")
                _write_result(stream, baseline)

        stream.write("\nReconfigurable\n")
        for seconds, data in results.items():
            if abs(seconds) < 1e-12:
                continue
            label = _latency_label(seconds)
            for mode in ("no_provision", "provision"):
                result = data[mode]
                stream.write(
                    "{}\t{}\t{}\t{}\n".format(
                        label,
                        result["sys0_cycles"],
                        result["cycles"],
                        result["exposed_cycles"],
                    )
                )


def main():
    parser = argparse.ArgumentParser(
        description="Run reconfigurable-backend experiments with different delays"
    )
    parser.add_argument(
        "base_dir",
        help="Directory containing the simulator run scripts and configuration files",
    )
    parser.add_argument(
        "--reconfig-times",
        default="0,0.01,0.05,0.1,0.25,0.5,0.75,1",
        help="Comma-separated delay values in seconds; include 0 for EPS",
    )
    parser.add_argument(
        "--include-analytical",
        action="store_true",
        help="Also run the analytical and bandwidth-balanced baseline backends",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Result filename or path. Relative paths are resolved under base_dir; "
            "the default is results_for_sheet_import.txt"
        ),
    )
    args = parser.parse_args()

    reconfig_times = [float(value) for value in args.reconfig_times.split(",")]
    results = run_with_reconfig_times(reconfig_times, args.base_dir)

    eps = next(
        data["no_provision"]
        for seconds, data in results.items()
        if abs(seconds) < 1e-12
    )
    print(
        "EPS (0 s) - Cycles:",
        eps["cycles"],
        "Exposed Cycles:",
        eps["exposed_cycles"],
    )
    for seconds, data in results.items():
        print(f"Reconfig Time: {seconds} seconds")
        print(
            "  Baseline    - Cycles:",
            data["no_provision"]["cycles"],
            "Exposed Cycles:",
            data["no_provision"]["exposed_cycles"],
        )
        if data["provision"] is None:
            print("  Provision   - skipped at zero delay")
        else:
            provision_cycles = data["provision"]["cycles"]
            provision_exposed = data["provision"]["exposed_cycles"]
            print(
                "  Provision   - Cycles:",
                provision_cycles,
                "Exposed Cycles:",
                provision_exposed,
            )

    analytical = baseline = None
    if args.include_analytical:
        analytical, baseline = run_analytical(args.base_dir)
        print(
            "Analytical - Cycles:",
            analytical["cycles"],
            "Exposed Cycles:",
            analytical["exposed_cycles"],
        )
        print(
            "Baseline   - Cycles:",
            baseline["cycles"],
            "Exposed Cycles:",
            baseline["exposed_cycles"],
        )

    output_path = args.output or "results_for_sheet_import.txt"
    if not os.path.isabs(output_path):
        output_path = os.path.join(args.base_dir, output_path)
    write_result_for_sheet_import(results, output_path, analytical, baseline)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
