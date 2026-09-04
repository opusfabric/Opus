"""Run the reconfigurable backend at several reconfiguration delays."""

import argparse
import os
import subprocess

import yaml


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

    The zero-delay baseline result is the EPS reference; provisioning is skipped
    at zero delay. All runs use the reconfigurable simulator and
    `run_network_reconfig.sh`.
    """
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


def write_result_for_sheet_import(results, filename):
    zero_delay = next(
        (data for seconds, data in results.items() if abs(seconds) < 1e-12),
        None,
    )
    if zero_delay is None:
        raise ValueError("reconfig_times must include 0 for the EPS reference")

    with open(filename, "w") as stream:
        stream.write("EPS\n")
        eps = zero_delay["no_provision"]
        stream.write(
            f"{eps['sys0_cycles']}\t{eps['cycles']}\t"
            f"{eps['exposed_cycles']}\n"
        )

        stream.write("\nReconfigurable\n")
        for seconds, data in results.items():
            if abs(seconds) < 1e-12:
                continue
            label = _latency_label(seconds)
            for mode in ("no_provision", "provision"):
                result = data[mode]
                stream.write(
                    f"{label}\t{result['sys0_cycles']}\t"
                    f"{result['cycles']}\t{result['exposed_cycles']}\n"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Run reconfigurable-backend experiments with different delays"
    )
    parser.add_argument(
        "base_dir",
        help="Directory containing network.yml and run_network_reconfig.sh",
    )
    parser.add_argument(
        "--reconfig-times",
        default="0,0.01,0.05,0.1,0.25,0.5,0.75,1",
        help="Comma-separated delay values in seconds; include 0 for EPS",
    )
    args = parser.parse_args()

    reconfig_times = [float(value) for value in args.reconfig_times.split(",")]
    results = run_with_reconfig_times(reconfig_times, args.base_dir)

    eps = next(
        data["no_provision"]
        for seconds, data in results.items()
        if abs(seconds) < 1e-12
    )
    print(f"EPS (0 s) - Cycles: {eps['cycles']}, Exposed Cycles: {eps['exposed_cycles']}")
    for seconds, data in results.items():
        print(f"Reconfig Time: {seconds} seconds")
        print(
            f"  Baseline    - Cycles: {data['no_provision']['cycles']}, "
            f"Exposed Cycles: {data['no_provision']['exposed_cycles']}"
        )
        if data["provision"] is None:
            print("  Provision   - skipped at zero delay")
        else:
            provision_cycles = data["provision"]["cycles"]
            provision_exposed = data["provision"]["exposed_cycles"]
            print(
                f"  Provision   - Cycles: {provision_cycles}, "
                f"Exposed Cycles: {provision_exposed}"
            )

    output_path = os.path.join(args.base_dir, "results_for_sheet_import.txt")
    write_result_for_sheet_import(results, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
