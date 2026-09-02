#!/usr/bin/env python3
"""Compile per-rank Opus event traces into keyed reconfiguration schedules."""

import argparse
import csv
from pathlib import Path


def read_events(path: Path):
    events = []
    with path.open(newline="") as stream:
        for row in csv.reader(stream):
            if not row or row[0].startswith("#"):
                continue
            (
                iteration,
                sequence,
                backend,
                counter,
                kind,
                collective,
                controller_backend,
                controller_counter,
            ) = row
            events.append(
                {
                    "iteration": int(iteration),
                    "sequence": int(sequence),
                    "backend": int(backend),
                    "counter": int(counter),
                    "kind": kind,
                    "collective": collective,
                    "controller_backend": int(controller_backend),
                    "controller_counter": int(controller_counter),
                }
            )
    return sorted(events, key=lambda event: event["sequence"])


def transitions(events, provision: bool):
    result = []
    previous = None
    for event in events:
        changed = previous is None or event["kind"] != previous["kind"]
        if changed:
            trigger = previous if provision and previous is not None else event
            # Do not move provisioning across an iteration boundary.
            if trigger["iteration"] != event["iteration"]:
                trigger = event
            result.append((trigger, event))
        previous = event
    return result


def transition_keys(all_events):
    """Return the union of boundaries observed by any participating rank."""
    keys = set()
    for events in all_events:
        previous = None
        for event in events:
            if previous is None or event["kind"] != previous["kind"]:
                keys.add((event["iteration"], event["backend"], event["counter"]))
            previous = event
    return keys


def synchronized_transitions(events, keys, provision: bool):
    """Emit every global boundary present in this rank's event stream."""
    result = []
    previous = None
    for event in events:
        key = (event["iteration"], event["backend"], event["counter"])
        if key in keys:
            trigger = previous if provision and previous is not None else event
            if trigger["iteration"] != event["iteration"]:
                trigger = event
            result.append((trigger, event))
        previous = event
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--mode", choices=("baseline", "provision"), required=True)
    args = parser.parse_args()

    inputs = sorted(args.profile_dir.glob("events_*_rank*.csv"))
    if not inputs:
        parser.error(f"no events_*_rank*.csv files found in {args.profile_dir}")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    traces = {source: read_events(source) for source in inputs}
    keys = transition_keys(traces.values())
    total = 0
    for source, events in traces.items():
        suffix = source.name.removeprefix("events_")
        destination = Path(f"{args.output_prefix}_{suffix}")
        rows = synchronized_transitions(events, keys, args.mode == "provision")
        with destination.open("w", newline="") as stream:
            stream.write("# iteration,trigger_backend,trigger_counter,target_backend,target_counter\n")
            writer = csv.writer(stream)
            for trigger, target in rows:
                writer.writerow(
                    (
                        target["iteration"],
                        trigger["backend"],
                        trigger["counter"],
                        target["controller_backend"],
                        target["controller_counter"],
                    )
                )
        total += len(rows)
        print(f"{source.name}: {len(rows)} transitions -> {destination}")
    print(f"compiled {total} transitions from {len(inputs)} rank traces")


if __name__ == "__main__":
    main()
