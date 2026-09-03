#!/usr/bin/env bash
# Check completion and Opus control-plane health for one or more run directories.

set -euo pipefail

if (( $# == 0 )); then
    echo "Usage: $0 JOB_ID [JOB_ID ...]" >&2
    exit 2
fi

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

for job_id in "$@"; do
    run_dir="${ROOT}/runs/${job_id}"
    if [[ ! -d "${run_dir}" ]]; then
        echo "${job_id}: FAIL (missing ${run_dir})"
        continue
    fi

    shopt -s nullglob
    worker_logs=("${run_dir}"/torchrun_*.log)
    shopt -u nullglob
    if (( ${#worker_logs[@]} == 0 )); then
        echo "${job_id}: FAIL (no worker logs)"
        continue
    fi

    step10=$(awk '/step: 10/ {count++} END {print count + 0}' "${worker_logs[@]}")
    fatal=$(awk 'BEGIN {IGNORECASE=1} /Traceback|ChildFailedError|NCCL error|RuntimeError/ {count++} END {print count + 0}' "${worker_logs[@]}")

    if [[ -f "${run_dir}/controller.log" ]]; then
        timeouts=$(awk '/CONFIG_ACK timeout/ {count++} END {print count + 0}' "${worker_logs[@]}")
        ready=$(grep -c 'ALL READY' "${run_dir}/controller.log" || true)
        changes=$(grep -c 'new topo: yes' "${run_dir}/controller.log" || true)
        printf '%s: %s; step10=%d/16, ACK_timeouts=%d, ALL_READY=%d, topology_changes=%d, fatal_markers=%d\n' \
            "${job_id}" "$([[ ${step10} -eq 16 && ${timeouts} -eq 0 && ${fatal} -eq 0 ]] && echo PASS || echo FAIL)" \
            "${step10}" "${timeouts}" "${ready}" "${changes}" "${fatal}"
    else
        opus_markers=$(awk '/CONFIG_REQ|CONFIG_ACK|PROVISION topo|Backend [0-9]+ RANK/ {count++} END {print count + 0}' "${worker_logs[@]}")
        printf '%s: %s (native); step10=%d/16, Opus_markers=%d, fatal_markers=%d\n' \
            "${job_id}" "$([[ ${step10} -eq 16 && ${opus_markers} -eq 0 && ${fatal} -eq 0 ]] && echo PASS || echo FAIL)" \
            "${step10}" "${opus_markers}" "${fatal}"
    fi
done
