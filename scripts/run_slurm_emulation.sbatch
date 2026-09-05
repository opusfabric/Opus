#!/usr/bin/env bash
# Site-neutral Slurm launcher for controller-based Opus emulation.
# Submit with, for example:
#   sbatch --nodes=4 --gpus-per-node=4 scripts/run_slurm_emulation.sbatch

#SBATCH --ntasks-per-node=1

set -euo pipefail

SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -z ${OPUS_ROOT:-} ]]; then
    if [[ -f ${SOURCE_DIR}/../README.md ]]; then
        OPUS_ROOT=$(cd -- "${SOURCE_DIR}/.." && pwd)
    elif [[ -n ${SLURM_SUBMIT_DIR:-} && -f ${SLURM_SUBMIT_DIR}/README.md ]]; then
        OPUS_ROOT=$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)
    else
        echo "Set OPUS_ROOT to the shared Opus checkout" >&2
        exit 2
    fi
fi
# sbatch executes a node-local spool copy; workers must recurse through the
# shared checkout instead of BASH_SOURCE's directory.
SCRIPT_PATH=${OPUS_ROOT}/scripts/run_slurm_emulation.sbatch

if [[ ${1:-} == "--worker" ]]; then
    cd "${OPUS_ROOT}/torchtitan"
    exec torchrun \
        --nnodes="${NUM_NODES}" \
        --nproc_per_node="${NUM_RANKS_PER_NODE}" \
        --node_rank="${SLURM_PROCID}" \
        --rdzv_id="${SLURM_JOB_ID:-opus}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        -m "${TRAIN_FILE}" \
        --job.config_file "${CONFIG_FILE}"
fi

: "${SLURM_JOB_NODELIST:?Run this script through sbatch or an existing Slurm allocation}"
NUM_NODES=${NUM_NODES:-${SLURM_JOB_NUM_NODES}}
NUM_RANKS_PER_NODE=${NUM_RANKS_PER_NODE:-${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}}
: "${NUM_RANKS_PER_NODE:?Set NUM_RANKS_PER_NODE to the number of GPUs used per node}"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
CONTROLLER_NODE=${CONTROLLER_NODE:-${NODES[0]}}
CONTROLLER_IP=${CONTROLLER_IP:-${CONTROLLER_NODE}}
SERVER_IPS=${SERVER_IPS:-$(IFS=,; printf '%s' "${NODES[*]}")}

export OPUS_ROOT NUM_NODES NUM_RANKS_PER_NODE CONTROLLER_IP SERVER_IPS
export MASTER_ADDR=${MASTER_ADDR:-${CONTROLLER_IP}}
export MASTER_PORT=${MASTER_PORT:-29500}
export OPUS_CONTROLLER_PORT=${OPUS_CONTROLLER_PORT:-1234}
export RECONFIG_DELAY_MS=${RECONFIG_DELAY_MS:-50}
export NUM_RAILS=${NUM_RAILS:-4}
export MODE=${MODE:-baseline}
export TRAIN_FILE=${TRAIN_FILE:-torchtitan.train}
export CONFIG_FILE=${CONFIG_FILE:-${OPUS_ROOT}/torchtitan/opus-test/dp-2-pp-2-tp-4-pm-8b/llama3_8b_lb.toml}
export COMM_PATTERN_PATH=${COMM_PATTERN_PATH:-${OPUS_ROOT}/torchtitan/opus-test/dp-2-pp-2-tp-4-pm-8b/comm_pattern/comm_pattern}
export OPUS_CONFIG_FILE=${OPUS_CONFIG_FILE:-${OPUS_ROOT}/src/opus-controller/config/config.yaml}
export OPUS_IPC_DIR=${OPUS_IPC_DIR:-${TMPDIR:-/tmp}/opus-${SLURM_JOB_ID}}
export OPUS_IPC_PREFIX=${OPUS_IPC_PREFIX:-opus-${SLURM_JOB_ID}}
export OPUS_OUTPUT_DIR=${OPUS_OUTPUT_DIR:-${OPUS_ROOT}/runs/${SLURM_JOB_ID}}

CONTROLLER_BIN=${CONTROLLER_BIN:-${OPUS_ROOT}/src/opus-controller/controller}
[[ -x "${CONTROLLER_BIN}" ]] || { echo "Missing executable: ${CONTROLLER_BIN}" >&2; exit 1; }
[[ -f "${CONFIG_FILE}" ]] || { echo "Missing CONFIG_FILE: ${CONFIG_FILE}" >&2; exit 1; }
[[ -d "${COMM_PATTERN_PATH}" || -f "${COMM_PATTERN_PATH}_0_rank0.txt" ]] || {
    echo "Missing communication patterns at: ${COMM_PATTERN_PATH}" >&2
    exit 1
}
mkdir -p "${OPUS_IPC_DIR}" "${OPUS_OUTPUT_DIR}"

cleanup() {
    kill "${CONTROLLER_STEP_PID:-}" 2>/dev/null || true
    wait "${CONTROLLER_STEP_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting Opus controller on ${CONTROLLER_NODE}"
srun --nodes=1 --ntasks=1 --nodelist="${CONTROLLER_NODE}" --export=ALL \
    "${CONTROLLER_BIN}" \
    -I 0.0.0.0 -d "${RECONFIG_DELAY_MS}" -e -r "${NUM_RAILS}" \
    -o "${OPUS_OUTPUT_DIR}" \
    >"${OPUS_OUTPUT_DIR}/controller.log" 2>&1 &
CONTROLLER_STEP_PID=$!

sleep "${CONTROLLER_STARTUP_WAIT:-2}"
if ! kill -0 "${CONTROLLER_STEP_PID}" 2>/dev/null; then
    cat "${OPUS_OUTPUT_DIR}/controller.log" >&2 || true
    exit 1
fi

echo "Starting ${NUM_NODES} training tasks with ${NUM_RANKS_PER_NODE} GPUs per node"
srun --nodes="${NUM_NODES}" \
    --ntasks="${NUM_NODES}" \
    --ntasks-per-node=1 \
    --gpus-per-task="${NUM_RANKS_PER_NODE}" \
    --export=ALL \
    --output="${OPUS_OUTPUT_DIR}/torchrun_%N.log" \
    bash "${SCRIPT_PATH}" --worker
