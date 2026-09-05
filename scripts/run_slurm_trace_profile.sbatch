#!/usr/bin/env bash
# Site-neutral Slurm launcher for PyTorch Chrome traces.
# Run this in the CUDA/PyTorch runtime used by the training job.  It does not
# start the Opus controller and does not require Shifter.

#SBATCH --ntasks-per-node=1

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OPUS_ROOT=${OPUS_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
SCRIPT_PATH=${OPUS_ROOT}/scripts/run_slurm_trace_profile.sbatch

if [[ ${1:-} == "--worker" ]]; then
    cd "${OPUS_ROOT}/torchtitan"
    exec torchrun \
        --nnodes="${NUM_NODES}" \
        --nproc_per_node="${NUM_RANKS_PER_NODE}" \
        --node_rank="${SLURM_PROCID}" \
        --rdzv_id="${SLURM_JOB_ID:-opus-trace}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        -m "${TRAIN_FILE}" \
        --job.config_file="${CONFIG_FILE}" \
        --job.dump_folder="${TRACE_ROOT}" \
        --profiling.enable_profiling \
        --profiling.profile_freq="${PROFILE_FREQ}" \
        --profiling.save_traces_folder=profile_trace \
        --training.steps="${PROFILE_STEPS}" \
        --training.no-enable-opus-backend
fi

: "${SLURM_JOB_NODELIST:?Run this script through sbatch or an existing Slurm allocation}"
: "${CONFIG_FILE:?Set CONFIG_FILE to a local Torchtitan TOML configuration}"

NUM_NODES=${NUM_NODES:-${SLURM_JOB_NUM_NODES}}
NUM_RANKS_PER_NODE=${NUM_RANKS_PER_NODE:-${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}}
: "${NUM_RANKS_PER_NODE:?Set NUM_RANKS_PER_NODE to the number of GPUs used per node}"
TRAIN_FILE=${TRAIN_FILE:-torchtitan.train}
PROFILE_FREQ=${PROFILE_FREQ:-10}
PROFILE_STEPS=${PROFILE_STEPS:-10}
TRACE_ROOT=${TRACE_ROOT:-${OPUS_ROOT}/runs/${SLURM_JOB_ID}/figure4}

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
MASTER_ADDR=${MASTER_ADDR:-${NODES[0]}}
MASTER_PORT=${MASTER_PORT:-29500}

export OPUS_ROOT CONFIG_FILE NUM_NODES NUM_RANKS_PER_NODE TRAIN_FILE
export PROFILE_FREQ PROFILE_STEPS TRACE_ROOT MASTER_ADDR MASTER_PORT
mkdir -p "${TRACE_ROOT}"

srun --nodes="${NUM_NODES}" \
    --ntasks="${NUM_NODES}" \
    --ntasks-per-node=1 \
    --gpus-per-task="${NUM_RANKS_PER_NODE}" \
    --export=ALL \
    --output="${TRACE_ROOT}/torchrun_%N.log" \
    bash "${SCRIPT_PATH}" --worker
