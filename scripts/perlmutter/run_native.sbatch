#!/usr/bin/env bash
# Perlmutter native NCCL control for the Opus emulation experiments.

#SBATCH --ntasks-per-node=1

set -euo pipefail

: "${SLURM_JOB_NODELIST:?Run this script through sbatch}"
: "${OPUS_ROOT:?Set OPUS_ROOT to the shared Opus checkout}"
: "${OPUS_SHIFTER_IMAGE:=ericd16/opus:2.0}"
NUM_NODES=${SLURM_JOB_NUM_NODES}
NUM_RANKS_PER_NODE=${NUM_RANKS_PER_NODE:-${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}}
: "${NUM_RANKS_PER_NODE:?Set NUM_RANKS_PER_NODE}"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
MASTER_ADDR=${MASTER_ADDR:-${NODES[0]}}
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG_FILE=${CONFIG_FILE:-${OPUS_ROOT}/torchtitan/opus-test/dp-2-pp-2-tp-4-pm-8b/llama3_8b_lb.toml}
OUTPUT_DIR=${OUTPUT_DIR:-${OPUS_ROOT}/runs/${SLURM_JOB_ID}}
mkdir -p "${OUTPUT_DIR}"

export OPUS_ROOT NUM_NODES NUM_RANKS_PER_NODE MASTER_ADDR MASTER_PORT CONFIG_FILE
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-rocep}
export NCCL_IB_MERGE_NICS=${NCCL_IB_MERGE_NICS:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-0}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-25}
export NCCL_LAUNCH_ORDER_IMPLICIT=${NCCL_LAUNCH_ORDER_IMPLICIT:-1}
export NCCL_COMM_BLOCKING=${NCCL_COMM_BLOCKING:-1}

srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 \
    --gpus-per-task="${NUM_RANKS_PER_NODE}" --export=ALL \
    --output="${OUTPUT_DIR}/torchrun_%N.log" \
    shifter --image="${OPUS_SHIFTER_IMAGE}" --module=gpu bash -c '
      cd "${OPUS_ROOT}/torchtitan"
      exec torchrun \
        --nnodes="${NUM_NODES}" \
        --nproc_per_node="${NUM_RANKS_PER_NODE}" \
        --node_rank="${SLURM_PROCID}" \
        --rdzv_id="${SLURM_JOB_ID}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        -m torchtitan.train \
        --job.config_file "${CONFIG_FILE}" \
        --training.no-enable-opus-backend
    '
