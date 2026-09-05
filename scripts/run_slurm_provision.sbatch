#!/usr/bin/env bash
# Slurm entry point for Opus look-ahead provisioning.

#SBATCH --ntasks-per-node=1

set -euo pipefail

: "${OPUS_ROOT:=${SLURM_SUBMIT_DIR:-}}"
: "${OPUS_ROOT:?Set OPUS_ROOT to the shared Opus checkout}"
MODE=provision exec bash "${OPUS_ROOT}/scripts/run_slurm_emulation.sbatch" "$@"
