#!/usr/bin/env bash
# Slurm entry point for Opus look-ahead provisioning.

#SBATCH --ntasks-per-node=1

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=provision exec "${SCRIPT_DIR}/run_slurm_emulation.sbatch" "$@"
