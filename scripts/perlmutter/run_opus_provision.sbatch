#!/usr/bin/env bash
# Perlmutter/Shifter entry point for Opus look-ahead provisioning.

#SBATCH --ntasks-per-node=1

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=provision exec "${SCRIPT_DIR}/run_opus_emulation.sbatch" "$@"
