#!/usr/bin/env bash
# Perlmutter/Shifter entry point for Opus look-ahead provisioning.

#SBATCH --ntasks-per-node=1

set -euo pipefail

: "${OPUS_ROOT:?Set OPUS_ROOT to the shared Opus checkout}"
MODE=provision exec bash \
    "${OPUS_ROOT}/scripts/perlmutter/run_opus_emulation.sbatch" "$@"
