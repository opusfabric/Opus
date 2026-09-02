#!/usr/bin/env bash
set -euo pipefail
: "${OPUS_ROOT:?Set OPUS_ROOT to the shared Opus checkout}"
export MODE=profile
export OPUS_PROFILE_ITERATIONS=${OPUS_PROFILE_ITERATIONS:-1:2:3}
export OPUS_PROFILE_DIR=${OPUS_PROFILE_DIR:-${OPUS_ROOT}/runs/${SLURM_JOB_ID}/profile}
mkdir -p "${OPUS_PROFILE_DIR}"
exec bash "${OPUS_ROOT}/scripts/perlmutter/run_opus_emulation.sbatch" "$@"
