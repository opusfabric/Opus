#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(dirname "$(realpath "$0")")
OPUS_ROOT="$(realpath ${SCRIPT_DIR:?}/../../..)"

RECONFIG_BACKEND="${OPUS_ROOT}/simulation/reconfig_backend"
PYTHON="${PYTHON:-python3}"

PLOT_SCRIPT="${RECONFIG_BACKEND}/examples/helpers/plot_sweep_combined.py"

CENTER_RUN_DIR="${RECONFIG_BACKEND}/examples/gb200_stg_dp4_pp4_tp32_batch_256_mb-1_96stack_seq4096_100BW"

if [ ! -d "${CENTER_RUN_DIR}" ]; then
    echo "Error: Expected output directory not found at ${CENTER_RUN_DIR}"
    exit 1
fi


"${PYTHON}" "${PLOT_SCRIPT}" "${CENTER_RUN_DIR}" --output "${SCRIPT_DIR}/fig13.pdf"

if [ ! -f "${SCRIPT_DIR}/fig13.pdf" ]; then
    echo "Error: Expected output file not found at ${SCRIPT_DIR}/fig13.pdf"
    exit 1
fi