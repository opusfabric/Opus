#!/bin/bash
set -eo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
python3 "${SCRIPT_DIR}/../large_scale_ep_sweeps_scaleout/sweep_ep.py" \
    --config "${SCRIPT_DIR}/config.json" \
    "$@"
