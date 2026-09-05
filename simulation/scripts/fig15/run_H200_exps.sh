#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
OPUS_ROOT="$(realpath ${SCRIPT_DIR:?}/../../..)"

RECONFIG_BACKEND="${OPUS_ROOT}/simulation/reconfig_backend"



TP=8
PP=4
DP_SWEEP=(2 4 8 16)

MIXED_PRECISION=0
SCALE_OUT_SWEEPS="50"
PYTHON="${PYTHON:-python3}"

cd ${RECONFIG_BACKEND}

for DP in ${DP_SWEEP[@]}; do
    OUT_DIR=${RECONFIG_BACKEND}/examples/stg_dp${DP}_pp${PP}_tp${TP}_batch_256_mb-1_96stack_seq4096_${SCALE_OUT_SWEEPS}BW
    LAST_RANK=$((DP * TP * PP - 1))
    if [[ -s "${OUT_DIR}/workload.0.et" &&
          -s "${OUT_DIR}/workload.${LAST_RANK}.et" &&
          -s "${OUT_DIR}/system.json" &&
          -s "${OUT_DIR}/remote_memory.json" &&
          -s "${OUT_DIR}/workload.json" &&
          -s "${OUT_DIR}/schedules.txt" &&
          -s "${OUT_DIR}/run_network_reconfig.sh" ]]; then
        echo "Output directory ${OUT_DIR} already exists with expected workload. Skipping generation."
        echo "WARNING: for a fresh run, please delete the existing output directory at ${OUT_DIR} and re-run this script."
    else
        echo "Output directory ${OUT_DIR} does not exist or is missing expected workload. Generating experiment..."
        SCALE_OUT_SWEEPS=${SCALE_OUT_SWEEPS} TP=${TP} PP=${PP} DP=${DP} MIXED_PRECISION=${MIXED_PRECISION} ./run_stg_exp_fg_pp.sh
    fi

    if [ ! -d "${OUT_DIR}" ]; then
        echo "Error: Expected output directory not found at ${OUT_DIR}"
        exit 1
    fi

    cd ${OUT_DIR}
    echo "Using Run Helper to run all experiments in ${OUT_DIR}"

    RUN_HELPER=${RECONFIG_BACKEND}/examples/helpers/run_helper.py

    "${PYTHON}" "${RUN_HELPER}" . --reconfig-times 0,0.01

    if [[ ! -s "results_for_sheet_import.txt" ]]; then
        echo "Error: Expected output file not found in ${OUT_DIR}"
        exit 1
    fi

    echo "Generated results file in ${OUT_DIR}/results_for_sheet_import.txt"
    cd ${RECONFIG_BACKEND}
done

echo "All experiments completed for DP sweep: ${DP_SWEEP[@]}"



