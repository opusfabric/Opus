#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OPUS_ROOT="$(realpath "${SCRIPT_DIR}/../../..")"
RECONFIG_BACKEND="${OPUS_ROOT}/simulation/reconfig_backend"
ANALYTICAL_BACKEND="${OPUS_ROOT}/simulation/analytical_backend"
RUN_HELPER="${RECONFIG_BACKEND}/examples/helpers/run_helper.py"
RECONFIG_EXE="${RECONFIG_BACKEND}/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"
ANALYTICAL_EXE="${ANALYTICAL_BACKEND}/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"

TP=32
PP=4
DP=4
MIXED_PRECISION=0
SCALE_OUT_BW=100
PYTHON="${PYTHON:-python3}"

if [[ ! -x "${RECONFIG_EXE}" || ! -x "${ANALYTICAL_EXE}" ]]; then
    echo "Building the simulator backends..."
    "${OPUS_ROOT}/simulation/scripts/build_backends.sh"
fi

OUT_DIR="${RECONFIG_BACKEND}/examples/gb200_stg_dp${DP}_pp${PP}_tp${TP}_batch_256_mb-1_96stack_seq4096_${SCALE_OUT_BW}BW"
if [[ ! -f "${OUT_DIR}/workload.0.et" ||
      ! -f "${OUT_DIR}/run_network_reconfig.sh" ||
      ! -f "${OUT_DIR}/run_network_analytical.sh" ||
      ! -f "${OUT_DIR}/schedules-collapsed.txt" ]]; then
    echo "Generating Figure 13 workload and analytical/reconfigurable schedules..."
    SCALE_OUT_SWEEPS="${SCALE_OUT_BW}" TP="${TP}" PP="${PP}" DP="${DP}" \
        MIXED_PRECISION="${MIXED_PRECISION}" PYTHON="${PYTHON}" \
        "${RECONFIG_BACKEND}/run_stg_exp_fg_pp_GB200.sh"
fi

if [[ ! -d "${OUT_DIR}" ]]; then
    echo "Error: expected output directory not found at ${OUT_DIR}" >&2
    exit 1
fi

cd "${OUT_DIR}"
export OPUS_SKIP_LEGACY_BUILD=1
"${PYTHON}" "${RUN_HELPER}" . \
    --reconfig-times 0,0.01,0.05,0.1,0.25,0.5,0.75,1 \
    --include-analytical \
    --output latency_results_for_sheet_import.txt

if [[ ! -s latency_results_for_sheet_import.txt ]]; then
    echo "Error: expected latency results file not found in ${OUT_DIR}" >&2
    exit 1
fi

echo "Generated analytical EPS/baseline and reconfigurable latency results in ${OUT_DIR}/latency_results_for_sheet_import.txt"
