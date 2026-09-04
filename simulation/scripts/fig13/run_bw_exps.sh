#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OPUS_ROOT="$(realpath "${SCRIPT_DIR}/../../..")"
RECONFIG_BACKEND="${OPUS_ROOT}/simulation/reconfig_backend"
ANALYTICAL_BACKEND="${OPUS_ROOT}/simulation/analytical_backend"
ANALYTICAL_EXE="${ANALYTICAL_BACKEND}/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"
RECONFIG_EXE="${RECONFIG_BACKEND}/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"
RUN_HELPER="${RECONFIG_BACKEND}/examples/helpers/run_helper.py"

TP=32
PP=4
DP=4
MIXED_PRECISION=0
PYTHON="${PYTHON:-python3}"
SCALE_OUT_SWEEPS="${SCALE_OUT_SWEEPS:-12.5 25 50 200}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "Error: Python interpreter not found: ${PYTHON}" >&2
    exit 2
fi

echo "Ensuring the analytical and reconfigurable simulators are built..."
if [[ ! -x "${RECONFIG_EXE}" || ! -x "${ANALYTICAL_EXE}" ]]; then
    "${OPUS_ROOT}/simulation/scripts/build_backends.sh"
fi

if [[ ! -x "${ANALYTICAL_EXE}" ]]; then
    echo "Error: analytical simulator was not built" >&2
    exit 1
fi

NEEDS_GENERATION=0
for bw in ${SCALE_OUT_SWEEPS}; do
    OUT_DIR="${RECONFIG_BACKEND}/examples/gb200_stg_dp${DP}_pp${PP}_tp${TP}_batch_256_mb-1_96stack_seq4096_${bw}BW"
    if [[ ! -f "${OUT_DIR}/workload.0.et" ||
          ! -f "${OUT_DIR}/run_network_reconfig.sh" ||
          ! -f "${OUT_DIR}/run_network_analytical.sh" ||
          ! -f "${OUT_DIR}/schedules-collapsed.txt" ]]; then
        NEEDS_GENERATION=1
        break
    fi
done

if [[ "${NEEDS_GENERATION}" == "1" ]]; then
    echo "Generating Figure 13 workloads and analytical/reconfigurable schedules..."
    SCALE_OUT_SWEEPS="${SCALE_OUT_SWEEPS}" TP="${TP}" PP="${PP}" DP="${DP}" \
        MIXED_PRECISION="${MIXED_PRECISION}" PYTHON="${PYTHON}" \
        "${RECONFIG_BACKEND}/run_stg_exp_fg_pp_GB200.sh"
fi

export OPUS_SKIP_LEGACY_BUILD=1

for bw in ${SCALE_OUT_SWEEPS}; do
    echo "Running experiments for SCALE_OUT_BW=${bw} ..."
    OUT_DIR="${RECONFIG_BACKEND}/examples/gb200_stg_dp${DP}_pp${PP}_tp${TP}_batch_256_mb-1_96stack_seq4096_${bw}BW"

    if [[ ! -d "${OUT_DIR}" ]]; then
        echo "Error: expected output directory not found at ${OUT_DIR}" >&2
        exit 1
    fi

    cd "${OUT_DIR}"
    "${PYTHON}" "${RUN_HELPER}" . --reconfig-times 0,0.01 --include-analytical \
        --output bandwidth_results_for_sheet_import.txt

    if [[ ! -s bandwidth_results_for_sheet_import.txt ]]; then
        echo "Error: expected bandwidth results file not found in ${OUT_DIR}" >&2
        exit 1
    fi

    echo "Generated analytical EPS/baseline and reconfigurable bandwidth results in ${OUT_DIR}/bandwidth_results_for_sheet_import.txt"
done
