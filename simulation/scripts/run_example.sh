#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
OPUS_ROOT="$(realpath ${SCRIPT_DIR:?}/../..)"
RECONFIG_BACKEND="${OPUS_ROOT}/simulation/reconfig_backend"
RECONFIG_EXE="${RECONFIG_BACKEND}/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"

: "${PYTHON:=python3}"
export OPUS_STG_CACHE_DIR="${OPUS_STG_CACHE_DIR:-${TMPDIR:-/tmp}/opus-stg-cache-${BASHPID}}"
mkdir -p "${OPUS_STG_CACHE_DIR}"
export PYTHON

if ! "${PYTHON}" -c "import graphviz, numpy, pandas, sympy, tqdm, yaml, google.protobuf" >/dev/null 2>&1; then
    echo "Error: the STG generator needs Python packages (graphviz, numpy, pandas, sympy, tqdm, pyyaml, protobuf)." >&2
    echo "Activate the repository .venv first, then install with: ${PYTHON} -m pip install numpy sympy graphviz pandas tqdm pyyaml protobuf" >&2
    exit 2
fi

echo "Ensuring the reconfigurable simulator is built..."
if [[ ! -x "${RECONFIG_EXE}" ]]; then
    "${OPUS_ROOT}/simulation/scripts/build_backends.sh"
else
    echo "Using existing reconfigurable simulator binary."
fi
export OPUS_SKIP_LEGACY_BUILD=1

TP=2
PP=2
DP=2
MIXED_PRECISION=0
SCALE_OUT_SWEEPS="50"
RECONFIG_TIMES_MS=(0 1 10 50)

cd "${RECONFIG_BACKEND}"
echo "Generating experiment with TP=${TP}, PP=${PP}, DP=${DP}, MIXED_PRECISION=${MIXED_PRECISION}, SCALE_OUT_SWEEPS=${SCALE_OUT_SWEEPS} ..."
SCALE_OUT_SWEEPS=${SCALE_OUT_SWEEPS} TP=${TP} PP=${PP} DP=${DP} MIXED_PRECISION=${MIXED_PRECISION} ./run_stg_exp_fg_pp.sh

OUT_DIR=${RECONFIG_BACKEND}/examples/stg_dp${DP}_pp${PP}_tp${TP}_batch_256_mb-1_96stack_seq4096_${SCALE_OUT_SWEEPS}BW

if [ ! -d "${OUT_DIR}" ]; then
    echo "Error: Expected output directory not found at ${OUT_DIR}"
    exit 1
fi

echo "Generated example workload in ${OUT_DIR}"
cd "${OUT_DIR}"

set_reconfig_latency() {
    local latency_ms="$1"
    local latency_ns=$((latency_ms * 1000000))
    sed -i -E "s/^reconfig_time:.*/reconfig_time: [ ${latency_ns}.0 ]  # ${latency_ms} ms/" network.yml
}

runtime_cycles() {
    awk "/sys\[0\], Wall time:/ { wall=\$NF } END { if (wall == \"\") exit 1; print wall }" "$1"
}

cycles_to_seconds() {
    awk -v cycles="$1" "BEGIN { print cycles / 1000000000 }"
}

delta_vs_eps() {
    awk -v value="$1" -v reference="${EPS_CYCLES}" "BEGIN { printf \"%+.3f%%\", (value / reference - 1) * 100 }"
}

printf "\nReconfigurable-backend runtime sweep (simulated wall time):\n"
printf "| Reconfiguration | EPS (s) | Baseline (s) | Provisioning (s) | Baseline vs EPS | Provisioning vs EPS |\n"
printf "| ---: | ---: | ---: | ---: | ---: | ---: |\n"

for RECONFIG_MS in "${RECONFIG_TIMES_MS[@]}"; do
    set_reconfig_latency "${RECONFIG_MS}"
    echo "Running reconfigurable backend at ${RECONFIG_MS} ms: baseline and provisioning" >&2
    ./run_network_reconfig.sh > /dev/null

    if [ ! -s "debug_no_provision.txt" ] || [ ! -s "debug_provision.txt" ]; then
        echo "Error: Expected reconfigurable-backend output files not found in ${OUT_DIR}" >&2
        exit 1
    fi

    cp debug_no_provision.txt "debug_no_provision_${RECONFIG_MS}ms.txt"
    cp debug_provision.txt "debug_provision_${RECONFIG_MS}ms.txt"

    if [ "${RECONFIG_MS}" == "0" ]; then
        EPS_CYCLES=$(runtime_cycles "debug_no_provision_${RECONFIG_MS}ms.txt")
    fi
    BASELINE_CYCLES=$(runtime_cycles "debug_no_provision_${RECONFIG_MS}ms.txt")
    PROVISION_CYCLES=$(runtime_cycles "debug_provision_${RECONFIG_MS}ms.txt")

    printf "| %s ms | %s | %s | %s | %s | %s |\n" \
        "${RECONFIG_MS}" \
        "$(cycles_to_seconds "${EPS_CYCLES}")" \
        "$(cycles_to_seconds "${BASELINE_CYCLES}")" \
        "$(cycles_to_seconds "${PROVISION_CYCLES}")" \
        "$(delta_vs_eps "${BASELINE_CYCLES}")" \
        "$(delta_vs_eps "${PROVISION_CYCLES}")"
done

printf "\nOutput files generated in %s\n" "${OUT_DIR}"