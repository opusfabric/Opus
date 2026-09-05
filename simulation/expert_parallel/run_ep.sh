#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
OPUS_ROOT=$(realpath "${SCRIPT_DIR}/../..")
BACKEND="${OPUS_ROOT}/simulation/reconfig_backend"
EXAMPLES="${BACKEND}/examples"
PYTHON=${PYTHON:-python3}

if [[ -z "${ASTRA_SIM_BIN_DIR:-}" ]]; then
    ASTRA_SIM_BIN_DIR="${BACKEND}/build/astra_analytical/build/bin"
fi
export ASTRA_SIM_BIN_DIR

require_binary() {
    local name=$1
    if [[ ! -x "${ASTRA_SIM_BIN_DIR}/${name}" ]]; then
        echo "Missing ${ASTRA_SIM_BIN_DIR}/${name}" >&2
        echo "Build first with: ./simulation/scripts/build_backends.sh" >&2
        exit 2
    fi
}

run_source() {
    local directory=$1
    "${PYTHON}" "${EXAMPLES}/large_scale_ep_sweeps/sweep_ep.py" \
        --config "${EXAMPLES}/${directory}/config.json" run --rerun
}

run_paper_case() {
    local size=$1
    local original scaleout final
    if [[ "${size}" == 256 ]]; then
        original=large_scale_ep_8gpu_256_all2allv_topk_context
        scaleout=large_scale_ep_scaleout_8gpu_256_all2allv_topk_context
        final=large_scale_ep_8gpu_256_all2allv_topk_context_reconfig_full_bw_eps
    elif [[ "${size}" == 512 ]]; then
        original=large_scale_ep_32gpu_512_all2allv_topk_context
        scaleout=large_scale_ep_scaleout_32gpu_512_all2allv_topk_context
        final=large_scale_ep_32gpu_512_all2allv_topk_context_reconfig_full_bw_eps
    else
        echo "Unsupported GPU count: ${size}" >&2
        exit 2
    fi

    run_source "${original}"
    run_source "${scaleout}"
    "${PYTHON}" "${EXAMPLES}/${final}/run_full_bw_eps.py" all \
        --baseline constrained --rerun --stat max
}

plot_archived() {
    "${PYTHON}" "${EXAMPLES}/large_scale_ep_8gpu_256_all2allv_topk_context_reconfig_full_bw_eps/run_full_bw_eps.py" \
        plot --baseline constrained --stat max
    "${PYTHON}" "${EXAMPLES}/large_scale_ep_32gpu_512_all2allv_topk_context_reconfig_full_bw_eps/run_full_bw_eps.py" \
        plot --baseline constrained --stat max
}

action=${1:-smoke}
case "${action}" in
    smoke)
        require_binary AstraSim_Analytical_Reconfigurable
        "${PYTHON}" "${EXAMPLES}/large_scale_ep_sweeps/sweep_ep.py" \
            --config "${SCRIPT_DIR}/configs/mixed/config.json" run --rerun
        "${PYTHON}" "${SCRIPT_DIR}/check_ep_trend.py" \
            "${SCRIPT_DIR}/configs/mixed/results/summary.csv"
        "${PYTHON}" "${EXAMPLES}/large_scale_ep_sweeps_scaleout/sweep_ep.py" \
            --config "${SCRIPT_DIR}/configs/scaleout/config.json" run --rerun
        "${PYTHON}" "${SCRIPT_DIR}/check_ep_trend.py" \
            "${SCRIPT_DIR}/configs/scaleout/results/summary.csv"
        ;;
    256|512)
        require_binary AstraSim_Analytical_Reconfigurable
        run_paper_case "${action}"
        ;;
    all)
        require_binary AstraSim_Analytical_Reconfigurable
        run_paper_case 256
        run_paper_case 512
        ;;
    plot)
        plot_archived
        ;;
    *)
        echo "Usage: $0 [smoke|256|512|all|plot]" >&2
        exit 2
        ;;
esac
