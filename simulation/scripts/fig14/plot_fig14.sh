SCRIPT_DIR=$(dirname "$(realpath "$0")")
OPUS_ROOT="$(realpath ${SCRIPT_DIR:?}/../../..)"

RECONFIG_BACKEND="${OPUS_ROOT}/simulation/reconfig_backend"

PLOT_SCRIPT="${RECONFIG_BACKEND}/plot_combined_dp_cost_power.py"

DP_1="${RECONFIG_BACKEND}/examples/stg_dp4_pp4_tp8_batch_256_mb-1_96stack_seq4096_50BW"
DP_2="${RECONFIG_BACKEND}/examples/gb200_stg_dp4_pp4_tp32_batch_256_mb-1_96stack_seq4096_100BW"

if [ ! -d "${DP_1}" ] || [ ! -d "${DP_2}" ]; then
    echo "Error: Expected output directory not found at ${DP_1} or ${DP_2}"
    exit 1
fi


python ${PLOT_SCRIPT} --latency '10 ms' --bw1 "50" --bw2 "100" --output ${SCRIPT_DIR}/fig14.pdf ${DP_1} ${DP_2}

if [ ! -f "${SCRIPT_DIR}/fig14.pdf" ]; then
    echo "Error: Expected output file not found at ${SCRIPT_DIR}/fig14.pdf"
    exit 1
fi