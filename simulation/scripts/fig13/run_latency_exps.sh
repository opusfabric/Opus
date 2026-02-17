SCRIPT_DIR=$(dirname "$(realpath "$0")")
OPUS_ROOT="$(realpath ${SCRIPT_DIR:?}/../../..)"

ANALYTICAL_BACKEND="${OPUS_ROOT}/simulation/analytical_backend"
RECONFIG_BACKEND="${OPUS_ROOT}/simulation/reconfig_backend"

ANALYTICAL_EXE="${ANALYTICAL_BACKEND}/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"
RECONFIG_EXE="${RECONFIG_BACKEND}/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"

STG_DIR="${OPUS_ROOT}/simulation/symbolic_tensor_graph"

TP=32
PP=4
DP=4
MIXED_PRECISION=0
SCALE_OUT_SWEEPS="100"

cd ${RECONFIG_BACKEND}

OUT_DIR=${RECONFIG_BACKEND}/examples/gb200_stg_dp${DP}_pp${PP}_tp${TP}_batch_256_mb-1_96stack_seq4096_${SCALE_OUT_SWEEPS}BW
if [ -d "${OUT_DIR}" ] && [ -f "${OUT_DIR}/workload.0.et" ]; then
    echo "Output directory ${OUT_DIR} already exists with expected workload. Skipping generation."
    echo "WARNING: for a fresh run, please delete the existing output directory at ${OUT_DIR} and re-run this script."
else
    echo "Output directory ${OUT_DIR} does not exist or is missing expected workload. Generating experiment..."
    SCALE_OUT_SWEEPS=${SCALE_OUT_SWEEPS} TP=${TP} PP=${PP} DP=${DP} MIXED_PRECISION=${MIXED_PRECISION} ./run_stg_exp_fg_pp_GB200.sh > /dev/null
fi

if [ ! -d "${OUT_DIR}" ]; then
    echo "Error: Expected output directory not found at ${OUT_DIR}"
    exit 1
fi

cd ${OUT_DIR}
echo "Using Run Helper to run all experiments in ${OUT_DIR}"

RUN_HELPER=${RECONFIG_BACKEND}/examples/helpers/run_helper.py

python ${RUN_HELPER} . --reconfig-times 0,0.01,0.05,0.1,0.25,0.5,0.75,1

if [ ! -f "results_for_sheet_import.txt" ]; then
    echo "Error: Expected output file not found in ${OUT_DIR}"
    exit 1
fi

echo "Generated results file in ${OUT_DIR}/results_for_sheet_import.txt"