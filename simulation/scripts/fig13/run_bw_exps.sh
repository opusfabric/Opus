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
SCALE_OUT_SWEEPS="12.5 25 50 200"

cd ${RECONFIG_BACKEND}
echo "Generating experiment with TP=${TP}, PP=${PP}, DP=${DP}, MIXED_PRECISION=${MIXED_PRECISION}, SCALE_OUT_SWEEPS=${SCALE_OUT_SWEEPS} ..."

GOOD=1
for bw in ${SCALE_OUT_SWEEPS}; do
    OUT_DIR=${RECONFIG_BACKEND}/examples/gb200_stg_dp${DP}_pp${PP}_tp${TP}_batch_256_mb-1_96stack_seq4096_${bw}BW
    if [ -d "${OUT_DIR}" ] && [ -f "${OUT_DIR}/workload.0.et" ]; then
        echo "Output directory ${OUT_DIR} already exists with expected workload. "
    else
       GOOD=0
       break
    fi
done

if [ ${GOOD} -eq 1 ]; then
    echo "All expected output directories already exist with expected workloads. Skipping generation."
    echo "WARNING: for a fresh run, please delete the existing output directories at ${RECONFIG_BACKEND}/examples/gb200_stg_dp${DP}_pp${PP}_tp${TP}_batch_256_mb-1_96stack_seq4096_*BW and re-run this script."
else
    echo "At least one expected output directory is missing or does not contain expected workload. Generating experiments..."
    SCALE_OUT_SWEEPS=${SCALE_OUT_SWEEPS} TP=${TP} PP=${PP} DP=${DP} MIXED_PRECISION=${MIXED_PRECISION} ./run_stg_exp_fg_pp_GB200.sh > /dev/null
fi
# SCALE_OUT_SWEEPS=${SCALE_OUT_SWEEPS} TP=${TP} PP=${PP} DP=${DP} MIXED_PRECISION=${MIXED_PRECISION} ./run_stg_exp_fg_pp.sh > /dev/null

for bw in ${SCALE_OUT_SWEEPS}; do
    echo "Running experiments for SCALE_OUT_BW=${bw} ..."

    OUT_DIR=${RECONFIG_BACKEND}/examples/gb200_stg_dp${DP}_pp${PP}_tp${TP}_batch_256_mb-1_96stack_seq4096_${bw}BW

    if [ ! -d "${OUT_DIR}" ]; then
        echo "Error: Expected output directory not found at ${OUT_DIR}"
        exit 1
    fi

    echo "Generated example workload in ${OUT_DIR}"
    cd ${OUT_DIR}

    echo "Using Run Helper to run all experiments in ${OUT_DIR}"

    RUN_HELPER=${RECONFIG_BACKEND}/examples/helpers/run_helper.py
    python ${RUN_HELPER} . --reconfig-times 0.01


    if [ ! -f "results_for_sheet_import.txt" ]; then
        echo "Error: Expected output file not found in ${OUT_DIR}"
        exit 1
    fi

    echo "Generated results file in ${OUT_DIR}"

done

echo "Completed all experiments for SCALE_OUT_BW values: ${SCALE_OUT_SWEEPS}"