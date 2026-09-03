SCRIPT_DIR=$(dirname "$(realpath "$0")")

OPUS_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
: "${STG_DIR:=${OPUS_ROOT}/simulation/symbolic_tensor_graph}"
EXAMPLE_DIR="${SCRIPT_DIR}/examples"
TEMPLATE_DIR="${EXAMPLE_DIR}/stg-template"

: "${TP:=32}"
: "${PP:=8}"
: "${DP:=8}"

: "${MIXED_PRECISION:=0}"

SCALE_UP_BW=1800

if [ -z "${SCALE_OUT_SWEEPS}" ]; then
    SCALE_OUT_SWEEPS=(12.5 25 50 100 200)
else
    # If passed as space-separated string, convert to array
    read -ra SCALE_OUT_SWEEPS <<< "${SCALE_OUT_SWEEPS}"
fi
echo "SWEEPING over SCALE_OUT_BW values: ${SCALE_OUT_SWEEPS[*]}"

SCALE_OUT_BW=200

WS=0
NS=96

SEQ_LEN=4096

# Weak Scaling Batch Size
# BATCH=$((DP / 2 * 64))

# Strong Scaling Batch Size
BATCH=256

MB=-1

NPU_COUNT=$((DP * TP * PP))

NPU_PEAK_MEM_BW=8000
NPU_PEAK_PERF=1250

for bw in "${SCALE_OUT_SWEEPS[@]}"; do
    echo "Generating experiment with SCALE_OUT_BW=${bw} ..."
    SCALE_OUT_BW=${bw}

    if [ ${MIXED_PRECISION} -eq 1 ]; then
        OUT_DIR="$EXAMPLE_DIR/gb200_llama_dp${DP}_pp${PP}_tp${TP}_batch_${BATCH}_mb${MB}_${NS}stack_seq${SEQ_LEN}_${SCALE_OUT_BW}BW"
    else
        OUT_DIR="$EXAMPLE_DIR/gb200_stg_dp${DP}_pp${PP}_tp${TP}_batch_${BATCH}_mb${MB}_${NS}stack_seq${SEQ_LEN}_${SCALE_OUT_BW}BW"
    fi


    echo "Output directory: ${OUT_DIR}"

    if [ -d "${OUT_DIR}" ]; then
        echo "Warning: Output directory already exists."
    fi

    cd ${STG_DIR} || exit 1
    mkdir -p ${OUT_DIR}

    echo "Generating workload files..."
    python ${STG_DIR}/main.py --output_dir ${OUT_DIR} \
                --output_name workload.%d.et \
                --dp ${DP} --pp ${PP} --tp ${TP} \
                --micro_batch ${MB} \
                --batch ${BATCH} \
                --weight_sharded ${WS} \
                --seq ${SEQ_LEN} \
                --num_stacks ${NS} \
                --mixed_precision ${MIXED_PRECISION}


    echo "Copying reconfigurable-backend run files..."
    cp "${TEMPLATE_DIR}/network.yml" "${TEMPLATE_DIR}/remote_memory.json" "${TEMPLATE_DIR}/run_network_reconfig.sh" "${TEMPLATE_DIR}/system.json" "${OUT_DIR}/"


    cd ${OUT_DIR} 

    echo "Generating reconfigurable topology..."
    python ${EXAMPLE_DIR}/helpers/topo_gen_pp_split.py ${DP} ${TP} ${PP} ${SCALE_OUT_BW} ${SCALE_UP_BW} ${SCALE_OUT_BW} true fg-pp

    echo "Customizing run scripts and system configuration..."
    sed -i "s/REPLACE_NPU_COUNT/${NPU_COUNT}/g" network.yml

    # Replace "REPLACE_LOCAL_MEM_BW" and "REPLACE_PEAK_PERF" in system.json
    sed -i "s/\"local-mem-bw\": 4800/\"local-mem-bw\": ${NPU_PEAK_MEM_BW}/" system.json
    sed -i "s/\"peak-perf\": 989/\"peak-perf\": ${NPU_PEAK_PERF}/" system.json

done