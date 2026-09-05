SCRIPT_DIR=$(dirname "$(realpath "$0")")
OPUS_ROOT="$(realpath "${SCRIPT_DIR}/../..")"

: "${STG_DIR:=${OPUS_ROOT}/simulation/symbolic_tensor_graph}"
EXAMPLE_DIR="${SCRIPT_DIR}/examples"
TEMPLATE_DIR="${EXAMPLE_DIR}/stg-template"

TP=32
PP=8
DP=16

MIXED_PRECISION=1

SCALE_UP_BW=1800

SCALE_OUT_SWEEPS=(200)

SCALE_OUT_BW=200

WS=0

SEQ_LEN=4096

MB=-1

NPU_COUNT=$((DP * TP * PP))

NPU_PEAK_MEM_BW=8000
NPU_PEAK_PERF=1250

for bw in "${SCALE_OUT_SWEEPS[@]}"; do
    echo "Generating experiment with SCALE_OUT_BW=${bw} ..."
    SCALE_OUT_BW=${bw}

    OUT_DIR="$EXAMPLE_DIR/gb200_llama_405B_dp${DP}_pp${PP}_tp${TP}_${SCALE_OUT_BW}BW"


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
                --batch 256 \
                --weight_sharded ${WS} \
                --mixed_precision ${MIXED_PRECISION} \
                --dmodel 16384 \
                --dff 53248 \
                --num_stacks 128 \
                --head 128 \
                --kvhead 8 \
                --dvocal 32000 \
                --seq 8192


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