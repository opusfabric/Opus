SCRIPT_DIR=$(dirname "$(realpath "$0")")

: "${STG_DIR:=/home/soxehli/work/symbolic_tensor_graph}"
EXAMPLE_DIR="${SCRIPT_DIR}/examples"
TEMPLATE_DIR="${EXAMPLE_DIR}/stg-template"

: "${TP:=2}"
: "${PP:=2}"
: "${DP:=2}"

: "${MIXED_PRECISION:=0}"

SCALE_UP_BW=450

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

NPU_PEAK_MEM_BW=4800
NPU_PEAK_PERF=989

for bw in "${SCALE_OUT_SWEEPS[@]}"; do
    echo "Generating experiment with SCALE_OUT_BW=${bw} ..."
    SCALE_OUT_BW=${bw}

    if [ ${MIXED_PRECISION} -eq 1 ]; then
        OUT_DIR="$EXAMPLE_DIR/llama_dp${DP}_pp${PP}_tp${TP}_batch_${BATCH}_mb${MB}_${NS}stack_seq${SEQ_LEN}_${SCALE_OUT_BW}BW"
    else
        OUT_DIR="$EXAMPLE_DIR/stg_dp${DP}_pp${PP}_tp${TP}_batch_${BATCH}_mb${MB}_${NS}stack_seq${SEQ_LEN}_${SCALE_OUT_BW}BW"
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


    echo "Copying run template files..."
    cp -r ${TEMPLATE_DIR}/* ${OUT_DIR}/


    cd ${OUT_DIR} 

    echo "Generating analytical topology..."
    #  Usage: Usage: python topo_gen.py <dp> <tp> <pp> <dp_bw> <tp_bw> <pp_bw> <swap_dp_tp> <mono-pp|analy|fg-pp>
    python ${EXAMPLE_DIR}/helpers/topo_gen_pp_split.py ${DP} ${TP} ${PP} ${SCALE_OUT_BW} ${SCALE_UP_BW} ${SCALE_OUT_BW} true analy
    # python ${EXAMPLE_DIR}/helpers/topo_gen_pp_split.py ${DP} ${TP} ${PP} 100 450 100 false mono-pp

    echo "Generating reconfig topology..."
    python ${EXAMPLE_DIR}/helpers/topo_gen_pp_split.py ${DP} ${TP} ${PP} ${SCALE_OUT_BW} ${SCALE_UP_BW} ${SCALE_OUT_BW} true fg-pp

    echo "Customizing run scripts and system configuration..."
    sed -i "s/REPLACE_NPU_COUNT/${NPU_COUNT}/g" network.yml

    # Replace "REPLACE_DP", "REPLACE_TP", and "REPLACE_PP" in run_network_analytical.sh
    sed -i "s/REPLACE_DP/${DP}/g" run_network_analytical.sh
    sed -i "s/REPLACE_TP/${TP}/g" run_network_analytical.sh
    sed -i "s/REPLACE_PP/${PP}/g" run_network_analytical.sh

    sed -i "s/REPLACE_SCALE_OUT_BW/${SCALE_OUT_BW}/g" run_network_analytical.sh
    sed -i "s/REPLACE_SCALE_UP_BW/${SCALE_UP_BW}/g" run_network_analytical.sh

    # Replace "REPLACE_LOCAL_MEM_BW" and "REPLACE_PEAK_PERF" in system.json
    sed -i "s/REPLACE_LOCAL_MEM_BW/${NPU_PEAK_MEM_BW}/g" system.json
    sed -i "s/REPLACE_PEAK_PERF/${NPU_PEAK_PERF}/g" system.json

done