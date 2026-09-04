#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")

OPUS_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
: "${STG_DIR:=${OPUS_ROOT}/simulation/symbolic_tensor_graph}"
EXAMPLE_DIR="${SCRIPT_DIR}/examples"
TEMPLATE_DIR="${EXAMPLE_DIR}/stg-template"

: "${TP:=32}"
: "${PP:=4}"
: "${DP:=4}"
: "${EP:=1}"
: "${PYTHON:=python3}"
: "${MIXED_PRECISION:=0}"
: "${SCALE_UP_BW:=1800}"
: "${SCALE_OUT_SWEEPS:=12.5 25 50 100 200}"
: "${WS:=0}"
: "${NS:=96}"
: "${SEQ_LEN:=4096}"
: "${BATCH:=256}"
: "${MB:=-1}"

read -ra SCALE_OUT_VALUES <<< "${SCALE_OUT_SWEEPS}"
echo "SWEEPING over SCALE_OUT_BW values: ${SCALE_OUT_VALUES[*]}"

NPU_COUNT=$((DP * TP * PP * EP))
NPU_PEAK_MEM_BW=8000
NPU_PEAK_PERF=1250

for bw in "${SCALE_OUT_VALUES[@]}"; do
    echo "Generating experiment with SCALE_OUT_BW=${bw} ..."
    SCALE_OUT_BW="${bw}"

    if [[ "${MIXED_PRECISION}" -eq 1 ]]; then
        OUT_DIR="${EXAMPLE_DIR}/gb200_llama_dp${DP}_pp${PP}_tp${TP}_batch_${BATCH}_mb${MB}_${NS}stack_seq${SEQ_LEN}_${SCALE_OUT_BW}BW"
    else
        OUT_DIR="${EXAMPLE_DIR}/gb200_stg_dp${DP}_pp${PP}_tp${TP}_batch_${BATCH}_mb${MB}_${NS}stack_seq${SEQ_LEN}_${SCALE_OUT_BW}BW"
    fi

    echo "Output directory: ${OUT_DIR}"
    mkdir -p "${OUT_DIR}"

    cd "${STG_DIR}"
    echo "Generating workload files..."
    "${PYTHON}" "${STG_DIR}/main.py" \
        --output_dir "${OUT_DIR}" \
        --output_name workload.%d.et \
        --dp "${DP}" --pp "${PP}" --tp "${TP}" --ep "${EP}" \
        --micro_batch "${MB}" \
        --batch "${BATCH}" \
        --weight_sharded "${WS}" \
        --seq "${SEQ_LEN}" \
        --num_stacks "${NS}" \
        --mixed_precision "${MIXED_PRECISION}"

    echo "Copying simulator run files..."
    cp "${TEMPLATE_DIR}/network.yml" \
       "${TEMPLATE_DIR}/remote_memory.json" \
       "${TEMPLATE_DIR}/run_network_reconfig.sh" \
       "${TEMPLATE_DIR}/run_network_analytical.sh" \
       "${TEMPLATE_DIR}/system.json" \
       "${OUT_DIR}/"

    cd "${OUT_DIR}"

    echo "Generating analytical topology..."
    "${PYTHON}" "${EXAMPLE_DIR}/helpers/topo_gen_pp_split.py" \
        "${DP}" "${TP}" "${PP}" "${SCALE_OUT_BW}" "${SCALE_UP_BW}" "${SCALE_OUT_BW}" true analy

    echo "Generating reconfigurable topology..."
    "${PYTHON}" "${EXAMPLE_DIR}/helpers/topo_gen_pp_split.py" \
        "${DP}" "${TP}" "${PP}" "${SCALE_OUT_BW}" "${SCALE_UP_BW}" "${SCALE_OUT_BW}" true fg-pp

    echo "Customizing run scripts and system configuration..."
    sed -i "s/REPLACE_DP/${DP}/g" run_network_analytical.sh
    sed -i "s/REPLACE_TP/${TP}/g" run_network_analytical.sh
    sed -i "s/REPLACE_PP/${PP}/g" run_network_analytical.sh
    sed -i "s/REPLACE_EP/${EP}/g" run_network_analytical.sh
    sed -i "s/REPLACE_SCALE_OUT_BW/${SCALE_OUT_BW}/g" run_network_analytical.sh
    sed -i "s/REPLACE_SCALE_UP_BW/${SCALE_UP_BW}/g" run_network_analytical.sh
    sed -i "s/REPLACE_NPU_COUNT/${NPU_COUNT}/g" network.yml
    sed -i "s/\"local-mem-bw\": 4800/\"local-mem-bw\": ${NPU_PEAK_MEM_BW}/" system.json
    sed -i "s/\"peak-perf\": 989/\"peak-perf\": ${NPU_PEAK_PERF}/" system.json
done
