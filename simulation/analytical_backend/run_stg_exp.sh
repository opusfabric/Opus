SCRIPT_DIR=$(dirname "$(realpath "$0")")

STG_DIR="/home/soxehli/work/symbolic_tensor_graph"
EXAMPLE_DIR="${SCRIPT_DIR}/examples"
TEMPLATE_DIR="${EXAMPLE_DIR}/stg-template"

TP=4
PP=4
DP=4
MB=16
WS=0
NS=10

NPU_COUNT=$((DP * TP * PP))

OUT_DIR="$EXAMPLE_DIR/stg_dp${DP}_pp${PP}_tp${TP}_mb${MB}_${NS}stack"

cd ${STG_DIR} || exit 1
mkdir -p ${OUT_DIR}

python ${STG_DIR}/main.py --output_dir ${OUT_DIR} \
               --output_name workload.%d.et \
               --dp ${DP} --pp ${PP} --tp ${TP} \
               --micro_batch ${MB} \
               --weight_sharded ${WS} \
               --num_stacks ${NS}

cp -r ${TEMPLATE_DIR}/* ${OUT_DIR}/

cd ${OUT_DIR} 

#  Usage: python topo_gen.py <dp> <tp> <pp> [<dp_bw>] [<tp_bw>] [<pp_bw>
python ${EXAMPLE_DIR}/helpers/topo_gen.py ${DP} ${TP} ${PP} 100 450 100 true

sed -i "s/REPLACE_NPU_COUNT/${NPU_COUNT}/g" network.yml


