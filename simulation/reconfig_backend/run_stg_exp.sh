SCRIPT_DIR=$(dirname "$(realpath "$0")")

STG_DIR="/home/soxehli/work/symbolic_tensor_graph"
EXAMPLE_DIR="${SCRIPT_DIR}/examples"
TEMPLATE_DIR="${EXAMPLE_DIR}/stg-template"

TP=8
PP=2
DP=4
MB=-1
WS=0
NS=100

SEQ_LEN=4096

BATCH=$((DP / 2 * 64))

NPU_COUNT=$((DP * TP * PP))

OUT_DIR="$EXAMPLE_DIR/stg_dp${DP}_pp${PP}_tp${TP}_batch_${BATCH}_mb${MB}_${NS}stack_seq${SEQ_LEN}"

cd ${STG_DIR} || exit 1
mkdir -p ${OUT_DIR}

python ${STG_DIR}/main.py --output_dir ${OUT_DIR} \
               --output_name workload.%d.et \
               --dp ${DP} --pp ${PP} --tp ${TP} \
               --micro_batch ${MB} \
               --batch ${BATCH} \
               --weight_sharded ${WS} \
               --seq ${SEQ_LEN} \
               --num_stacks ${NS}

cp -r ${TEMPLATE_DIR}/* ${OUT_DIR}/

cd ${OUT_DIR} 

#  Usage: python topo_gen.py <dp> <tp> <pp> [<dp_bw>] [<tp_bw>] [<pp_bw>
python ${EXAMPLE_DIR}/helpers/topo_gen.py ${DP} ${TP} ${PP} 100 450 100 true
python ${EXAMPLE_DIR}/helpers/topo_gen.py ${DP} ${TP} ${PP} 100 450 100 true --collapse-topo

sed -i "s/REPLACE_NPU_COUNT/${NPU_COUNT}/g" network.yml


