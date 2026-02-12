#!/usr/bin/env bash

export MASTER_ADDR=128.253.51.206
export MASTER_PORT=29500

# SERVER_IPS=128.253.51.206,128.253.51.207,128.253.51.208,128.253.51.209
export SERVER_IPS=singh-compute-06.cs.cornell.edu,singh-compute-07.cs.cornell.edu,en-cs-singh-compute-08.coecis.cornell.edu,en-cs-singh-compute-09.coecis.cornell.edu
export CONTROLLER_IP=128.253.51.206
export IS_EMULATION=0
export MODE="baseline"

export NODE_RANK=$(echo $SERVER_IPS | tr ',' '\n' | grep -n $(hostname) | cut -d':' -f1)
export NODE_RANK=$((NODE_RANK - 1))

echo "Node IPs: $SERVER_IPS"
echo "Node Rank: $NODE_RANK"

export RECONFIG_LATENCY=0


export NCCL_DEBUG=DEBUG
export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_DEBUG_SUBSYS=COLL

export NCCL_IB_HCA=rocep
export NCCL_IB_MERGE_NICS=0
export NCCL_IB_GID_INDEX=0
export NCCL_IB_TIMEOUT=25
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_IB_DISABLE=0
export NCCL_LAUNCH_ORDER_IMPLICIT=1
export NCCL_COMM_BLOCKING=1
export CUDA_LAUNCH_BLOCKING=1

export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=8

# export NCCL_MIN_NCHANNELS=4
# export NCCL_TOPO_DUMP_FILE=nccl_topo_1link_4channel.xml

# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eno8303
# export NCCL_PROTO=SIMPLE
# export NCCL_ALGO=ring


export LOG_RANK=0,1,2,3,4,5,6,7

export JOB="dp-2-tp-2-pp-2-eval"
export CONFIG_FILE=${CONFIG_FILE:-"/Opus/torchtitan/opus-test/$JOB/llama3_8b.toml"}
export TRAIN_FILE="torchtitan.train"


export COMM_PATTERN_PATH=${COMM_PATTERN_PATH:-"/Opus/evaluation/llama-3-3d-16-latency/comm_pattern/comm_pattern"}
export OUTPUT_DIR="/Opus/torchtitan/opus-test/$JOB/output"
if [ -d "$OUTPUT_DIR" ]; then
    mv "$OUTPUT_DIR" "${OUTPUT_DIR}_tmp"
    echo "[INFO] Existing output directory moved to ${OUTPUT_DIR}_tmp"
fi
mkdir -p $OUTPUT_DIR

export NUM_NODES=4
export NUM_RANKS_PER_NODE=2

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

export NODE_HOSTNAME=$(hostname)
if [ "$NODE_HOSTNAME" = "$(echo $SERVER_IPS | cut -d',' -f1)" ]; then
    echo '[INFO] Launching controller on $NODE_HOSTNAME'
    cd /Opus/src/opus-controller
    ./controller -d $RECONFIG_LATENCY -r 2 -o $OUTPUT_DIR \
        > $OUTPUT_DIR/controller.ans 2>&1 &
fi

sleep 2
# nsys profile --output nsys_rank${RANK} --trace=cuda,nvtx \

cd /Opus/torchtitan
torchrun --nnodes=4 --nproc_per_node=2 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE}