#!/usr/bin/env bash

MASTER_ADDR=128.253.51.206
MASTER_PORT=29500
WORLD_SIZE=4
RANK=$1
if [ -z "$RANK" ]; then
    echo "Usage: $0 <node_rank>"
    exit 1
fi

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_HCA=rocep
export NCCL_IB_GID_INDEX=0
export NCCL_IB_TIMEOUT=22
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_IB_DISABLE=0
export NCCL_MIN_NCHANNELS=4
export NCCL_CROSS_NIC=2
export NCCL_TOPO_DUMP_FILE=nccl_topo_1link_4channel.xml

export LOG_RANK=0,1,2,3

CONFIG_FILE=${CONFIG_FILE:-"./opus-test/pp-2-tp-2/llama3_debug.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

PYTORCH_ALLOC_CONF="expandable_segments:True" \
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
torchrun --nnodes=2 --nproc_per_node=2 \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE}