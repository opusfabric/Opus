#!/usr/bin/env bash

MASTER_ADDR=localhost
MASTER_PORT=29500

export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_SUBSYS=COLL

export NCCL_IB_HCA=rocep
export NCCL_IB_MERGE_NICS=0
export NCCL_IB_GID_INDEX=0
export NCCL_IB_TIMEOUT=25
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_IB_DISABLE=0
export NCCL_LAUNCH_ORDER_IMPLICIT=1
export NCCL_COMM_BLOCKING=1

export HF_HOME=/tmp/hf
export HF_DATASETS_CACHE=/tmp/hf/datasets
export TRANSFORMERS_CACHE=/tmp/hf/transformers
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

mkdir -p /tmp/hf/datasets


# export NCCL_MIN_NCHANNELS=4
# export NCCL_TOPO_DUMP_FILE=nccl_topo_1link_4channel.xml

# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eno8303
# export NCCL_PROTO=SIMPLE
# export NCCL_ALGO=ring


export LOG_RANK=0,1,2,3

CONFIG_FILE=${CONFIG_FILE:-"./opus-test/tp-4/llama3_debug.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

# IMPORTANT, set all server names
export SERVER_IPS=$(hostname)

# nsys profile --output nsys_rank${RANK} --trace=cuda,nvtx \
torchrun --nnodes=1 --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE}