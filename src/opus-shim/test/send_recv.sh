#!/usr/bin/env bash

MASTER_ADDR=128.253.51.206
MASTER_PORT=29502
WORLD_SIZE=2
RANK=$1
if [ -z "$RANK" ]; then
    echo "Usage: $0 <node_rank>"
    exit 1
fi

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_HCA=rocep
export NCCL_IB_MERGE_NICS=0
export NCCL_IB_GID_INDEX=0
export NCCL_IB_TIMEOUT=22
export CUDA_VISIBLE_DEVICES=0
export NCCL_IB_DISABLE=0
export NCCL_MIN_NCHANNELS=4
export NCCL_CROSS_NIC=2

torchrun --nnodes=2 --nproc_per_node=1 \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    send_recv.py