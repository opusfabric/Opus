#!/usr/bin/env bash

#SBATCH --job-name=opus-multinode
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --partition=debug      # change as needed
#SBATCH --account=YOUR_ACCOUNT # replace with your allocation

set -e

# --------------------------
# Node / Controller setup
# --------------------------
SERVER_IPS=$(scontrol show hostname $SLURM_JOB_NODELIST | tr '\n' ',' | sed 's/,$//')
CONTROLLER_IP=$(echo "$SERVER_IPS" | cut -d',' -f1)
export MASTER_ADDR="$CONTROLLER_IP"
export MASTER_PORT=29500
RECONFIG_LATENCY=1000 # in milliseconds
MODE="baseline"

# --------------------------
# HF / Transformers caches
# --------------------------
export HF_HOME=/tmp/hf
export HF_DATASETS_CACHE=/tmp/hf/datasets
export TRANSFORMERS_CACHE=/tmp/hf/transformers
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
mkdir -p /tmp/hf/datasets

# --------------------------
# Opus / Torchrun config
# --------------------------
JOB="dp-2-pp-2-tp-4-pm-8b-provision"
CONFIG_FILE=${CONFIG_FILE:-"~/Opus/torchtitan/opus-test/$JOB/llama3_8b.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}
COMM_PATTERN_PATH=${COMM_PATTERN_PATH:-"~/Opus/torchtitan/opus-test/dp-2-pp-2-tp-4-pm-8b/comm_pattern/comm_pattern"}
export LOG_RANK=0,1,2,3
NUM_NODES=${NUM_NODES:-$SLURM_JOB_NUM_NODES}
NUM_RANKS_PER_NODE=${NUM_RANKS_PER_NODE:-4}

# --------------------------
# Launch torchrun via Shifter across all nodes
# --------------------------
export NCCL_ROOT=/opt/udiImage/modules/nccl-2.18
export LD_LIBRARY_PATH=$NCCL_ROOT/lib:/opt/udiImage/modules/gpu/lib64
export LIBRARY_PATH=$LD_LIBRARY_PATH
export CPATH=$NCCL_ROOT/include

srun -N $NUM_NODES -n $NUM_NODES --ntasks-per-node=1 --gpus-per-task=4 \
shifter --image=ericd16/opus:2.0 --module=gpu \
bash -c "
# Install opus-shim inside container

source /opt/conda/etc/profile.d/conda.sh
conda activate base
export PATH=/opt/conda/bin:$PATH

export CONFIG_FILE=$CONFIG_FILE
export NUM_NODES=$NUM_NODES
export NUM_RANKS_PER_NODE=$NUM_RANKS_PER_NODE
export SERVER_IPS=$SERVER_IPS
export CONTROLLER_IP=$CONTROLLER_IP
export COMM_PATTERN_PATH=$COMM_PATTERN_PATH
export MODE=$MODE

export NODE_RANK=$(echo $SERVER_IPS | tr ',' '\n' | grep -n $(hostname) | cut -d':' -f1)
export NODE_RANK=$((NODE_RANK - 1))

# --------------------------
# NCCL / GPU environment
# --------------------------
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

export NCCL_IB_DISABLE=0

export NCCL_LAUNCH_ORDER_IMPLICIT=1
export NCCL_COMM_BLOCKING=1

export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=8

# export NCCL_ROOT=/opt/udiImage/modules/nccl-2.18
# export LD_LIBRARY_PATH=$NCCL_ROOT/lib:/opt/udiImage/modules/gpu/lib64:$LD_LIBRARY_PATH
# export LIBRARY_PATH=$NCCL_ROOT/lib:/opt/udiImage/modules/gpu/lib64:$LIBRARY_PATH
# export CPATH=$NCCL_ROOT/include:$CPATH

# !!!BUILD AND INSTALL OPUS-SHIM WHEN NECESSARY

# cd ~/Opus/src/opus-shim
# pip install -e . --no-build-isolation


# !!!BUILD AND INSTALL OPUS-CONTROLLER WHEN NECESSARY
# cd ~/Opus/src/opus-controller
# make

# Launch controller only on first node
NODE_HOSTNAME=\$(hostname)
if [ \"\$NODE_HOSTNAME\" = \"$(echo $SERVER_IPS | cut -d',' -f1)\" ]; then
    echo '[INFO] Launching controller on \$NODE_HOSTNAME'
    cd ~/Opus/src/opus-controller
    ./controller -d $RECONFIG_LATENCY -e -r 4 -o ~/Opus/torchtitan/opus-test/$JOB \
        > ~/Opus/torchtitan/opus-test/$JOB/controller.ans 2>&1 &
fi

# Launch torchrun across all local GPUs
cd ~/Opus/torchtitan

torchrun --nnodes=$NUM_NODES --nproc_per_node=$NUM_RANKS_PER_NODE \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --role rank \
    -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} \
    > ~/Opus/torchtitan/opus-test/$JOB/torchrun_\$(hostname).ans 2>&1
"

