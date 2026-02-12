#!/usr/bin/env bash

#SBATCH --job-name=opus-multinode
#SBATCH --nodes=2
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
JOB="dp-2-tp-4-pm"
CONFIG_FILE=${CONFIG_FILE:-"~/Opus/torchtitan/opus-test/$JOB/llama3_debug.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}
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
shifter --image=ericd16/opus:1.0 --module=gpu,nccl-2.18 \
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

# Launch controller only on first node
NODE_HOSTNAME=\$(hostname)
if [ \"\$NODE_HOSTNAME\" = \"$(echo $SERVER_IPS | cut -d',' -f1)\" ]; then
    echo '[INFO] Launching controller on \$NODE_HOSTNAME'
    cd ~/Opus/src/opus-controller
    ./controller -d 0 -e -r 4 > ~/Opus/torchtitan/opus-test/$JOB/controller_4.ans 2>&1 &
fi

# Launch torchrun across all local GPUs
cd ~/Opus/torchtitan
torchrun --nnodes=$NUM_NODES --nproc_per_node=$NUM_RANKS_PER_NODE \
         --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
         --role rank --tee 3 \
         -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} 2>&1
        #  > ./opus-test/$JOB/output_node_\${SLURM_PROCID}_2.ans 2>&1
"

