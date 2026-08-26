#!/usr/bin/env bash
set -euo pipefail

WORKLOAD="${WORKLOAD:-dp_reconfig.py}"
nsys profile \
    --output "my_profile_${WORKLOAD}.nsys-rep" \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    torchrun --standalone --nproc_per_node=2 "${WORKLOAD}"
