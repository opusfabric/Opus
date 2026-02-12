WORKLOAD="pp.py"
nsys profile \
    --output my_profile_$WORKLOAD.nsys-rep \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    torchrun --nproc-per-node=2 $WORKLOAD