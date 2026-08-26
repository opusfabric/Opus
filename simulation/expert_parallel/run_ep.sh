#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
OPUS_ROOT="$(realpath ${SCRIPT_DIR:?}/../..)"

ORIG_BACKEND="${OPUS_ROOT}/simulation/orig_backend"
ANALYTICAL_BACKEND="${OPUS_ROOT}/simulation/analytical_backend"
RECONFIG_BACKEND="${OPUS_ROOT}/simulation/reconfig_backend"

ORIG_EXE="${ORIG_BACKEND}/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"
ANALYTICAL_EXE="${ANALYTICAL_BACKEND}/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"
RECONFIG_EXE="${RECONFIG_BACKEND}/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"

STG_DIR="${OPUS_ROOT}/simulation/symbolic_tensor_graph"

# Try to generate an example workload
TP=1
PP=1
DP=2
EP=2

MIXED_PRECISION=0
SCALE_OUT_SWEEPS="50"

cd ${RECONFIG_BACKEND}
echo "Generating experiment with TP=${TP}, PP=${PP}, DP=${DP}, EP=${EP} MIXED_PRECISION=${MIXED_PRECISION}, SCALE_OUT_SWEEPS=${SCALE_OUT_SWEEPS} ..."
SCALE_OUT_SWEEPS=${SCALE_OUT_SWEEPS} TP=${TP} PP=${PP} DP=${DP} EP=${EP} MIXED_PRECISION=${MIXED_PRECISION} ./run_stg_exp_fg_pp.sh > /dev/null

OUT_DIR=${RECONFIG_BACKEND}/examples/stg_dp${DP}_pp${PP}_tp${TP}_ep${EP}_batch_256_mb-1_96stack_seq4096_${SCALE_OUT_SWEEPS}BW

if [ ! -d "${OUT_DIR}" ]; then
    echo "Error: Expected output directory not found at ${OUT_DIR}"
    exit 1
fi

echo "Generated example workload in ${OUT_DIR}"
# cd ${OUT_DIR}

# echo "Running EPS + Ideal"
# ./run_network_analytical.sh > /dev/null