#!/bin/bash
set -e

## ******************************************************************************
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.
##
## Copyright (c) 2024 Georgia Institute of Technology
## ******************************************************************************

# find the absolute path to this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR="${SCRIPT_DIR:?}/../.."
EXAMPLE_DIR="${SCRIPT_DIR:?}"

# paths
ASTRA_SIM="/home/soxehli/work/astra-sim-hier-analytical/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"
WORKLOAD="${EXAMPLE_DIR:?}/workload"
SYSTEM="${EXAMPLE_DIR:?}/system.json"
NETWORK="${EXAMPLE_DIR:?}/network.yml"
REMOTE_MEMORY="${EXAMPLE_DIR:?}/remote_memory.json"
COMM_GROUP="${EXAMPLE_DIR:?}/workload.json"
CIRCUIT_SCHEDULES="${EXAMPLE_DIR:?}/schedules-collapsed.txt"
BASELINE_SCHEDULES="${EXAMPLE_DIR:?}/schedules_baseline.txt"
BASELINE_TP_GEN="${EXAMPLE_DIR:?}/baseline_bw_log.txt"

# start
echo "[ASTRA-sim] Compiling ASTRA-sim with the Analytical Network Backend..."
echo ""

# Compile
"${PROJECT_DIR:?}"/build/astra_analytical/build.sh

echo ""
echo "[ASTRA-sim] Compilation finished."
echo "[ASTRA-sim] Running ASTRA-sim Example with Analytical Network Backend..."
echo ""

# run ASTRA-sim
export ASAN_OPTIONS=detect_container_overflow=0

"${ASTRA_SIM:?}" \
    --workload-configuration="${WORKLOAD}" \
    --system-configuration="${SYSTEM:?}" \
    --remote-memory-configuration="${REMOTE_MEMORY:?}" \
    --network-configuration="${NETWORK:?}" \
    --comm-group-configuration="${COMM_GROUP:?}" \
    --circuit-schedules="${CIRCUIT_SCHEDULES:?}" > debug_analytical.txt

# finalize
echo ""
echo "[ASTRA-sim] Finished the execution."

python ../helpers/topo_gen_baseline.py 4 8 4 25 450 > "${BASELINE_TP_GEN:?}"

"${ASTRA_SIM:?}" \
    --workload-configuration="${WORKLOAD}" \
    --system-configuration="${SYSTEM:?}" \
    --remote-memory-configuration="${REMOTE_MEMORY:?}" \
    --network-configuration="${NETWORK:?}" \
    --comm-group-configuration="${COMM_GROUP:?}" \
    --circuit-schedules="${BASELINE_SCHEDULES:?}" > debug_baseline.txt

# finalize
echo ""
echo "[ASTRA-sim] Finished the execution."

echo ""
echo "ANALYTICAL RUN OUTPUT:"
tail -n 20 debug_analytical.txt

echo ""
echo "BASELINE RUN OUTPUT:"
tail -n 20 debug_baseline.txt