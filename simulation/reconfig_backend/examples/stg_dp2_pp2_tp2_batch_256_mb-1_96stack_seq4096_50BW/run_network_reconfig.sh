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
ASTRA_SIM="${PROJECT_DIR:?}/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"
WORKLOAD="${EXAMPLE_DIR:?}/workload"
SYSTEM="${EXAMPLE_DIR:?}/system.json"
NETWORK="${EXAMPLE_DIR:?}/network.yml"
REMOTE_MEMORY="${EXAMPLE_DIR:?}/remote_memory.json"
COMM_GROUP="${EXAMPLE_DIR:?}/workload.json"
CIRCUIT_SCHEDULES="${EXAMPLE_DIR:?}/schedules.txt"

TRACE_PARSER_PATH="${PROJECT_DIR:?}/examples/helpers/trace_parser.py"

# start
echo "[ASTRA-sim] Compiling ASTRA-sim with the Reconfigurable Network Backend..."
echo ""

# Compile
if [[ "${OPUS_SKIP_LEGACY_BUILD:-0}" != "1" ]]; then
    "${PROJECT_DIR:?}"/build/astra_analytical/build.sh
fi

echo ""
echo "[ASTRA-sim] Compilation finished."
echo "[ASTRA-sim] Running ASTRA-sim Example with Reconfigurable Network Backend..."
echo ""

# run ASTRA-sim
export ASAN_OPTIONS=detect_container_overflow=0:detect_leaks=0

"${ASTRA_SIM:?}" \
    --workload-configuration="${WORKLOAD}" \
    --system-configuration="${SYSTEM:?}" \
    --remote-memory-configuration="${REMOTE_MEMORY:?}" \
    --network-configuration="${NETWORK:?}" \
    --comm-group-configuration="${COMM_GROUP:?}" \
    --circuit-schedules="${CIRCUIT_SCHEDULES:?}" > debug_no_provision.txt

python3 ${TRACE_PARSER_PATH:?} debug_no_provision.txt

if [[ "${OPUS_SKIP_PROVISION:-0}" != "1" ]]; then
    PROVISION_CONFIG="${EXAMPLE_DIR:?}/rank_comm_groups.yaml"

"${ASTRA_SIM:?}" \
    --workload-configuration="${WORKLOAD}" \
    --system-configuration="${SYSTEM:?}" \
    --remote-memory-configuration="${REMOTE_MEMORY:?}" \
    --network-configuration="${NETWORK:?}" \
    --comm-group-configuration="${COMM_GROUP:?}" \
    --circuit-schedules="${CIRCUIT_SCHEDULES:?}" \
    --provision-config="${PROVISION_CONFIG:?}" > debug_provision.txt
fi


echo ""
echo "NON-PROVISIONED RUN OUTPUT:"
tail -n 20 debug_no_provision.txt

if [[ ${OPUS_SKIP_PROVISION:-0} == "1" ]]; then
    echo "PROVISIONING RUN SKIPPED"
elif [[ -s debug_provision.txt ]]; then
    echo ""
    echo "PROVISIONED RUN OUTPUT:"
    tail -n 20 debug_provision.txt
else
    echo "PROVISIONING RUN SKIPPED"
fi