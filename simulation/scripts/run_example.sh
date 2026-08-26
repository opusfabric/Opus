#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
OPUS_ROOT="$(realpath ${SCRIPT_DIR:?}/../..)"

ANALYTICAL_BACKEND="${OPUS_ROOT}/simulation/analytical_backend"
RECONFIG_BACKEND="${OPUS_ROOT}/simulation/reconfig_backend"

ANALYTICAL_EXE="${ANALYTICAL_BACKEND}/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"
RECONFIG_EXE="${RECONFIG_BACKEND}/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"

STG_DIR="${OPUS_ROOT}/simulation/symbolic_tensor_graph"
: "${PYTHON:=python3}"
export PYTHON

if ! "${PYTHON}" -c "import graphviz, numpy, pandas, sympy, tqdm, yaml, google.protobuf" >/dev/null 2>&1; then
    echo "Error: the STG generator needs Python packages (graphviz, numpy, pandas, sympy, tqdm, pyyaml, protobuf)." >&2
    echo "Activate the repository .venv first, then install with: ${PYTHON} -m pip install numpy sympy graphviz pandas tqdm pyyaml protobuf" >&2
    exit 2
fi

# Try to generate an example workload
TP=2
PP=2
DP=2
MIXED_PRECISION=0
SCALE_OUT_SWEEPS="50"

cd "${RECONFIG_BACKEND}"
echo "Generating experiment with TP=${TP}, PP=${PP}, DP=${DP}, MIXED_PRECISION=${MIXED_PRECISION}, SCALE_OUT_SWEEPS=${SCALE_OUT_SWEEPS} ..."
SCALE_OUT_SWEEPS=${SCALE_OUT_SWEEPS} TP=${TP} PP=${PP} DP=${DP} MIXED_PRECISION=${MIXED_PRECISION} ./run_stg_exp_fg_pp.sh

OUT_DIR=${RECONFIG_BACKEND}/examples/stg_dp${DP}_pp${PP}_tp${TP}_batch_256_mb-1_96stack_seq4096_${SCALE_OUT_SWEEPS}BW

if [ ! -d "${OUT_DIR}" ]; then
    echo "Error: Expected output directory not found at ${OUT_DIR}"
    exit 1
fi

echo "Generated example workload in ${OUT_DIR}"
cd "${OUT_DIR}"

echo "Running EPS + Ideal"
./run_network_analytical.sh > /dev/null

if [ ! -s "debug_analytical.txt" ] || [ ! -s "debug_baseline.txt" ]; then
    echo "Error: Expected output file not found in ${OUT_DIR}"
    exit 1
fi

echo "Running Opus + Opus w/ Provision"
./run_network_reconfig.sh > /dev/null

if [ ! -f "debug_no_provision.txt" ] || [ ! -f "debug_provision.txt" ]; then
    echo "Error: Expected output files not found in ${OUT_DIR}"
    exit 1
fi

echo "All runs completed."
echo "Output files generated in ${OUT_DIR}"