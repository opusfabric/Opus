#!/usr/bin/env bash
# End-to-end Figure 4(a-c) trace capture on 256 GPUs.
# The defaults request 32 nodes with 8 GPUs each. Sites with a different GPU
# density can override NUM_RANKS_PER_NODE, but the total must remain 256.

#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --job-name=opus-fig4-trace

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OPUS_ROOT=${OPUS_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
CONFIG_FILE=${CONFIG_FILE:-${SCRIPT_DIR}/deepseek_236b_trace.toml}
PYTHON=${PYTHON:-python3}

: "${SLURM_JOB_ID:?Submit this file with sbatch}"
: "${SLURM_JOB_NUM_NODES:?Slurm did not report the allocated node count}"

NUM_NODES=${NUM_NODES:-${SLURM_JOB_NUM_NODES}}
NUM_RANKS_PER_NODE=${NUM_RANKS_PER_NODE:-8}
WORLD_SIZE=$((NUM_NODES * NUM_RANKS_PER_NODE))
if (( WORLD_SIZE != 256 )); then
    echo "Figure 4 requires 256 ranks; got ${NUM_NODES} nodes x ${NUM_RANKS_PER_NODE} GPUs = ${WORLD_SIZE}." >&2
    exit 2
fi

TRACE_ROOT=${TRACE_ROOT:-${OPUS_ROOT}/runs/${SLURM_JOB_ID}/figure4}
PROFILE_FREQ=${PROFILE_FREQ:-5}
PROFILE_STEPS=${PROFILE_STEPS:-5}
export OPUS_ROOT CONFIG_FILE NUM_NODES NUM_RANKS_PER_NODE
export TRACE_ROOT PROFILE_FREQ PROFILE_STEPS

# Fail before allocating model memory if the shared tokenizer or plotting
# environment is missing. Training starts from random weights; no checkpoint is
# downloaded or loaded.
TOKENIZER_DIR=${OPUS_ROOT}/torchtitan/assets/hf/DeepSeek-V2
if [[ ! -f ${TOKENIZER_DIR}/tokenizer.json ]]; then
    echo "Missing ${TOKENIZER_DIR}/tokenizer.json; run the README tokenizer command first." >&2
    exit 2
fi
"${PYTHON}" -c 'import matplotlib, numpy'

mkdir -p "${TRACE_ROOT}"
bash "${OPUS_ROOT}/scripts/run_slurm_trace_profile.sbatch"

# TorchTitan writes one Chrome trace per global rank under iteration_<step>.
# Preserve all 256 raw traces under runs/<job-id>/ and stage ranks 0-3 beside
# the Figure 4 analysis scripts, matching the paper analysis input set.
mapfile -t ITERATION_DIRS < <(
    find "${TRACE_ROOT}/profile_trace" -type f -name rank0_trace.json -printf '%h\n' | sort -V
)
if (( ${#ITERATION_DIRS[@]} == 0 )); then
    echo "No rank traces were produced under ${TRACE_ROOT}/profile_trace." >&2
    exit 3
fi
ITERATION_DIR=${ITERATION_DIRS[${#ITERATION_DIRS[@]}-1]}

TRACE_COUNT=$(find "${ITERATION_DIR}" -maxdepth 1 -type f -name 'rank*_trace.json' | wc -l)
if (( TRACE_COUNT != WORLD_SIZE )); then
    echo "Expected ${WORLD_SIZE} traces in ${ITERATION_DIR}; found ${TRACE_COUNT}." >&2
    exit 3
fi

for rank in 0 1 2 3; do
    cp "${ITERATION_DIR}/rank${rank}_trace.json" "${SCRIPT_DIR}/rank${rank}_trace.json"
done

cd "${SCRIPT_DIR}"
"${PYTHON}" trace_timing_gap.py > trace_timing_gap.txt
"${PYTHON}" timing_gap_deepseek_236b_256gpu.py > timing_gap_model.txt
"${PYTHON}" comm_gap_deepseek_236b_256gpu.py > comm_gap_model.txt
"${PYTHON}" plot_transition_cdf.py > transition_cdf_summary.txt

echo "Figure 4(a-c) artifacts are in ${SCRIPT_DIR}."
echo "All ${TRACE_COUNT} raw GPU traces are retained in ${ITERATION_DIR}."
