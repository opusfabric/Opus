#!/usr/bin/env bash
# Submit one Figure 10/11 GPU experiment with the matching TorchTitan config
# and archived communication pattern.

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: scripts/submit_figure10_11.sh CASE MODE DELAY_MS [SBATCH_OPTIONS...]

CASE: llama16 | llama64 | deepseek2d | deepseek3d
MODE: baseline | provision

Examples:
  scripts/submit_figure10_11.sh llama16 baseline 50 --account=YOUR_ACCOUNT
  scripts/submit_figure10_11.sh llama16 provision 50 --account=YOUR_ACCOUNT

Set OPUS_SLURM_LAUNCHER=perlmutter to use the Shifter launcher.
The command prints only the submitted Slurm job ID on success.
EOF
    exit 2
}

(( $# >= 3 )) || usage
CASE_NAME=$1
MODE=$2
DELAY_MS=$3
shift 3

[[ ${MODE} == baseline || ${MODE} == provision ]] || usage
[[ ${DELAY_MS} =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "DELAY_MS must be a non-negative number: ${DELAY_MS}" >&2
    exit 2
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OPUS_ROOT=${OPUS_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}

case ${CASE_NAME} in
    llama16)
        EXPECTED_RANKS=16
        CONFIG_FILE=${OPUS_ROOT}/torchtitan/opus-test/dp-2-pp-2-tp-4-pm-8b/llama3_8b_lb.toml
        COMM_PATTERN_PATH=${OPUS_ROOT}/evaluation/llama-3-3d-16-latency/comm_pattern/comm_pattern
        USE_CUDA_STREAM=0
        ;;
    llama64)
        EXPECTED_RANKS=64
        CONFIG_FILE=${OPUS_ROOT}/torchtitan/opus-test/dp-8-pp-2-tp-4-pm-8b/llama3_8b_lb_dp8.toml
        COMM_PATTERN_PATH=${OPUS_ROOT}/evaluation/llama-3-3d-64-latency/comm_pattern_no_ctl_issue/comm_pattern
        USE_CUDA_STREAM=0
        ;;
    deepseek2d)
        EXPECTED_RANKS=16
        CONFIG_FILE=${OPUS_ROOT}/torchtitan/opus-test/deepseek-pp-4-tp-4-pm/deepseek_v3_16b.toml
        COMM_PATTERN_PATH=${OPUS_ROOT}/evaluation/deepseek_v3_16b-2d-16-latency/comm_pattern/comm_pattern
        USE_CUDA_STREAM=3
        ;;
    deepseek3d)
        EXPECTED_RANKS=16
        CONFIG_FILE=${OPUS_ROOT}/torchtitan/opus-test/deepseek-dp-2-pp-2-tp-4-pm/deepseek_v3_16b.toml
        COMM_PATTERN_PATH=${OPUS_ROOT}/evaluation/deepseek_v3_16b-3d-16-latency/comm_pattern/comm_pattern
        USE_CUDA_STREAM=1
        ;;
    *)
        usage
        ;;
esac

[[ ${GPUS_PER_NODE} =~ ^[1-9][0-9]*$ ]] || {
    echo "GPUS_PER_NODE must be a positive integer: ${GPUS_PER_NODE}" >&2
    exit 2
}
if (( EXPECTED_RANKS % GPUS_PER_NODE != 0 )); then
    echo "${EXPECTED_RANKS} ranks cannot be divided across ${GPUS_PER_NODE} GPUs per node" >&2
    exit 2
fi
NUM_NODES=$((EXPECTED_RANKS / GPUS_PER_NODE))

[[ -f ${CONFIG_FILE} ]] || { echo "Missing config: ${CONFIG_FILE}" >&2; exit 1; }
[[ -f ${COMM_PATTERN_PATH}_0_rank0.txt ]] || {
    echo "Missing communication pattern: ${COMM_PATTERN_PATH}_0_rank0.txt" >&2
    exit 1
}
command -v sbatch >/dev/null || { echo "sbatch is not available" >&2; exit 1; }

LAUNCHER_ROOT=${OPUS_ROOT}/scripts
case ${OPUS_SLURM_LAUNCHER:-generic} in
    generic)
        ;;
    perlmutter)
        LAUNCHER_ROOT=${LAUNCHER_ROOT}/perlmutter
        ;;
    *)
        echo "OPUS_SLURM_LAUNCHER must be generic or perlmutter" >&2
        exit 2
        ;;
esac

if [[ ${MODE} == provision ]]; then
    LAUNCHER=${LAUNCHER_ROOT}/run_opus_provision.sbatch
else
    LAUNCHER=${LAUNCHER_ROOT}/run_opus_emulation.sbatch
fi

EXPORTS="ALL,OPUS_ROOT=${OPUS_ROOT},CONFIG_FILE=${CONFIG_FILE}"
EXPORTS+=",COMM_PATTERN_PATH=${COMM_PATTERN_PATH},NUM_RANKS_PER_NODE=${GPUS_PER_NODE}"
EXPORTS+=",RECONFIG_DELAY_MS=${DELAY_MS},USE_CUDA_STREAM=${USE_CUDA_STREAM}"

SUBMISSION=$(sbatch --parsable \
    --nodes="${NUM_NODES}" \
    --ntasks-per-node=1 \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --job-name="opus-${CASE_NAME}-${MODE}-${DELAY_MS}ms" \
    --export="${EXPORTS}" \
    "$@" \
    "${LAUNCHER}")

# Some federated Slurm installations return "job-id;cluster" with
# --parsable. runs/ is keyed by SLURM_JOB_ID, so expose the numeric portion.
printf '%s\n' "${SUBMISSION%%;*}"
