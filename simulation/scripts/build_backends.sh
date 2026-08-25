#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
OPUS_ROOT="$(realpath "${SCRIPT_DIR}/../..")"

ANALYTICAL_BACKEND="${OPUS_ROOT}/simulation/analytical_backend"
RECONFIG_BACKEND="${OPUS_ROOT}/simulation/reconfig_backend"

ANALYTICAL_EXE="${ANALYTICAL_BACKEND}/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"
RECONFIG_EXE="${RECONFIG_BACKEND}/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"

build_backend() {
    local backend="$1"
    local executable="$2"
    local build_dir="${backend}/build/astra_analytical/build"

    echo "Configuring ${backend}..."
    cmake -S "${backend}/build/astra_analytical" -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
        -DBUILDTARGET=all
    cmake --build "${build_dir}" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"

    if [ ! -x "${executable}" ]; then
        echo "Error: build completed without executable ${executable}" >&2
        exit 1
    fi
}

echo "Building the Analytical Network Backend..."
build_backend "${ANALYTICAL_BACKEND}" "${ANALYTICAL_EXE}"

echo "Building the Reconfigurable Network Backend..."
build_backend "${RECONFIG_BACKEND}" "${RECONFIG_EXE}"
echo "Build completed successfully. Executables are located at:"
echo " - Analytical Network Backend: ${ANALYTICAL_EXE}"
echo " - Reconfigurable Network Backend: ${RECONFIG_EXE}"
