#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
OPUS_ROOT="$(realpath "${SCRIPT_DIR}/../..")"

RECONFIG_BACKEND="${OPUS_ROOT}/simulation/reconfig_backend"

RECONFIG_EXE="${RECONFIG_BACKEND}/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"

build_backend() {
    local backend="$1"
    local executable="$2"
    local build_dir="${backend}/build/astra_analytical/build"
    local source_dir="${backend}/build/astra_analytical"
    local cached_source=""
    if [[ -f "${build_dir}/CMakeCache.txt" ]]; then
        cached_source=$(sed -n "s/^CMAKE_HOME_DIRECTORY:INTERNAL=//p" "${build_dir}/CMakeCache.txt")
        if [[ "${cached_source}" != "${source_dir}" ]]; then
            echo "Resetting stale CMake cache from ${cached_source}"
            rm -rf "${build_dir}"
        fi
    fi

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

echo "Building the Reconfigurable Network Backend..."
build_backend "${RECONFIG_BACKEND}" "${RECONFIG_EXE}"
echo "Build completed successfully. Executables are located at:"
echo " - Reconfigurable Network Backend: ${RECONFIG_EXE}"
