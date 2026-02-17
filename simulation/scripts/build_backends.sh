SCRIPT_DIR=$(dirname "$(realpath "$0")")
OPUS_ROOT="$(realpath ${SCRIPT_DIR:?}/../..)"

ANALYTICAL_BACKEND="${OPUS_ROOT}/simulation/analytical_backend"
RECONFIG_BACKEND="${OPUS_ROOT}/simulation/reconfig_backend"

ANALYTICAL_EXE="${ANALYTICAL_BACKEND}/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"
RECONFIG_EXE="${RECONFIG_BACKEND}/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable"

# Initialize and update backend modules
cd "${OPUS_ROOT}"
git submodule update --init --recursive

# Build the analytical network backend
echo "Building the Analytical Network Backend..."
cd "${ANALYTICAL_BACKEND}"
git checkout analy-w-bw-mat
./build/astra_analytical/build.sh > /dev/null

if [ ! -f "${ANALYTICAL_EXE}" ]; then
    echo "Error: Analytical Network Backend build failed. Executable not found at ${ANALYTICAL_EXE}."
    exit 1
fi

# Build the reconfigurable network backend
echo "Building the Reconfigurable Network Backend..."
cd "${RECONFIG_BACKEND}"
./build/astra_analytical/build.sh > /dev/null

if [ ! -f "${RECONFIG_EXE}" ]; then
    echo "Error: Reconfigurable Network Backend build failed. Executable not found at ${RECONFIG_EXE}."
    exit 1
fi

echo "Build completed successfully. Executables are located at:"
echo " - Analytical Network Backend: ${ANALYTICAL_EXE}"
echo " - Reconfigurable Network Backend: ${RECONFIG_EXE}"