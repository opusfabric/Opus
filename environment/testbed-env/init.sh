#!/usr/bin/env bash
# install_conda.sh
# Script to install Miniconda (latest version)

set -e  # Exit on error


# Check if conda is already installed
if command -v conda &> /dev/null; then
    echo "Conda is already installed. Version: $(conda --version)"
    echo "Proceeding to CUDA installation..."
else
    # Detect OS
    OS_TYPE=$(uname)
    if [[ "$OS_TYPE" == "Linux" ]]; then
        CONDA_OS="Linux"
    elif [[ "$OS_TYPE" == "Darwin" ]]; then
        CONDA_OS="MacOSX"
    else
        echo "Unsupported OS: $OS_TYPE"
        exit 1
    fi

    # Choose Miniconda installer URL
    MINICONDA_VERSION="latest"
    MINICONDA_INSTALLER="Miniconda3-${MINICONDA_VERSION}-${CONDA_OS}-x86_64.sh"
    INSTALLER_URL="https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}"

    # Installation directory
    INSTALL_DIR="$HOME/miniconda3"

    echo "Downloading Miniconda installer..."
    curl -fsSL -o "$MINICONDA_INSTALLER" "$INSTALLER_URL"

    echo "Running installer..."
    bash "$MINICONDA_INSTALLER" -b -p "$INSTALL_DIR"

    echo "Initializing conda..."
    # Add conda to PATH in ~/.bashrc or ~/.zshrc
    if [[ -n "$ZSH_VERSION" ]]; then
        SHELL_RC="$HOME/.zshrc"
    else
        SHELL_RC="$HOME/.bashrc"
    fi

    "$INSTALL_DIR/bin/conda" init

    # Ensure PATH update takes effect
    echo "source $INSTALL_DIR/etc/profile.d/conda.sh" >> "$SHELL_RC"

    echo "Cleaning up..."
    rm "$MINICONDA_INSTALLER"

    echo "Installation complete! Please restart your shell or run:"
    echo "    source $SHELL_RC"
    echo "Then test with: conda --version"
fi

# Proceed to CUDA installation
echo "Starting CUDA installation..."
# Add CUDA installation logic here

# Check if CUDA is already installed
if command -v nvcc &> /dev/null; then
    echo "CUDA is already installed. Version: $(nvcc --version | grep 'release' | awk '{print $6}' | sed 's/,//')"
else
    echo "CUDA is not installed. Proceeding with installation..."

    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
    sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/13.0.1/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.1-580.82.07-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2404-13-0-local_13.0.1-580.82.07-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2404-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-13-0

    echo "CUDA installation complete!"
    # Add CUDA to PATH and LD_LIBRARY_PATH in ~/.bashrc or ~/.zshrc
    echo "export PATH=/usr/local/cuda/bin:\$PATH" >> "$SHELL_RC"
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> "$SHELL_RC"

    # Reload shell configuration
    source "$SHELL_RC"

    echo "Environment variables updated. CUDA setup complete!"
fi

# Proceed to NCCL installation
echo "Starting NCCL installation..."
# Check if NCCL is already installed
if dpkg-query -W libnccl2 &> /dev/null && dpkg-query -W libnccl-dev &> /dev/null; then
    echo "NCCL is already installed. Version: $(dpkg-query -W -f='${Version}' libnccl2)"
else
    echo "NCCL is not installed. Proceeding with installation..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt install libnccl2=2.28.3-1+cuda13.0 libnccl-dev=2.28.3-1+cuda13.0
    echo "NCCL installation complete! Installed direction:"

    ldconfig -p | grep nccl
fi

# Proceed to MPI installation
echo "Checking MPI installation..."
if command -v mpirun &> /dev/null; then
    echo "MPI is already installed. Version: $(mpirun --version | head -n 1)"
else
    echo "MPI is not installed. Proceeding with OpenMPI installation..."
    sudo apt-get update
    sudo apt-get -y install openmpi-bin openmpi-common libopenmpi-dev
    echo "OpenMPI installation complete!"
    echo "Installed version: $(mpirun --version | head -n 1)"
fi

# Clone the NCCL test repository and build
echo "Cloning NCCL test repository..."
git clone https://github.com/NVIDIA/nccl-tests.git "$HOME/nccl-tests"

echo "Building NCCL tests..."
cd "$HOME/nccl-tests"
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi/ CUDA_HOME=/usr/local/cuda NCCL_HOME=/lib/x86_64-linux-gnu/



echo "NCCL tests built successfully!"
