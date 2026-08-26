FROM ghcr.io/pytorch/pytorch-nightly:2.10.0.dev20251209-cuda12.8-cudnn9-devel

# Set noninteractive frontend
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /Opus

# CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV CPATH=/usr/local/cuda/include
ENV LIBRARY_PATH=/usr/local/cuda/lib64

# Hugging Face cache
ENV HF_HOME=/tmp/hf
ENV HF_DATASETS_CACHE=/tmp/hf/datasets
ENV TRANSFORMERS_CACHE=/tmp/hf/transformers
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
RUN mkdir -p /tmp/hf/datasets /tmp/hf/transformers

# Base utilities + RDMA userspace
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        graphviz \
        libprotobuf-dev \
        protobuf-compiler \
        python3-dev \
        git \
        curl \
        wget \
        vim \
        iputils-ping \
        net-tools \
        rdma-core \
        infiniband-diags \
        libibverbs1 \
        ibverbs-providers \
        pciutils \
        perftest \
    && rm -rf /var/lib/apt/lists/*

# The PyTorch CUDA devel base image already supplies nvcc and the CUDA toolkit.
# Do not reinstall CUDA through apt: mixing repository revisions can break the image build.
RUN nvcc --version

# Copy project code
COPY . /Opus

# Keep the common image usable for both the GPU shim and CPU simulation.
RUN python -m pip install --upgrade pip setuptools wheel

# -------------------------------
# Install Python dependencies for the shim, simulator, and plotting helpers
# -------------------------------
RUN python -m pip install \
        numpy \
        sympy \
        graphviz \
        pandas \
        tqdm \
        pyyaml \
        protobuf==5.28.2 && \
    cd /Opus/torchtitan && \
    python -m pip install -r requirements.txt

# The emulation controller does not require the hardware-only PyPolatis
# package. Build the small TCP controller here so the container can run the
# software emulation without depending on the caller's working directory.
RUN make -C /Opus/src/opus-controller

# Default command
CMD ["/bin/bash"]
