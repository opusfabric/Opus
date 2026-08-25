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

# Install CUDA toolkit (already in image, but just in case)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-8 && \
    rm -rf /var/lib/apt/lists/* cuda-keyring_1.1-1_all.deb

# Copy project code
COPY . /Opus

# Upgrade pip
RUN python -m pip install --upgrade pip

# -------------------------------
# Install Python dependencies
# -------------------------------
RUN cd /Opus/torchtitan && \
    pip install -r requirements.txt && \
    pip install pyyaml

# The emulation controller does not require the hardware-only PyPolatis
# package. Build the small TCP controller here so the container can run the
# software emulation without depending on the caller's working directory.
RUN make -C /Opus/src/opus-controller

# Default command
CMD ["/bin/bash"]
