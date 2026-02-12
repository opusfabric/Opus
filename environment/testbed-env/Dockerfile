# Base image: PyTorch Nightly (CUDA 13.0 + cuDNN9)
FROM ghcr.io/pytorch/pytorch-nightly:2.10.0.dev20250914-cuda13.0-cudnn9-runtime

WORKDIR /Opus
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV CPATH=/usr/local/cuda/include
ENV LIBRARY_PATH=/usr/local/cuda/lib64

# Install deps + RDMA tools + SSH
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
        vim \
        iputils-ping \
        net-tools \
        openssh-client \
        openssh-server \
        rdma-core \
        infiniband-diags \
        libibverbs1 \
        ibverbs-providers \
        openmpi-bin \
        libopenmpi-dev \
        pciutils \
        perftest && \
    rm -rf /var/lib/apt/lists/*

# CUDA Toolkit 13.0 install
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-13-0 && \
    rm -rf /var/lib/apt/lists/* cuda-keyring_1.1-1_all.deb

# NCCL symlink
RUN ln -sf /opt/conda/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2 \
           /opt/conda/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so

RUN mkdir -p /var/run/sshd && \
    mkdir -p /root/.ssh && chmod 700 /root/.ssh && \
    ssh-keygen -t rsa -f /root/.ssh/id_rsa -q -N "" && \
    cat /root/.ssh/id_rsa.pub > /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys && \
    echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config && \
    echo "UserKnownHostsFile=/dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's@session\s\+required\s\+pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

EXPOSE 22
EXPOSE 29500/tcp

RUN mkdir -p /Opus
RUN mkdir -p /etc
RUN mkdir -p /etc/hosts

CMD ["/usr/sbin/sshd", "-D"]
