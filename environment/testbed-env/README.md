# Testbed

## Build docker container and run
```
docker build -t opus-cuda13 .

docker run -it \
  --gpus all \
  --network=host \
  --ipc=host \
  --cap-add=SYS_ADMIN \
  --device=/dev/infiniband \
  -v $(pwd):/Opus \
  -v $(pwd)/hostfile:/etc/hosts \
  --privileged \
  --shm-size=2g \
  --ulimit memlock=-1 \
  --cap-add SYS_NICE \
  opus-cuda13 \
  bash -c "/usr/sbin/sshd && bash"

```

Build NCCL

```
cd NCCL
make -j src.build
NCCL_HOME=/Opus/nccl/build/
```
```
cd torchtitan
pip install -r requirements.txt
hf auth login
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer

```
<!-- ## Install
```
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
sudo docker run --gpus all -it --rm  -v ~/Opus:/Opus    -w /Opus    pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel     bash

# or: nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

# upgrade nccl to 2.28
# build pytorch
<!-- git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
git checkout <nightly-or-tag>
export USE_CUDA=1
export USE_SYSTEM_NCCL=1
export NCCL_ROOT_DIR=/usr/lib/x86_64-linux-gnu   # path where you installed NCCL 2.28
python3 setup.py install -->


