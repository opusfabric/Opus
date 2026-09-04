# Opus common emulation and simulation image

This image provides the shared environment for the first two-GPU shim smoke test, full GPU-cluster emulation, and CPU software simulation. It includes CUDA-enabled PyTorch, CUDA development tools, NCCL build support, CMake, Protobuf, Graphviz, the simulator Python packages, and the controller.

## Build from the Opus repository root

```bash
cd /path/to/Opus
docker build --network=host -f environment/emulation-env/Dockerfile -t opus-emulation:artifact .
```

The repository `.dockerignore` excludes the local Python venv, build products, raw Chakra traces, profiler outputs, and logs from the image context.

## Two-GPU shim smoke test

On a host with the NVIDIA Container Toolkit:

```bash
docker run --rm -it --gpus 2 --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$(pwd):/Opus" -w /Opus opus-emulation:artifact bash
```

Inside the container:

```bash
make -B -C src/opus-controller
make -C nccl -j"${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}" src.build
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation -e src/opus-shim
```

If the shim source was changed after an earlier install, run `bash src/opus-shim/build.sh` to remove stale in-place binaries and rebuild it.

Then follow Section 1 of the root README.md to start the local controller and run src/opus-shim/test/dp_reconfig.py. That example uses two replicated data-parallel ranks and intentionally exercises one emulated topology transition. For the single-host Docker demo, keep SERVER_IPS=127.0.0.1 and OPUS_FORCE_DOMAIN=scale-out; remove the override on multi-node runs and provide every node hostname or IPv4 address in rank order.

## CPU software simulation

The simulator does not need GPUs:

```bash
docker run --rm -it --network host \
  -v "$(pwd):/Opus" -w /Opus opus-emulation:artifact bash
```

Inside the container:

```bash
./simulation/scripts/build_backends.sh
./simulation/scripts/run_example.sh
```

For Slurm, follow Section 2 of the root README for either Docker or the site-approved runtime such as Shifter, Enroot, Apptainer, or Charliecloud. Expose the GPUs and RDMA devices for emulation; the CPU simulator can run without GPU passthrough.
