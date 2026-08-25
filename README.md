# Opus artifact

Opus is a photonic-rail control plane for distributed ML communication. This artifact contains the source used for the paper, archived emulation inputs and outputs, and the analytical software simulator.

This README is the artifact-evaluation guide. It deliberately separates the three evaluation surfaces:

| Surface | What is provided | What a reader needs |
| --- | --- | --- |
| Software simulation | CPU/C++ analytical and reconfigurable ASTRA-sim backends, workload generators, sweep and plotting scripts | Linux build host or the simulator container, Python packages, and enough disk for generated traces |
| Network emulation | Controller, NCCL shim, Slingshot/NCCL launch scripts, communication-pattern inputs, and archived CSV/notebook outputs | A multi-GPU cluster with the site’s Slingshot provider and NCCL/GPUDirect RDMA support |
| Hardware prototype | Controller-side hardware integration and paper description | The Polatis OCS testbed and firmware; this is an overview only and is not expected to run on a reader’s cluster |

## Fastest useful checks

From the repository root:

```bash
cd Opus

# Build and inspect the controller. This does not require a GPU.
make -C src/opus-controller

# Exercise the software OCS-delay model without Slingshot or NCCL.
OPUS_IPC_DIR="$(mktemp -d)" OPUS_IPC_PREFIX=opus_reader_demo \
  python3 -u src/opus-controller/config.py -e 50 1 0
```

The last command is a server and therefore waits for a Unix-socket client. For a complete, finite smoke test, use this instead:

```bash
cd Opus
DEMO_DIR="$(mktemp -d)"
export OPUS_IPC_DIR="${DEMO_DIR}"
export OPUS_IPC_PREFIX=opus_reader_demo
python3 -u src/opus-controller/config.py -e 50 1 0 >"${DEMO_DIR}/controller.log" 2>&1 &
CONFIG_PID=$!
trap "kill ${CONFIG_PID} 2>/dev/null || true; rm -rf ${DEMO_DIR}" EXIT

python3 - "${DEMO_DIR}/${OPUS_IPC_PREFIX}_0" <<PY
import socket
import sys
import time

path = sys.argv[1]
for value in (0, 1, 3, 1):
    for _ in range(200):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(path)
                client.sendall((str(value) + "\n").encode())
                print(value, client.recv(256).decode().strip())
            break
        except FileNotFoundError:
            time.sleep(0.01)
    else:
        raise RuntimeError("timed out waiting for " + path)
PY
```

Expected responses contain `SUCCESS, CONFIG-ACK`. The first request is the initial state; the following requests change one or more logical topology bits and each incurs the configured 50 ms delay. This is the control-plane emulation, not a network dataplane benchmark.

## What actually emulates OCS?

The important answer to the reader question about `environment/emulation-env/Dockerfile` is: the Dockerfile itself does not emulate an OCS.

The Dockerfile only assembles a CUDA/PyTorch/RDMA userspace image. The `rdma-core`, `libibverbs`, and `perftest` packages make it possible to run the native NCCL/Slingshot path inside a suitable cluster container. The image does not create an optical switch, virtual links, a circuit-level network simulator, or a packet-forwarding dataplane. The old Dockerfile also tried to install a hardware-only `scale-out/pypolatis-controller` path that is absent from this artifact; that reference has been removed from the executable emulation image.

The emulated OCS behavior is implemented by this chain:

1. `src/opus-controller/controller.cpp` starts one `config.py` worker per rail and serves the TCP control socket used by the shim.
2. `src/opus-controller/config.py`, when invoked with `-e`, represents a topology ID as 64 logical bits. For every changed bit it starts a worker that sleeps for the configured delay and then updates that bit. Changed bits are processed concurrently, so a transition is approximately one configured delay, not one delay per bit. The worker sends `SUCCESS, CONFIG-ACK` after all changed-bit workers finish.
3. `src/opus-shim/src/opus.cpp` requests topology changes from the controller while the actual NCCL collectives continue to use the host cluster’s normal network provider. In the paper-style setup, start the controller with `-e` and set `IS_EMULATION=0` in the shim. This applies the delay in the controller once.
4. `IS_EMULATION=1` in the shim is a separate host-side delay hook used by the shim’s provisioning path. Do not enable both hooks unless you intentionally want to model two delays.

## Repository layout

```text
Opus/
├── environment/
│   ├── emulation-env/       CUDA/PyTorch/RDMA image for software emulation
│   └── testbed-env/         hardware/testbed image; overview only here
├── evaluation/              archived communication patterns, CSVs, notebooks, PDFs
├── nccl/                    vendored NCCL source used to build the Opus shim
├── simulation/
│   ├── analytical_backend/  vendored ASTRA-sim analytical backend
│   ├── reconfig_backend/    ASTRA-sim backend with reconfigurable topology support
│   ├── symbolic_tensor_graph/ workload/Chakra trace generator
│   └── scripts/             backend build, experiment sweeps, and figure plots
├── src/
│   ├── opus-controller/     C++ TCP controller and Python rail workers
│   └── opus-shim/           PyTorch/NCCL process-group extension
└── torchtitan/opus-test/    paper-oriented Slurm training launchers and configs
```

## Software simulation

The simulator is the most portable evaluation surface. It models workload traces, communication, topology bandwidth, and reconfiguration time; it does not require GPUs.

### Dependencies

On Ubuntu-like systems, install a C++17 toolchain, CMake, Protobuf development tools, and Graphviz. Python needs the packages used by the trace generator and plotting helpers:

```bash
sudo apt-get install build-essential cmake libprotobuf-dev protobuf-compiler graphviz python3-dev
python3 -m pip install --user \
  numpy sympy graphviz pandas tqdm pyyaml protobuf==5.28.2
```

### Build the two simulator backends

```bash
cd Opus
./simulation/scripts/build_backends.sh
```

The executables are written to:

```text
simulation/analytical_backend/build/astra_analytical/build/bin/
simulation/reconfig_backend/build/astra_analytical/build/bin/
```

To build one backend directly:

```bash
./simulation/analytical_backend/build/astra_analytical/build.sh
./simulation/reconfig_backend/build/astra_analytical/build.sh
```

If CMake stops at `find_package(Protobuf)`, the host is missing the C++ Protobuf development files. On Debian or Ubuntu, run:

```bash
sudo apt-get update
sudo apt-get install -y libprotobuf-dev protobuf-compiler
```

On a managed cluster, use the site module instead, for example `module avail protobuf` followed by `module load protobuf`. Verify that both `protoc --version` and the CMake package are available. If Protobuf is installed in a non-standard prefix, rerun CMake with `-DCMAKE_PREFIX_PATH=/path/to/protobuf`; the build needs headers, `libprotobuf`, and CMake metadata, not only the Python `protobuf` package. If a site uses a non-default compiler, set `CXX` and `CXXFLAGS` before invoking the build. Set `CMAKE_BUILD_PARALLEL_LEVEL` to limit memory use.

### Minimal simulation smoke test

```bash
cd Opus
./simulation/scripts/run_example.sh
```

This generates a small DP/PP/TP workload with the local `simulation/symbolic_tensor_graph` checkout, runs the analytical baseline and reconfigurable backend, and checks that non-empty debug output was produced. Generated workload files and debug logs are ignored by git.

### Paper Figure 12

Figure 12 is the Llama/H200-style scale-out study: DP=4, PP=4, TP=8, with reconfiguration-latency and scale-out-bandwidth sweeps.

```bash
cd Opus/simulation/scripts/fig12

# Latency sweep. Default values are 0, 10, 50, 100, 250, 500, 750, and 1000 ms.
./run_latency_exps.sh

# Bandwidth sweep. Default values are 12.5, 25, 50, 100, and 200 in the
# script’s scale-out bandwidth units.
./run_bw_exps.sh

# Plot from the generated center run directory.
./plot_fig12.sh
```

The plot script reads `simulation/reconfig_backend/examples/llama_dp4_pp4_tp8_batch_256_mb-1_96stack_seq4096_50BW` and writes `simulation/scripts/fig12/fig12.pdf`. The sweep scripts generate the workload traces and topology files before calling `examples/helpers/run_helper.py`, which rewrites `network.yml` with each reconfiguration time and records `results_for_sheet_import.txt`.

### Paper Figure 13

Figure 13 is the larger GB200/B200-style study: DP=4, PP=4, TP=32, with the GB200 trace generator and the same latency/bandwidth sweep pattern.

```bash
cd Opus/simulation/scripts/fig13
./run_latency_exps.sh
./run_bw_exps.sh
./plot_fig13.sh
```

The result is `simulation/scripts/fig13/fig13.pdf`. The generated center directory is `simulation/reconfig_backend/examples/gb200_stg_dp4_pp4_tp32_batch_256_mb-1_96stack_seq4096_100BW`.

### Paper Figure 14

Figure 14 compares the DP scale-out sweep for the H200-style and GB200-style configurations and feeds both into the cost/power plotter.

```bash
cd Opus/simulation/scripts/fig14
./run_H200_exps.sh
./run_GB200_exps.sh
./plot_fig14.sh
```

The output is `simulation/scripts/fig14/fig14.pdf`. This figure depends on generated run directories from both backends; it is not a hardware measurement.

### Other paper figures and archived artifacts

The artifact does not provide one universal command for every paper figure. Use the following pointers when documenting an evaluation run:

| Paper material | Artifact pointer | Reproduction status |
| --- | --- | --- |
| Emulation latency and provisioning results, including the configurations behind Figures 10 and 11 | `evaluation/llama-3-3d-16-latency`, `evaluation/llama-3-3d-64-latency`, `evaluation/deepseek_v3_16b-2d-16-latency`, `evaluation/deepseek_v3_16b-3d-16-latency` | Archived CSVs, communication patterns, and plotting notebooks; requires Slingshot/NCCL to regenerate raw runs |
| 128-GPU Llama emulation inputs | `evaluation/llama-3-70b-128` | Communication-pattern notebooks and rank logs are present; full training rerun requires the original GPU/model environment |
| Simulation scale-out sweeps, Figures 12 and 13 | `simulation/scripts/fig12`, `simulation/scripts/fig13` | Scripted after the tracked CMake/build-path fix above |
| DP sweep and cost/power plots, Figure 14 | `simulation/scripts/fig14` and `simulation/reconfig_backend/plot_combined_dp_cost_power.py` | Scripted; requires all generated DP directories |
| Hardware prototype and OCS link recovery | Paper Section 5.1 and the hardware/testbed environment files | Overview only; a Polatis OCS and compatible firmware are not included |
| Energy/cost topology plots | `evaluation/energy-analysis/topology/plot.ipynb` and the committed PDFs | Notebook plus archived rendered PDFs; use the notebook with its local data assumptions |
| Motivation and frontier/window studies, Figures 4–6 and 15–16 | simulator helpers, archived outputs, and paper captions | No single top-level reproduction command is packaged; report archived values unless the missing original sweep inputs are restored |

For notebook outputs, open the notebook in Jupyter, confirm that its CSV paths point inside `evaluation/`, and run all cells. The committed PDFs are useful as a reference for checking that a fresh plot has the same axes and trend; do not silently overwrite them during evaluation.

## Network emulation on a Slingshot cluster

### What is required

This path needs all of the following:

- multiple GPUs and the cluster’s Slingshot provider, with a working native NCCL/GPUDirect RDMA installation;
- a container runtime that exposes the host GPU, RDMA devices, and the provider libraries, or a site image that already supplies them;
- passwordless launch or the site equivalent for one controller process and all training ranks;
- an absolute, shared checkout path visible to the controller and training ranks;
- a communication-pattern output location writable by every rank;
- the model/configuration files used by the selected Torchtitan launcher.

A normal Ethernet-only workstation can run the controller smoke test and the software simulator, but it cannot reproduce the paper’s Slingshot emulation path.

### Build NCCL and the shim

Build NCCL for the CUDA and host compiler loaded on the target cluster, or point the shim at a site-provided NCCL build:

```bash
cd Opus
make -C nccl -j"${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}" src.build

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export NCCL_HOME="${NCCL_HOME:-${PWD}/nccl/build}"
python3 -m pip install --no-build-isolation -e src/opus-shim
make -C src/opus-controller
```

The shim now derives its repository root from `setup.py`. `NCCL_HOME` and `OPUS_TOML_INCLUDE` are the supported overrides. `NCCL_HOME` must contain `include/nccl.h` and a compatible `lib/libnccl.so` or equivalent library path. If the library is installed elsewhere, add its `lib` directory to `LD_LIBRARY_PATH`.

### Controller launch contract

Run exactly one controller on a node that all ranks can reach. The controller’s `-I` value must be an IPv4 address accepted by `inet_pton`; use `0.0.0.0` to listen on all interfaces or the node’s provider-reachable IPv4 address.

```bash
export OPUS_ROOT=/absolute/path/to/Opus
export CONTROLLER_IP=10.0.0.1
export OPUS_CONTROLLER_PORT=1234
export OPUS_IPC_DIR="${TMPDIR:-/tmp}/opus-${SLURM_JOB_ID:-manual}"
export OPUS_IPC_PREFIX="controller-${SLURM_JOB_ID:-manual}"
export OPUS_OUT="${OPUS_ROOT}/runs/${SLURM_JOB_ID:-manual}"
mkdir -p "${OPUS_IPC_DIR}" "${OPUS_OUT}"

NUM_NODES=4 \
NUM_RANKS_PER_NODE=16 \
SERVER_IPS=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 \
  "${OPUS_ROOT}/src/opus-controller/controller" \
  -I 0.0.0.0 -d 50 -e -r 4 -o "${OPUS_OUT}"
```

The command above starts the controller in emulation mode with four logical rails and a 50 ms delay. The controller starts the Python rail workers itself. It resolves `SERVER_IPS` for topology classification; use node hostnames if the site DNS is reliable, or use fixed IPv4 addresses. The Python worker reads `src/opus-controller/config/config.yaml` relative to itself, so the launch directory no longer matters. `OPUS_CONFIG_FILE` can override that path.

The controller’s TCP port is now shared through `OPUS_CONTROLLER_PORT`. Set the same value in the training-rank environment. `OPUS_IPC_DIR` and `OPUS_IPC_PREFIX` are local controller/worker coordination paths; using a job-specific prefix avoids collisions between simultaneous jobs on one node.

### Training-rank environment

The exact Slurm command is site-specific, but the required process-group variables are:

```bash
export CONTROLLER_IP=10.0.0.1
export OPUS_CONTROLLER_PORT=1234
export CONFIG_FILE="${OPUS_ROOT}/src/opus-controller/config/config.yaml"
export COMM_PATTERN_PATH="${OPUS_OUT}/comm_pattern"
export MODE=baseline              # baseline, provision, or no-ctl
export IS_EMULATION=0              # controller -e supplies the delay once
export NUM_NODES=4
export NUM_RANKS_PER_NODE=16
```

Use `MODE=baseline` for the controller-aware baseline and `MODE=provision` for the Opus provisioning path. `MODE=no-ctl` bypasses the controller and is useful as a native comparison, but it is not the Opus result. `COMM_PATTERN_PATH` must be writable and must match the path convention expected by the selected Torchtitan launcher. Set `IS_EMULATION=1` only when the experiment intentionally uses the shim-local delay hook; then set `RECONFIG_LATENCY` in milliseconds.

A portable Slurm shape is:

```bash
# Allocate first, then substitute the site’s container command for srun if needed.
# Start the controller only on the first allocated node.
srun --nodes=1 --ntasks=1 --nodelist="${CONTROLLER_NODE}" \
  bash -lc "${OPUS_ROOT}/src/opus-controller/controller -I 0.0.0.0 -d 50 -e -r 4 -o ${OPUS_OUT}" &

# Launch the training program on all ranks. Keep the controller variables in
# the exported environment and use the site’s GPU binding options.
srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 \
  bash -lc "srun --ntasks-per-node=${NUM_RANKS_PER_NODE} python -m torchtitan.examples..."
```

The final training module and configuration are intentionally shown as placeholders because they depend on the model and Slurm setup. The checked-in launchers under `torchtitan/opus-test/` are useful references, but their partitions, hostnames, account names, container paths, and GPU counts are not portable defaults. Copy the environment contract above into the site-specific launcher rather than copying its `#SBATCH` header unchanged.

### Docker image for emulation

Build from the repository root so `COPY . /Opus` includes `src`, `nccl`, and `torchtitan`:

```bash
cd Opus
docker build -f environment/emulation-env/Dockerfile -t opus-emulation:artifact .
```

On Slurm, replace Docker with the site-approved runtime such as Enroot, Apptainer, Shifter, or Charliecloud. Ensure that the runtime exposes `/dev/nvidia*`, the host RDMA devices, the Slingshot provider libraries, and the controller checkout. The image alone does not supply a virtual OCS.

## Hardware overview only

The hardware result uses a small testbed with four GPU servers and a Polatis Series 6000 optical circuit switch. The controller maps logical topology requests to Polatis configurations through the optional PyPolatis integration, while the data path depends on compatible NIC firmware and link bring-up behavior. The paper reports that measured OCS/link recovery dominates the prototype delay and discusses a firmware-supported lower bound.

This artifact does not include the missing hardware package path or a Polatis switch. Do not claim the hardware figure is reproducible from the emulation Dockerfile. For artifact evaluation, describe this section as an implementation overview and evaluate the software simulation plus controller emulation on available infrastructure.

## Known limitations and honest reporting

- The emulation is control-plane delay injection over the cluster’s native network, not a complete optical network emulator.
- Exact emulation figures depend on Slingshot firmware, NCCL/provider versions, GPU model, rank placement, model files, and site launch settings.
- The checked-in `evaluation/` data is valuable for plotting and inspection, but some directories contain communication patterns or notebooks without every raw training log needed for a clean rerun.
- The original hardware integration is not a portable artifact dependency. Physical mode still requires a separately installed `pypolatis` package and a reachable compatible switch.
- Software-simulation sweeps can generate large trace and debug files. Keep generated output outside the source tree when possible by setting `STG_DIR` and preserving the committed scripts.
- The simulator scripts mutate generated `network.yml` files to set `reconfig_time`. Run sweeps in generated experiment directories and retain their results files for audit.
- Results should be reported with the actual hardware, CUDA, NCCL, provider, compiler, and container versions. A trend match is meaningful even when absolute step times differ.

## Suggested artifact-evaluation order

1. Run `make -C src/opus-controller` and the finite Unix-socket smoke test.
2. Build both simulator backends and run `simulation/scripts/run_example.sh`.
3. Generate one small Figure 12 latency point by setting a narrow `SCALE_OUT_SWEEPS` list before launching the full sweep.
4. Inspect the archived emulation CSVs and notebooks under `evaluation/`.
5. If a Slingshot allocation is available, build NCCL and the shim, run one controller-aware configuration, then expand to the paper sweep.
6. Treat the hardware section as an overview unless a compatible Polatis testbed is available.

The SIGCOMM artifact process asks for a documented, retrievable artifact and distinguishes available from functional evaluation. This README identifies the required hardware and the non-reproducible hardware boundary explicitly; see the [SIGCOMM 2026 artifact call](https://conferences.sigcomm.org/sigcomm/2026/artifacts/) when preparing the submission package.
