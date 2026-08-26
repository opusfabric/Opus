# Opus artifact

Opus is a photonic-rail control plane for distributed ML communication. This artifact contains the source used for the paper, archived emulation inputs and outputs, and the analytical software simulator.

This README is the artifact-evaluation guide. It deliberately separates the three evaluation surfaces:

| Surface | What is provided | What a reader needs |
| --- | --- | --- |
| Software simulation | CPU/C++ analytical and reconfigurable ASTRA-sim backends, workload generators, sweep and plotting scripts | Linux build host or the simulator container, Python packages, and enough disk for generated traces |
| Network emulation | Controller, NCCL shim, Slingshot/NCCL launch scripts, communication-pattern inputs, and archived CSV/notebook outputs | A multi-GPU cluster with the site’s Slingshot provider and NCCL/GPUDirect RDMA support |
| Hardware prototype | Controller-side hardware integration and paper description | The Polatis OCS testbed and firmware; this is an overview only and is not expected to run on a reader’s cluster |

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

## 1. Small experiments on two GPUs for the shim

Use this path to validate the PyTorch/NCCL process-group extension on a two-GPU host before requesting a multi-node allocation. It exercises the `cuda:opus` backend and one controller-mediated emulated reconfiguration; it does not require Slingshot or a physical optical switch. The recommended path is the common Docker image described below. It already contains CUDA-enabled PyTorch, CUDA development tools, NCCL build support, and the simulator dependencies. A host/module environment also works if it provides the same CUDA-enabled PyTorch and NCCL prerequisites.

### Build and enter the common Docker image

Run these commands from the Opus repository root on a host with the NVIDIA Container Toolkit:

```bash
docker build --network=host -f environment/emulation-env/Dockerfile -t opus-emulation:artifact .
docker run --rm -it --gpus 2 --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/Opus" -w /Opus opus-emulation:artifact bash
```

The remaining commands in Sections 1 and 2 are run inside this container. The bind mount keeps source changes and generated results in the checkout. For a host/module run, omit the `docker run` step and execute the same commands from the repository root.

### Controller-only preflight (no GPU)

Before allocating GPUs, verify the emulation-delay worker and its Unix-socket protocol:

```bash
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

Expected responses contain `SUCCESS, CONFIG-ACK`. This tests only the controller-side delay model; it does not exercise NCCL or a network dataplane.

### Build the controller and shim

```bash
make -B -C src/opus-controller

# Use a site-provided NCCL build, or build the vendored NCCL source:
make -C nccl -j"${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}" src.build
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export NCCL_HOME="${NCCL_HOME:-${PWD}/nccl/build}"

# Verify that this interpreter sees CUDA-enabled PyTorch before pip evaluates setup.py.
python -c "import torch; print(\"PyTorch\", torch.__version__, \"CUDA available:\", torch.cuda.is_available())"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation -e src/opus-shim
```

If `import torch` fails, the active interpreter is the CPU-only simulation environment; load the site CUDA/PyTorch module or install the site-approved CUDA-enabled PyTorch wheel first. If pip reports `Cannot import setuptools.build_meta`, bootstrap the packaging tools inside the active environment with `python -m pip install --upgrade pip setuptools wheel`, then rerun the editable install.
If you changed C++ shim sources or still see an old shim error, force an in-place rebuild before rerunning the test:

```bash
bash src/opus-shim/build.sh
python -c 'import opus; print("Loaded shim:", opus.__file__)'
```

The vendored NCCL source includes `nccl/src/include/plugin/env/env_v1.h`, required by `nccl/src/include/plugin/nccl_env.h`; keep this header when copying or syncing the artifact checkout.
The shim also needs the header-only `toml++` library. Its headers are included at `third_party/tomlplusplus/include` and `setup.py` uses that path by default; set `OPUS_TOML_INCLUDE` if your site provides toml++ elsewhere.

`NCCL_HOME` must contain `include/nccl.h` and a compatible NCCL library. If PyTorch, CUDA, or NCCL is supplied by a module or container, load those modules first and set `CUDA_HOME`/`NCCL_HOME` accordingly.

### Run the two-GPU DP reconfiguration demo

The following is a single-host demonstration of the scale-out control path. Since both ranks are on one host, the controller would normally classify them as scale-up and bypass OCS provisioning. The environment variable OPUS_FORCE_DOMAIN=scale-out explicitly selects the scale-out branch for this local demonstration. Do not use that override on a real multi-node run; with multiple nodes, use the actual node addresses in SERVER_IPS and let the controller classify the group.

The -t 1 option seeds one active logical topology bit. The first DP stage then requests topology 0, which clears that bit. With -e -d 50, the Python worker sleeps for 50 ms for the changed bit and returns an acknowledgement. This makes the runtime request an observable emulated reconfiguration rather than a no-op transition from topology 0 to topology 0.

```bash
export CONTROLLER_IP=127.0.0.1
export OPUS_CONTROLLER_PORT=1234
export SERVER_IPS=127.0.0.1
export NUM_NODES=1
export NUM_RANKS_PER_NODE=2
export CONFIG_FILE="${PWD}/src/opus-shim/test/dp_2gpu.toml"
export COMM_PATTERN_PATH="${PWD}/src/opus-shim/test/dp_reconfig_pattern"
export MODE=baseline
export IS_EMULATION=0
export OPUS_FORCE_DOMAIN=scale-out

RUN_DIR="$(mktemp -d)"
export OPUS_IPC_DIR="${RUN_DIR}/ipc"
export OPUS_IPC_PREFIX=opus_dp_reconfig
mkdir -p "${OPUS_IPC_DIR}"

src/opus-controller/controller -I 127.0.0.1 -d 50 -e -r 1 -t 1 -o "${RUN_DIR}" \
  >"${RUN_DIR}/controller.log" 2>&1 &
CONTROLLER_PID=$!
cleanup() {
  kill "${CONTROLLER_PID}" 2>/dev/null || true
  wait "${CONTROLLER_PID}" 2>/dev/null || true
}
trap cleanup EXIT
sleep 1
if ! kill -0 "${CONTROLLER_PID}" 2>/dev/null; then
  echo "Controller exited before accepting connections:" >&2
  cat "${RUN_DIR}/controller.log" >&2
  exit 1
fi

cd src/opus-shim/test
torchrun --standalone --nproc_per_node=2 dp_reconfig.py
```

Inspect the two logs after the run:

```bash
grep -E "CONFIG_REQ|ALL READY|CONFIG_ACK|topoId|OPUS_FORCE_DOMAIN" "${RUN_DIR}/controller.log"
cat "${RUN_DIR}/python_output_rail_0.log"
```

A successful run shows both ranks checking in, a CONFIG_REQ from each rank for the first all-reduce, cnt/size: 2/2, a topology transition from 1 to 0, and CONFIG_ACK delivery. The first all-reduce should include approximately the configured 50 ms control-plane delay; later all-reduces use the already-selected topology and should not trigger another transition.

What is happening:

1. Each rank creates a two-member cuda:opus NCCL communicator and checks in with the controller.
2. The controller classifies the communicator as scale-out because of the explicit local-demo override.
3. The shim reads the checked-in DP communication pattern. At the first all-reduce boundary, both ranks send CONFIG_REQ.
4. The controller waits until both ranks arrive, sends logical topology 0 to the emulation worker, and waits for its 50 ms changed-bit delay.
5. The controller broadcasts CONFIG_ACK; the shim releases the pending collective and NCCL performs the all-reduce on the host’s ordinary network.
6. The three all-reduces verify the results 3, 5, and 7. Only the first one is intended to exercise reconfiguration.

This is OCS control-plane emulation, not a physical optical switch and not a packet-level optical-network emulator: the GPU data movement still uses NCCL and the host’s native network. On a real multi-node cluster, remove OPUS_FORCE_DOMAIN, set SERVER_IPS to every allocated node in rank order, and retain -e on the controller to model the OCS reconfiguration delay. MODE=provision is the paper’s provisioning path for communication patterns with look-ahead stages; this minimal DP example uses MODE=baseline so the first boundary directly issues a visible topology write.

For a two-node test, replace --standalone with the site-specific --nnodes, --node_rank, --master_addr, and --master_port arguments, set SERVER_IPS to the allocated node addresses, and start the controller on a reachable node. The two ranks in this example are intentionally local; a multi-node run should use the site’s GPU launcher and provider/NCCL settings.

## 2. GPU-cluster emulation with the whole control plane

The Opus controller and shim run on a real multi-GPU cluster, while NCCL continues to use the cluster's native packet-switched network. The control plane injects the configured OCS reconfiguration delay.

### What actually emulates OCS?

There are two controller modes. With -e, the worker logically delays each changed topology bit; without -e, the worker takes the physical-switch path and requires the optional PyPolatis installation and reachable hardware. The Dockerfile provides the software environment only; it does not create an optical switch or a packet-level OCS network.

The emulated OCS behavior is implemented by this chain:

1. `src/opus-controller/controller.cpp` starts one `config.py` worker per rail and serves the TCP control socket used by the shim.
2. `src/opus-controller/config.py`, when invoked with `-e`, represents a topology ID as 64 logical bits. For every changed bit it starts a worker that sleeps for the configured delay and then updates that bit. Changed bits are processed concurrently, so a transition is approximately one configured delay, not one delay per bit. The worker sends `SUCCESS, CONFIG-ACK` after all changed-bit workers finish.
3. `src/opus-shim/src/opus.cpp` requests topology changes from the controller while the actual NCCL collectives continue to use the host cluster’s normal network provider. In the paper-style setup, start the controller with `-e` and set `IS_EMULATION=0` in the shim. This applies the delay in the controller once.
4. `IS_EMULATION=1` in the shim is a separate host-side delay hook used by the shim’s provisioning path. Do not enable both hooks unless you intentionally want to model two delays.

### What is required

This path needs all of the following:

- multiple GPUs and a working native NCCL/provider installation; Slingshot/GPUDirect RDMA is required for the paper’s cluster results, while Ethernet can validate the control path only;
- a container runtime that exposes the host GPU, RDMA devices, and the provider libraries, or a site image that already supplies them;
- passwordless launch or the site equivalent for one controller process and all training ranks;
- an absolute, shared checkout path visible to the controller and training ranks;
- a communication-pattern output location writable by every rank;
- the model/configuration files used by the selected Torchtitan launcher.

A normal Ethernet-only workstation can run the Section 1 controller/DP demo and the software simulator, but it cannot reproduce the paper’s Slingshot performance results.

### Build NCCL and the shim

Build NCCL for the CUDA and host compiler loaded on the target cluster, or point the shim at a site-provided NCCL build:

```bash
make -C nccl -j"${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}" src.build

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export NCCL_HOME="${NCCL_HOME:-${PWD}/nccl/build}"
python -c "import torch; print(\"PyTorch\", torch.__version__, \"CUDA available:\", torch.cuda.is_available())"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation -e src/opus-shim
make -B -C src/opus-controller
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

Use `MODE=baseline` for the controller-aware baseline and `MODE=provision` for the Opus provisioning path. `MODE=no-ctl` skips topology-provisioning requests after the initial controller check-in and is useful as a native comparison, but it is not the Opus result. `COMM_PATTERN_PATH` must be writable and must match the path convention expected by the selected Torchtitan launcher. Set `IS_EMULATION=1` only when the experiment intentionally uses the shim-local delay hook; then set `RECONFIG_LATENCY` in milliseconds.

A portable Slurm shape is:

!TODO: can you provide sbatch script, with controller and job launcher in the same file

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

### Reuse the common Docker image for cluster emulation

The image built in Section 1 can also run the full multi-node control-plane path. On Slurm, replace Docker with the site-approved runtime such as Enroot, Apptainer, Shifter, or Charliecloud. Ensure that the runtime exposes `/dev/nvidia*`, the host RDMA devices, the Slingshot provider libraries, and the controller checkout. The image alone does not supply a virtual OCS.

### Hardware prototype overview (not runnable)

The hardware result uses a small testbed with four GPU servers and a Polatis Series 6000 optical circuit switch. The controller maps logical topology requests to Polatis configurations through the optional PyPolatis integration, while the data path depends on compatible NIC firmware and link bring-up behavior. The paper reports that measured OCS/link recovery dominates the prototype delay and discusses a firmware-supported lower bound.

This artifact does not include the missing hardware package path or a Polatis switch. Do not claim the hardware figure is reproducible from the emulation Dockerfile. For artifact evaluation, describe this section as an implementation overview and evaluate the software simulation plus controller emulation on available infrastructure.

## 3. Software simulation

The simulator is the most portable evaluation surface. It models workload traces, communication, topology bandwidth, and reconfiguration time; it does not require GPUs. The common Docker image also contains the C++/Protobuf/Graphviz toolchain and Python packages needed to run it without installing dependencies on the host.

### Run the simulator in Docker (recommended)

From the Opus repository root:

```bash
docker build --network=host -f environment/emulation-env/Dockerfile -t opus-emulation:artifact .
docker run --rm -it --network host \
  -v "$PWD:/Opus" -w /Opus opus-emulation:artifact bash
```

Inside the container:

```bash
./simulation/scripts/build_backends.sh
./simulation/scripts/run_example.sh
```

The simulator run is CPU-only; `--gpus` is not needed. The bind mount preserves generated experiment directories and logs in the checkout.

### Host fallback dependencies

On Ubuntu-like systems, install a C++17 toolchain, CMake, Protobuf development tools, and Graphviz. Python needs the packages used by the trace generator and plotting helpers:

```bash
sudo apt-get install build-essential cmake libprotobuf-dev protobuf-compiler graphviz python3-dev python3-venv

# Run from the Opus repository root.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  numpy sympy graphviz pandas tqdm pyyaml protobuf==5.28.2
```

Activate this `.venv` in every shell before running `build_backends.sh`, `run_example.sh`, the expert-parallel launcher, or the paper sweep scripts. The environment is local to the checkout and is ignored by git. If activation is not convenient, use the interpreter explicitly, for example `PYTHON="$PWD/.venv/bin/python" ./simulation/scripts/run_example.sh`.

### Build the two simulator backends

```bash
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

The Chakra bindings required by `feeder_v3` are generated automatically from `simulation/*_backend/extern/graph_frontend/chakra/schema/protobuf/et_def.proto` into the ignored `build/chakra_proto/` directory by the tracked CMake drivers. Do not copy generated `et_def.pb.h` or `et_def.pb.cc` files into the source tree. If the header is missing, verify `protoc --version` and rerun the backend build.

### Minimal simulation smoke test

```bash
./simulation/scripts/run_example.sh
```

This generates a small DP/PP/TP workload with the local `simulation/symbolic_tensor_graph` checkout, runs the analytical baseline and reconfigurable backend, and checks that non-empty debug output was produced. Generated workload files and debug logs are ignored by git. The STG generator also requires the checked-in CSV and companion JSON sharding tables under `simulation/symbolic_tensor_graph/sharding_spreadsheets/`; those JSON files are runtime inputs and are intentionally tracked.

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

### Supplemental expert-parallel experiments

The artifact now includes the additional EP workload material from the development checkout:

- `simulation/expert_parallel/run_ep.sh` generates the small DP=2, TP=1, PP=1, EP=2 workload.
- `simulation/reconfig_backend/examples/stg_dp2_pp2_tp2_ep2_batch_256_mb-1_96stack_seq4096_50BW` contains the 16-rank EP traces and simulator metadata.
- `simulation/reconfig_backend/examples/stg_dp1_pp1_tp1_ep2_batch_256_mb-1_96stack_seq4096_50BW` and `stg_dp2_pp1_tp1_ep2_batch_256_mb-1_96stack_seq4096_50BW` contain smaller EP cases.

Run the launcher from the repository root after installing the Python STG dependencies:

```bash
./simulation/expert_parallel/run_ep.sh
```

The copied Chakra `workload*.et` files and Torchtitan profiler JSON traces are intentionally ignored by git as raw/generated data. The JSON configuration, topology schedules, launch scripts, and EP launcher remain available for inspection and rerunning.

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

## Known limitations and honest reporting

- The emulation is control-plane delay injection over the cluster’s native network, not a complete optical network emulator.
- Exact emulation figures depend on Slingshot firmware, NCCL/provider versions, GPU model, rank placement, model files, and site launch settings.
- The checked-in `evaluation/` data is valuable for plotting and inspection, but some directories contain communication patterns or notebooks without every raw training log needed for a clean rerun.
- The original hardware integration is not a portable artifact dependency. Physical mode still requires a separately installed `pypolatis` package and a reachable compatible switch.
- Software-simulation sweeps can generate large trace and debug files. Keep generated output outside the source tree when possible by setting `STG_DIR` and preserving the committed scripts.
- The simulator scripts mutate generated `network.yml` files to set `reconfig_time`. Run sweeps in generated experiment directories and retain their results files for audit.
- Results should be reported with the actual hardware, CUDA, NCCL, provider, compiler, and container versions. A trend match is meaningful even when absolute step times differ.

## Suggested artifact-evaluation order

1. Build the controller and run the finite Unix-socket smoke test.
2. If two CUDA GPUs are available, run the small shim test in Section 1.
3. If a Slingshot allocation is available, follow Section 2 with one controller-aware configuration before expanding to the paper sweep.
4. Build the software backends and run Section 3's `simulation/scripts/run_example.sh`.
5. Generate one small Figure 12 latency point by setting a narrow `SCALE_OUT_SWEEPS` list before launching the full sweep.
6. Treat the hardware section as an overview unless a compatible Polatis testbed is available.
