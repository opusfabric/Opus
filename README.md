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
├── scripts/                 portable Slurm and Perlmutter launchers
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

Use this to validate the `cuda:opus` process-group extension on two GPUs. It includes one controller-mediated reconfiguration and needs neither Slingshot nor a physical switch. The common Docker image is the recommended environment; a matching host or module environment also works.

### Build and enter the common Docker image

Run these commands from the Opus repository root on a host with the NVIDIA Container Toolkit:

```bash
docker build --network=host -f environment/emulation-env/Dockerfile -t opus-emulation:artifact .
docker run --rm -it --gpus 2 --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/Opus" -w /Opus opus-emulation:artifact bash
```

Run the Section 1 commands inside this container. Section 2 uses the cluster environment or the Perlmutter launcher below. The bind mount keeps source changes and generated results in the checkout; for a host/module run, omit the `docker run` step.

### Controller-only preflight (no GPU)

Run the self-contained smoke test before allocating GPUs:

```bash
python3 src/opus-controller/test_emulation.py
```

It starts `config.py` in emulation mode, sends four topology values through the Unix socket, checks for `SUCCESS, CONFIG-ACK`, and cleans up. This tests only the controller-side delay model; it does not exercise NCCL or a network dataplane.

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

The active Python must provide CUDA-enabled PyTorch. If pip cannot import `setuptools.build_meta`, run `python -m pip install --upgrade pip setuptools wheel` in the same environment. Rebuild changed C++ shim sources with `bash src/opus-shim/build.sh`. `NCCL_HOME` must contain `include/nccl.h`; the vendored `toml++` headers are under `third_party/tomlplusplus/include` and can be overridden with `OPUS_TOML_INCLUDE`.

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

The demo uses `MODE=baseline`, not `MODE=provision`. Both ranks send a request at the first all-reduce boundary; the controller waits for both, changes logical topology `1 -> 0`, delays for 50 ms, and acknowledges the ranks. The later all-reduces reuse the selected topology. NCCL still moves data over the host’s normal network.

## 2. GPU-cluster emulation with the whole control plane

This experiment runs Llama 3 8B for 10 steps on four GPU nodes
(16 GPUs): TP=4, PP=2, and DP shard degree=2. Opus emulates rail reconfiguration in the control plane.

### Emulation architecture

Each rank's shim reads its tracked communication pattern and asks the controller
for the topology required by the next DP or PP stage. The controller waits until
all ranks in that communicator are ready, sends the topology to the four rail
workers, and acknowledges the ranks. In emulation mode the workers model a
changed rail by sleeping for 50 ms when there is a topology reconfiguration event.

Provisioning means requesting the topology for the *next* communication group
while the current group's NCCL operation just finished. The controller and
rail workers perform a real CONFIG_REQ/CONFIG_ACK transaction. The physical
switch operation is emulated by the 50 ms rail-worker delay. If the provisioned reconfiguration finishes before the next group starts, that group does not pay the delay on its critical path.

### Setup

Run from the shared Opus checkout. Replace `your_project_g` with an active
Perlmutter GPU account.

```bash
export OPUS_ROOT="$PWD"
export OPUS_ACCOUNT=your_project_g
export OPUS_SHIFTER_IMAGE=ericd16/opus:2.0

shifter --image="${OPUS_SHIFTER_IMAGE}" bash -lc \
  'unset SSL_CERT_FILE REQUESTS_CA_BUNDLE
   cd '"${OPUS_ROOT}"'
   make -B -C src/opus-controller
   cd torchtitan
   python scripts/download_hf_assets.py \
     --repo_id hf-internal-testing/llama-tokenizer \
     --local_dir tests/assets --asset tokenizer'
```

The tracked workload configuration is
`torchtitan/opus-test/dp-2-pp-2-tp-4-pm-8b/llama3_8b_lb.toml`:

```toml
[training]
steps = 10
dataset = "c4_test"
enable_opus_backend = true

[parallelism]
data_parallel_shard_degree = 2
tensor_parallel_degree = 4
pipeline_parallel_degree = 2
```

### Profile and compile the communication schedule
Opus needs to capture the workload schedule for topology provisioning.
The schedule is derived from the workload. By default iterations 1, 2, and 3 are captured:

```bash
PROFILE_JOB=$(sbatch --parsable \
  --account="${OPUS_ACCOUNT}" --qos=debug --time=00:30:00 \
  --nodes=4 --gpus-per-node=4 --constraint=gpu \
  --export=ALL,OPUS_ROOT="${OPUS_ROOT}",NUM_RANKS_PER_NODE=4,OPUS_SHIFTER_IMAGE="${OPUS_SHIFTER_IMAGE}",OPUS_PROFILE_ITERATIONS=1:2:3 \
  scripts/perlmutter/run_opus_profile.sbatch)

# Run after PROFILE_JOB completes successfully.
python scripts/compile_opus_schedule.py \
  --profile-dir="runs/${PROFILE_JOB}/profile" \
  --output-prefix="runs/${PROFILE_JOB}/comm_pattern" \
  --mode=baseline --format=legacy
```

### Experiment A: EPS baseline, zero reconfiguration delay

This control run uses the same workload and on-demand path, but sets the
emulated switch delay to zero. It measures the training and Opus control-plane
overhead without a reconfiguration penalty.

```bash
EPS_JOB=$(sbatch --parsable \
  --account="${OPUS_ACCOUNT}" --qos=debug --time=00:30:00 \
  --nodes=4 --gpus-per-node=4 --constraint=gpu \
  --export=ALL,OPUS_ROOT="${OPUS_ROOT}",NUM_RANKS_PER_NODE=4,OPUS_SHIFTER_IMAGE="${OPUS_SHIFTER_IMAGE}",COMM_PATTERN_PATH="${OPUS_ROOT}/runs/${PROFILE_JOB}/comm_pattern",RECONFIG_DELAY_MS=0 \
  scripts/perlmutter/run_opus_emulation.sbatch)
```

### Experiment B: on-demand reconfiguration

The next communication stage requests its topology at the stage boundary and
waits for the 10 ms emulated reconfiguration.

```bash
ONDEMAND_JOB=$(sbatch --parsable \
  --account="${OPUS_ACCOUNT}" --qos=debug --time=00:30:00 \
  --nodes=4 --gpus-per-node=4 --constraint=gpu \
  --export=ALL,OPUS_ROOT="${OPUS_ROOT}",NUM_RANKS_PER_NODE=4,OPUS_SHIFTER_IMAGE="${OPUS_SHIFTER_IMAGE}",COMM_PATTERN_PATH="${OPUS_ROOT}/runs/${PROFILE_JOB}/comm_pattern",RECONFIG_DELAY_MS=10 \
  scripts/perlmutter/run_opus_emulation.sbatch)
```

### Experiment C: topology provisioning

Provisioning uses the identical workload and delay, but requests the next
topology early so the 10 ms delay can overlap with useful work.

```bash
PROVISION_JOB=$(sbatch --parsable \
  --account="${OPUS_ACCOUNT}" --qos=debug --time=00:30:00 \
  --nodes=4 --gpus-per-node=4 --constraint=gpu \
  --export=ALL,OPUS_ROOT="${OPUS_ROOT}",NUM_RANKS_PER_NODE=4,OPUS_SHIFTER_IMAGE="${OPUS_SHIFTER_IMAGE}",COMM_PATTERN_PATH="${OPUS_ROOT}/runs/${PROFILE_JOB}/comm_pattern",RECONFIG_DELAY_MS=10 \
  scripts/perlmutter/run_opus_provision.sbatch)
```

### Expected result

For all three jobs, the top-level job and worker step must finish with
`COMPLETED 0:0`, every worker log must reach step 10, and the controller log
must contain `CONFIG_REQ`, `ALL READY`, topology changes, and `CONFIG_ACK`.

```bash
sacct -j "${EPS_JOB},${ONDEMAND_JOB},${PROVISION_JOB}" \
  --format=JobID,JobName%24,State,ExitCode,Elapsed

grep -h "step: 10" "runs/${EPS_JOB}"/torchrun_*.log | head -1
grep -h "step: 10" "runs/${ONDEMAND_JOB}"/torchrun_*.log | head -1
grep -h "step: 10" "runs/${PROVISION_JOB}"/torchrun_*.log | head -1

grep -E "CONFIG_REQ|ALL READY|new topo: yes|CONFIG_ACK" \
  "runs/${ONDEMAND_JOB}/controller.log" | head
```

The critical expected difference is request timing:

| Mode | Delay | When the topology is requested | Expected effect |
| --- | --- | --- | --- |
| EPS baseline | 0 ms | At the communication-stage boundary | Reference time without switch latency |
| On demand | 10 ms | At the communication-stage boundary | The stage observes the reconfiguration delay |
| Provisioning | 10 ms | Before the boundary | The delay overlaps preceding work, reducing or hiding the boundary stall |

All three modes should complete the same 10-step workload and exercise the same
topologies. Provisioning is successful when its controller requests appear
earlier relative to the corresponding stage and training still reaches step 10;
wall-clock time may vary slightly between cluster runs.

Sample result:

| Result | EPS baseline | On demand | Provisioning |
| --- | --- | --- | --- |
| Training | Step 10 | Step 10 | Step 10 |
| Reconfiguration delay | 0 ms | 10 ms at boundary | 10 ms requested early |
| Expected iteration time | Lowest reference | Higher than EPS | Closer to EPS than on demand |

### Hardware prototype (original testbed)

We also evaluated Opus on a four-node, eight-GPU prototype connected through a
Polatis optical circuit switch. This was a real hardware run, not the emulation
used above: the controller launched one worker per optical rail, the worker
logged in to the switch through PyPolatis, installed the requested
cross-connects, and returned `CONFIG-ACK` before GPU communication continued.

The repository retains the exact
[hardware launch script](torchtitan/opus-test/dp-2-tp-2-pp-2-eval/test-6-7-8-9-8gpu.sh),
the [Polatis control worker](src/opus-controller/config.py), and the recorded
[cross-connect maps](src/opus-controller/config/config.yaml). On every testbed
node, the experiment was launched from the repository root with:

```bash
bash torchtitan/opus-test/dp-2-tp-2-pp-2-eval/test-6-7-8-9-8gpu.sh
```

The launcher sets `IS_EMULATION=0`, starts the Opus controller on the first
node, and runs two GPU processes on each of four nodes. Successful hardware
operation is visible in the controller/worker logs as switch product
information, `Applying <configuration>`, `Connections set`, and
`SUCCESS, CONFIG-ACK` messages. Re-running this command requires the original
Polatis switch, its site-approved PyPolatis package, the listed testbed hosts,
and compatible GPU/NIC firmware. Those external components are not distributed
with this artifact.

## 3. Software simulation

The simulator is CPU-only and models workload traces, communication, topology bandwidth, and reconfiguration time. The common Docker image includes its C++/Protobuf/Graphviz and Python dependencies.

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

If CMake stops at `find_package(Protobuf)`, install `libprotobuf-dev` and `protobuf-compiler` (or load the site Protobuf module). The build needs the C++ headers, library, and CMake metadata. Chakra generates `et_def.pb.h` and `et_def.pb.cc` under the ignored `build/chakra_proto/` directory; do not copy them into the source tree. Set `CMAKE_BUILD_PARALLEL_LEVEL` if memory is limited.

### Minimal simulation smoke test

```bash
./simulation/scripts/run_example.sh
```

This generates a small DP/PP/TP workload, runs both simulator backends, and checks for non-empty output. Generated traces and logs are ignored; the sharding JSON files under `simulation/symbolic_tensor_graph/sharding_spreadsheets/` are tracked runtime inputs.

### Paper Figure 12

Figure 12 is the Llama/H200-style scale-out study: DP=4, PP=4, TP=8, with reconfiguration-latency and scale-out-bandwidth sweeps.

```bash
cd simulation/scripts/fig12

# Latency sweep. Default values are 0, 10, 50, 100, 250, 500, 750, and 1000 ms.
./run_latency_exps.sh

# Bandwidth sweep. Default values are 12.5, 25, 50, 100, and 200 in the
# script’s scale-out bandwidth units.
./run_bw_exps.sh

# Plot from the generated center run directory.
./plot_fig12.sh
```

The scripts generate the workload and topology files, run the reconfigurable backend, and write `fig12.pdf`; the center run is under `simulation/reconfig_backend/examples/llama_dp4_pp4_tp8_batch_256_mb-1_96stack_seq4096_50BW`.

### Paper Figure 13

Figure 13 is the larger GB200/B200-style study: DP=4, PP=4, TP=32, with the GB200 trace generator and the same latency/bandwidth sweep pattern.

```bash
cd simulation/scripts/fig13
./run_latency_exps.sh
./run_bw_exps.sh
./plot_fig13.sh
```

The result is `simulation/scripts/fig13/fig13.pdf`. The generated center directory is `simulation/reconfig_backend/examples/gb200_stg_dp4_pp4_tp32_batch_256_mb-1_96stack_seq4096_100BW`.

### Paper Figure 14

Figure 14 compares the DP scale-out sweep for the H200-style and GB200-style configurations and feeds both into the cost/power plotter.

```bash
cd simulation/scripts/fig14
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

Raw `workload*.et` and profiler traces are ignored; the EP launcher, configurations, schedules, and metadata remain tracked.

### Other paper figures and archived artifacts

The artifact does not provide one universal command for every paper figure. Use the following pointers when documenting an evaluation run:

| Paper material | Artifact pointer | Reproduction status |
| --- | --- | --- |
| Emulation latency and provisioning results, including the configurations behind Figures 10 and 11 | `evaluation/llama-3-3d-16-latency`, `evaluation/llama-3-3d-64-latency`, `evaluation/deepseek_v3_16b-2d-16-latency`, `evaluation/deepseek_v3_16b-3d-16-latency` | Archived CSVs, communication patterns, and plotting notebooks; requires Slingshot/NCCL to regenerate raw runs |
| 128-GPU Llama emulation inputs | `evaluation/llama-3-70b-128` | Communication-pattern notebooks and rank logs are present; full training rerun requires the original GPU/model environment |
| Simulation scale-out sweeps, Figures 12 and 13 | `simulation/scripts/fig12`, `simulation/scripts/fig13` | Scripts and inputs are provided; rerun on a CPU build host |
| DP sweep and cost/power plots, Figure 14 | `simulation/scripts/fig14` and `simulation/reconfig_backend/plot_combined_dp_cost_power.py` | Scripted; requires all generated DP directories |
| Hardware prototype and OCS link recovery | Paper Section 5.1 and the hardware/testbed environment files | Overview only; a Polatis OCS and compatible firmware are not included |
| Energy/cost topology plots | `evaluation/energy-analysis/topology/plot.ipynb` and the committed PDFs | Notebook plus archived rendered PDFs; use the notebook with its local data assumptions |
| Motivation and frontier/window studies, Figures 4–6 and 15–16 | simulator helpers, archived outputs, and paper captions | No single top-level reproduction command is packaged; use archived values unless the original inputs are restored |

For notebook outputs, open the notebook in Jupyter, confirm that its CSV paths point inside `evaluation/`, and run all cells. The committed PDFs are useful as a reference for checking that a fresh plot has the same axes and trend; do not silently overwrite them during evaluation.

## Known limitations and honest reporting

- Emulation injects control-plane delay while NCCL uses the native network; it is not a packet-level optical-network emulator.
- Exact cluster results depend on Slingshot firmware, NCCL/provider versions, GPU model, rank placement, and launch settings.
- Some `evaluation/` directories contain archived patterns or notebooks rather than every raw log needed for a rerun.
- Physical mode requires the separate `pypolatis` package and a reachable compatible switch.
- Simulation sweeps generate output and rewrite `network.yml`; report the hardware, CUDA, NCCL, provider, compiler, and container versions with results.

## Suggested artifact-evaluation order

1. Run the controller preflight and, when available, the two-GPU shim test.
2. Use one Slurm controller-aware configuration before a full cluster sweep.
3. Build the simulator, run `simulation/scripts/run_example.sh`, and then a narrow Figure 12 sweep.
4. Treat hardware as an overview unless a compatible Polatis testbed is available.
