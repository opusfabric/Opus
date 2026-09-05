# Opus artifact

Opus is a photonic-rail control plane for communication traffic in hybrid-parallel distributed LLM training. This artifact contains the source used for the paper, the reconfigurable network software simulator, and experiment scripts.

There are four sections:

1. Small experiments on two GPUs for testing the shim
- Requirement: 2 NVIDIA GPUs and an NVIDIA driver compatible with CUDA 12.8. The docker container in this artifact provides CUDA 12.8, cuDNN 9, and PyTorch nightly `2.10.0.dev20251209` (`cu128`). No particular NVIDIA GPU model is required.

2. GPU-cluster emulation with the whole control plane
- Requirement: 16 NVIDIA GPUs (4 nodes × 4 GPUs), a Slurm environment, RDMA-capable networking, and a driver compatible with CUDA 12.8. The Perlmutter (the cluster we used for the evaluation) launcher uses Shifter and its site-specific image. The reader can use the equivalent CUDA 12.8-compatible image on another cluster. We have only tested with the Perlmutter environment.

3. Software simulation (CPU)
- Requirement: CPU only. We recommend having at least 8 GB RAM for the smoke test. 32 GB is recommended for normal sweeps, and 64 GB is recommended for the largest EP or strong-scaling trace-generation runs. No GPU is needed.

4. Paper figure replication
- Requirement: figure-dependent. Figures 4(d), 5, and 12–15 use the CPU simulator. Figures 4(a-c) and 10–11 need the original Slingshot/NCCL GPU environment. Figure 9 needs the Polatis OCS testbed (we cannot provide access unfortunately).

## Repository layout

```text
Opus/
├── environment/
│   ├── emulation-env/       CUDA/PyTorch/RDMA image for software emulation
│   └── testbed-env/         hardware/testbed image
├── evaluation/              archived communication patterns, CSVs, notebooks, PDFs
├── nccl/                    vendored NCCL source used to build the Opus shim
├── scripts/                 portable Slurm and Perlmutter launchers
├── simulation/
│   ├── reconfig_backend/    ASTRA-sim backend with reconfigurable topology support
│   ├── symbolic_tensor_graph/ workload/Chakra trace generator
│   └── scripts/             backend build, experiment sweeps, and figure plots
├── src/
│   ├── opus-controller/     C++ TCP controller and Python rail workers
│   └── opus-shim/           PyTorch/NCCL process-group extension
└── torchtitan/opus-test/    paper-oriented Slurm training launchers and configs
```

## 1. Small experiments on two GPUs for testing the shim

This section validate the `cuda:opus` process-group extension on two connected GPUs by NVLink/PCIe. It includes one controller-mediated "reconfiguration", a blocking event in the data plane, with no actual physical reconfigurations.

### Build and enter the common Docker image

Run these commands from the Opus repository root on a host with the NVIDIA Container Toolkit:

```bash
docker build --network=host -f environment/emulation-env/Dockerfile -t opus-emulation:artifact .
docker run --rm -it --gpus 2 --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/Opus" -w /Opus opus-emulation:artifact bash
```

Run the Section 1 commands inside this container.

### Controller-only preflight (no GPU)

Run the self-contained smoke test before allocating GPUs:

```bash
python3 src/opus-controller/test_emulation.py
```

It starts `config.py` in emulation mode, sends four topology values through the Unix socket, checks for `SUCCESS, CONFIG-ACK`, and cleans up. This tests the controller-side delay model.

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

### Run the two-GPU DP reconfiguration demo

The following is a single-host demonstration of the scale-out control path. Since both ranks are on one host, the controller would normally classify them as scale-up and bypass OCS provisioning. The environment variable OPUS_FORCE_DOMAIN=scale-out explicitly selects the scale-out branch for this local demonstration. Do not use that override on a real multi-node run. With multiple nodes, use the actual node addresses in SERVER_IPS and let the controller classify the group.

The -t 1 option seeds one active logical topology bit. The first DP stage then requests topology 0, which clears that bit. With -e -d 50, the Python worker sleeps for 50 ms for the changed bit and returns an acknowledgement. This makes the runtime request an observable emulated reconfiguration rather than a no-op transition from topology 1 to topology 0.

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

A successful run shows both ranks checking in, a CONFIG_REQ from each rank for the first all-reduce, cnt/size: 2/2, a topology transition from 1 to 0, and CONFIG_ACK delivery. The first all-reduce should include approximately the configured 50 ms control-plane delay. Later all-reduces use the already-selected topology and should not trigger another transition.

Both ranks send a request at the first all-reduce boundary. The controller waits for both, changes logical topology `1 -> 0`, delays for 50 ms, and acknowledges the ranks. The later all-reduces reuse the selected topology. NCCL still moves data over the host’s normal network.

## 2. GPU-cluster emulation with the whole control plane

This experiment runs Llama 3 8B for 10 steps on four GPU nodes
(16 GPUs): TP=4, PP=2, and DP shard degree=2. Opus emulates rail reconfiguration in the control plane across DP and PP commmunication groups.

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

### Cluster setup (Perlmutter/Shifter)

The commands below are the complete Perlmutter path: they use the common CUDA 12.8 image through Shifter, build the controller, and prepare the tokenizer. The source image is [`ericd16/opus`](https://hub.docker.com/r/ericd16/opus).

On a Docker-based Slurm cluster, build and push the same image from `environment/emulation-env/Dockerfile`, then replace each `shifter --image=... --module=gpu` invocation in the launcher with `docker run --rm --gpus all --network host --ipc host --device=/dev/infiniband --hostname "$SLURMD_NODENAME" -v "$OPUS_ROOT:/Opus" -w /Opus ...`. Keep the same `SERVER_IPS`, controller address, and Slurm experiment commands below.

Run from the shared Opus checkout. Replace `your_project_g` with an active
Perlmutter GPU account.

```bash
export OPUS_ROOT="$PWD"
export OPUS_ACCOUNT=your_project_g
export OPUS_SHIFTER_IMAGE=ericd16/opus:2.0  # source: https://hub.docker.com/r/ericd16/opus

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

### Track a communication pattern

When the workload or parallelism changes, capture a fresh communication trace
before updating the tracked pattern. The profile launcher writes raw per-rank
traces under `runs/${PROFILE_JOB}/profile`:

```bash
PROFILE_JOB=$(sbatch --parsable \
  --account="${OPUS_ACCOUNT}" --qos=debug --time=00:30:00 \
  --nodes=4 --gpus-per-node=4 --constraint=gpu \
  --export=ALL,OPUS_ROOT="${OPUS_ROOT}",NUM_RANKS_PER_NODE=4,OPUS_SHIFTER_IMAGE="${OPUS_SHIFTER_IMAGE}",OPUS_PROFILE_ITERATIONS=1:2:3 \
  scripts/perlmutter/run_opus_profile.sbatch)
```

After `PROFILE_JOB` completes successfully, compile the legacy-format files
directly into the example workload directory:

```bash
sacct -j "${PROFILE_JOB}" --format=JobID,State,Elapsed,ExitCode

python scripts/compile_opus_schedule.py \
  --profile-dir="runs/${PROFILE_JOB}/profile" \
  --output-prefix="${OPUS_ROOT}/torchtitan/opus-test/dp-2-pp-2-tp-4-pm-8b/comm_pattern/comm_pattern" \
  --mode=baseline --format=legacy
```

Review the generated files and commit them when the new pattern is intended.
The raw traces remain runtime artifacts under `runs/`.

### Tracked communication pattern

The example uses the captured communication pattern in
[`torchtitan/opus-test/dp-2-pp-2-tp-4-pm-8b/comm_pattern`](torchtitan/opus-test/dp-2-pp-2-tp-4-pm-8b/comm_pattern).
It contains the 16 per-rank legacy-format files required by the four-node,
four-GPU-per-node workload. `COMM_PATTERN_PATH` is the filename prefix, so
the value for this example is:

```bash
export COMM_PATTERN_PATH="${OPUS_ROOT}/torchtitan/opus-test/dp-2-pp-2-tp-4-pm-8b/comm_pattern/comm_pattern"
```

### Experiment A: EPS baseline

This control run uses the same workload and on-demand path, but sets the
emulated switch delay to zero.

```bash
EPS_JOB=$(sbatch --parsable \
  --account="${OPUS_ACCOUNT}" --qos=debug --time=00:30:00 \
  --nodes=4 --gpus-per-node=4 --constraint=gpu \
  --export=ALL,OPUS_ROOT="${OPUS_ROOT}",NUM_RANKS_PER_NODE=4,OPUS_SHIFTER_IMAGE="${OPUS_SHIFTER_IMAGE}",COMM_PATTERN_PATH="${OPUS_ROOT}/torchtitan/opus-test/dp-2-pp-2-tp-4-pm-8b/comm_pattern/comm_pattern",RECONFIG_DELAY_MS=0 \
  scripts/perlmutter/run_opus_emulation.sbatch)
```

For Docker or another non-Shifter environment, use the site-neutral launcher
instead:

```bash
EPS_JOB=$(sbatch --parsable \
  --nodes=4 --gpus-per-node=4 \
  --export=ALL,OPUS_ROOT="${OPUS_ROOT}",NUM_RANKS_PER_NODE=4,RECONFIG_DELAY_MS=0 \
  scripts/run_slurm_emulation.sbatch)
```


### Experiment B: on-demand reconfiguration

The next communication stage requests its topology at the stage boundary and
waits for the 10 ms emulated reconfiguration.

```bash
ONDEMAND_JOB=$(sbatch --parsable \
  --account="${OPUS_ACCOUNT}" --qos=debug --time=00:30:00 \
  --nodes=4 --gpus-per-node=4 --constraint=gpu \
  --export=ALL,OPUS_ROOT="${OPUS_ROOT}",NUM_RANKS_PER_NODE=4,OPUS_SHIFTER_IMAGE="${OPUS_SHIFTER_IMAGE}",COMM_PATTERN_PATH="${OPUS_ROOT}/torchtitan/opus-test/dp-2-pp-2-tp-4-pm-8b/comm_pattern/comm_pattern",RECONFIG_DELAY_MS=10 \
  scripts/perlmutter/run_opus_emulation.sbatch)
```

For Docker or another non-Shifter environment, use the site-neutral launcher
instead:

```bash
ONDEMAND_JOB=$(sbatch --parsable \
  --nodes=4 --gpus-per-node=4 \
  --export=ALL,OPUS_ROOT="${OPUS_ROOT}",NUM_RANKS_PER_NODE=4,RECONFIG_DELAY_MS=10 \
  scripts/run_slurm_emulation.sbatch)
```


### Experiment C: topology provisioning

Provisioning uses the identical workload and delay, but requests the next
topology early so the 10 ms delay can overlap with useful work.

```bash
PROVISION_JOB=$(sbatch --parsable \
  --account="${OPUS_ACCOUNT}" --qos=debug --time=00:30:00 \
  --nodes=4 --gpus-per-node=4 --constraint=gpu \
  --export=ALL,OPUS_ROOT="${OPUS_ROOT}",NUM_RANKS_PER_NODE=4,OPUS_SHIFTER_IMAGE="${OPUS_SHIFTER_IMAGE}",COMM_PATTERN_PATH="${OPUS_ROOT}/torchtitan/opus-test/dp-2-pp-2-tp-4-pm-8b/comm_pattern/comm_pattern",RECONFIG_DELAY_MS=10 \
  scripts/perlmutter/run_opus_provision.sbatch)
```

For Docker or another non-Shifter environment, use the site-neutral
provisioning launcher instead:

```bash
PROVISION_JOB=$(sbatch --parsable \
  --nodes=4 --gpus-per-node=4 \
  --export=ALL,OPUS_ROOT="${OPUS_ROOT}",NUM_RANKS_PER_NODE=4,RECONFIG_DELAY_MS=10 \
  scripts/run_slurm_provision.sbatch)
```


### Expected result

For all four jobs, the top-level job and worker step must finish with
`COMPLETED 0:0` and every worker log must reach step 10. For the three Opus
jobs, the controller log must also contain `CONFIG_REQ`, `ALL READY`, topology
changes, and `CONFIG_ACK`; the native reference intentionally has no
controller log.

```bash
sacct -j "${NATIVE_JOB},${EPS_JOB},${ONDEMAND_JOB},${PROVISION_JOB}" \
  --format=JobID,JobName%24,State,ExitCode,Elapsed

grep -h "step: 10" "runs/${NATIVE_JOB}"/torchrun_*.log | head -1
grep -h "step: 10" "runs/${EPS_JOB}"/torchrun_*.log | head -1
grep -h "step: 10" "runs/${ONDEMAND_JOB}"/torchrun_*.log | head -1
grep -h "step: 10" "runs/${PROVISION_JOB}"/torchrun_*.log | head -1

grep -E "CONFIG_REQ|ALL READY|new topo: yes|CONFIG_ACK" \
  "runs/${ONDEMAND_JOB}/controller.log" | head
```

The expected difference between two reconfiguration modes is request timing:

| Mode | Delay | When the topology is requested | Expected effect |
| --- | --- | --- | --- |
| On demand | 10 ms | At the communication-stage boundary | The stage observes the reconfiguration delay |
| Provisioning | 10 ms | Before the boundary | The delay overlaps preceding work, reducing or hiding the boundary stall |

Sample result:

Average iteration latency is measured from worker timestamps over the last five
iterations (steps 6–10) and averaged across the four node logs.

From the repository root, check run health and reproduce the latency summary
with:

```bash
scripts/check_run_output.sh "${NATIVE_JOB}" "${EPS_JOB}" "${ONDEMAND_JOB}" "${PROVISION_JOB}"
scripts/summarize_iteration_latency.py "${NATIVE_JOB}" "${EPS_JOB}" "${ONDEMAND_JOB}" "${PROVISION_JOB}"
```

The checker reports step completion, fatal errors, ACK timeouts, and topology
changes (or confirms that a native run has no Opus markers). The summarizer
prints one tab-separated row per job. Pass `--last N` to change the measurement
window.

| Experiment | Stack | Reconfiguration | Result | Last-five average |
| --- | --- | --- | --- | ---: |
| A | Opus EPS baseline | 0 ms | Step 10, `COMPLETED 0:0`; 0 ACK timeouts | **11.175 s** |
| B | Opus baseline | On demand, 10 ms | Step 10, `COMPLETED 0:0`; 0 ACK timeouts | **11.767 s** |
| C | Opus provisioning | Provisioned early, 10 ms | Step 10, `COMPLETED 0:0`; 0 ACK timeouts | **10.880 s** |

We can see that adding the
10 ms delay on demand makes B 5.3% slower than the zero-delay A baseline.
Provisioning hides that delay and is 7.5% faster than B.

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

The launcher starts the Opus controller on the first node and
runs two GPU processes on each of four nodes. Successful hardware
operation is visible in the controller/worker logs as switch product
information, `Applying <configuration>`, `Connections set`, and
`SUCCESS, CONFIG-ACK` messages. Re-running this command requires the original
Polatis switch, its site-approved PyPolatis package, the listed testbed hosts,
and compatible GPU/NIC firmware. We won't share those resources in this artifact as the testbed 
is being actively used.

## 3. Software simulation

The simulator is CPU-only and models workload traces, communication, topology bandwidth, and reconfiguration time. The common Docker image includes its C++/Protobuf/Graphviz and Python dependencies.

### Run the simulator in Docker (recommended)

From the Opus repository root:

```bash
docker build --network=host -f environment/emulation-env/Dockerfile -t opus-emulation:artifact .
docker run --rm -it --network host \
  --ulimit nofile=65536:65536 \
  -v "$PWD:/Opus" -w /Opus opus-emulation:artifact bash
```

Inside the container:

```bash
./simulation/scripts/build_backends.sh
./simulation/scripts/run_example.sh
```

The simulator run is CPU-only; `--gpus` is not needed. Keep the `nofile` setting because ASTRA-Sim opens one workload trace per simulated GPU. The bind mount preserves generated experiment directories and logs in the checkout.

### Host fallback dependencies

On Ubuntu-like systems, install a C++17 toolchain, CMake, Protobuf development tools, and Graphviz. Python needs the packages used by the trace generator and plotting helpers:

```bash
sudo apt-get install build-essential cmake libprotobuf-dev protobuf-compiler graphviz python3-dev python3-venv

# Run from the Opus repository root.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  numpy sympy graphviz pandas matplotlib seaborn tqdm pyyaml protobuf==5.28.2
```

Activate this `.venv` in every shell before running `build_backends.sh`, `run_example.sh`, the expert-parallel launcher, or the paper sweep scripts. The environment is local to the checkout and is ignored by git. If activation is not convenient, use the interpreter explicitly, for example `PYTHON="$PWD/.venv/bin/python" ./simulation/scripts/run_example.sh`.

### Build the reconfigurable simulator

```bash
./simulation/scripts/build_backends.sh
```

This builds only the reconfigurable ASTRA-Sim executable:

```text
simulation/reconfig_backend/build/astra_analytical/build/bin/AstraSim_Analytical_Reconfigurable
```

If CMake stops at `find_package(Protobuf)`, install `libprotobuf-dev` and
`protobuf-compiler` (or load the site Protobuf module). Chakra generates
`et_def.pb.h` and `et_def.pb.cc` under the ignored
`simulation/reconfig_backend/build/astra_analytical/build/chakra_proto/` directory. Set
`CMAKE_BUILD_PARALLEL_LEVEL` if memory is limited.

### Minimal simulation test

```bash
./simulation/scripts/run_example.sh
```

This generates a small DP/PP/TP workload and runs only the reconfigurable
backend. It sweeps 0, 1, 10, and 50 ms. The 0 ms run is the EPS reference:
the topology-change delay is zero. Positive values inject reconfiguration
delay.

- baseline: the next topology is requested at the communication boundary;
- provisioning: the next topology is requested early using
  `rank_comm_groups.yaml`.

The generated `schedules.txt` contains the fine-grained candidate topologies.
The output directory is
`simulation/reconfig_backend/examples/stg_dp2_pp2_tp2_batch_256_mb-1_96stack_seq4096_50BW/`.
It contains `debug_no_provision_<delay>ms.txt`, `debug_provision_<delay>ms.txt`
for positive delays, and the Markdown summary printed by the driver.

The small experiment uses DP=2, PP=2, TP=2 (8 simulated NPUs), batch 256,
sequence length 4096, 96 stacks, and scale-out/scale-up bandwidths 50/450
GB/s. The simulator is CPU-only; the Docker image does not need
`--gpus`.

Expected result:

| Reconfiguration | EPS (s) | Baseline (s) | Provisioning (s) |
| ---: | ---: | ---: | ---: |
| 1 ms | 62.2416 | 62.2436 | 62.2415 |
| 10 ms | 62.2416 | 62.2616 | 62.2505 |
| 50 ms | 62.2416 | 62.3416 | 62.2905 |


## 4. Paper figure replication


### Paper Figure 12
Figure 12 is the Llama/H200-style scale-out study: DP=4, PP=4, TP=8, with reconfiguration-latency and scale-out-bandwidth sweeps.

From the previous simulation Docker environment:

```bash
cd /Opus
./simulation/scripts/build_backends.sh
cd simulation/scripts/fig12

# Latency sweep. Default values are 0, 10, 50, 100, 250, 500, 750, and 1000 ms.
./run_latency_exps.sh

# Bandwidth sweep. Default values are 12.5, 25, 50, 100, and 200 in the
# script’s scale-out bandwidth units.
./run_bw_exps.sh

# Plot from the generated center run directory.
./plot_fig12.sh
```

The scripts generate the workload and topology files and write `fig12.pdf`. The center run is under `simulation/reconfig_backend/examples/llama_dp4_pp4_tp8_batch_256_mb-1_96stack_seq4096_50BW`. The bandwidth sweep invokes both backends: the analytical backend supplies the EPS and fixed-topology baseline columns, while the reconfigurable backend supplies the positive-delay Opus columns.

### Paper Figure 13

Figure 13 is the larger GB200/B200-style study: DP=4, PP=4, TP=32, with the GB200 trace generator and the same latency/bandwidth sweep pattern.

```bash
cd simulation/scripts/fig13
./run_latency_exps.sh
./run_bw_exps.sh
./plot_fig13.sh
```

The result is `simulation/scripts/fig13/fig13.pdf`. The generated center directory is `simulation/reconfig_backend/examples/gb200_stg_dp4_pp4_tp32_batch_256_mb-1_96stack_seq4096_100BW`. The latency and bandwidth sweeps use the analytical backend for EPS and the fixed-topology baseline. Positive-delay points use the reconfigurable backend.

### Paper Figure 14

Requirement: CPU only; 64 GB RAM is recommended for the complete 256/512-GPU sweeps.

From the repository root, build the simulator and regenerate all Figure 14 results:

```bash
./simulation/scripts/build_backends.sh
./simulation/expert_parallel/run_ep.sh all
```

Run only one system size if desired:

```bash
./simulation/expert_parallel/run_ep.sh 256
./simulation/expert_parallel/run_ep.sh 512
```

Results are written to the [256-GPU directory](simulation/reconfig_backend/examples/large_scale_ep_8gpu_256_all2allv_topk_context_reconfig_full_bw_eps/) and [512-GPU directory](simulation/reconfig_backend/examples/large_scale_ep_32gpu_512_all2allv_topk_context_reconfig_full_bw_eps/). Set `ASTRA_SIM_BIN_DIR` if the simulator executable was built elsewhere.

### Paper Figure 15

Figure 15 is the DP scale-out, performance, cost, and power study: DGX H200 with 400 Gbps links and B200 with 800 Gbps links.

```bash
cd simulation/scripts/fig15
./run_H200_exps.sh
./run_GB200_exps.sh
./plot_fig15.sh
```

The output is `simulation/scripts/fig15/fig15.pdf`. This output corresponds to paper Figure 15.

### Paper Figure 4

Figure 4 shows how communication reconfiguration affects computation windows.

#### Panels (a-c): trace analysis

Requirement: a 256-GPU Slurm allocation with CUDA-enabled PyTorch/TorchTitan
(the default launcher requests 32 nodes x 8 GPUs). The profiling run uses one experiment's configuration: DeepSeek-V2 236B, DP=8, PP=8, TP=4, EP=32, sequence
length 4096, and global batch 256. It initializes the model locally and records
real CUDA/NCCL activity.

Scripts and configuration:

- [GPU trace launcher](evaluation/deepseek-236b-256/run_gpu_trace_profile.sbatch)
- [DeepSeek-236B profiling configuration](evaluation/deepseek-236b-256/deepseek_236b_trace.toml)
- [Measured transition-window extractor](evaluation/deepseek-236b-256/trace_timing_gap.py)
- [Analytical transition-window model](evaluation/deepseek-236b-256/timing_gap_deepseek_236b_256gpu.py)
- [Communication-volume and time model](evaluation/deepseek-236b-256/comm_gap_deepseek_236b_256gpu.py)
- [Transition CDF plotter](evaluation/deepseek-236b-256/plot_transition_cdf.py)

Prepare the tokenizer once from the repository root. Only tokenizer metadata is
downloaded; the profiling job trains from random initialization:

```bash
cd torchtitan
python3 scripts/download_hf_assets.py \
  --repo_id deepseek-ai/DeepSeek-V2 --assets tokenizer
cd ..
```

Submit the end-to-end GPU profiling and analysis job:

```bash
sbatch evaluation/deepseek-236b-256/run_gpu_trace_profile.sbatch
```

Use normal site flags such as `--account`, `--partition`, or `--constraint` on
the `sbatch` command when required. For a cluster with four GPUs per node,
request 64 nodes and tell the launcher the GPU density:

```bash
sbatch --nodes=64 --gpus-per-node=4 \
  --export=ALL,NUM_RANKS_PER_NODE=4 \
  evaluation/deepseek-236b-256/run_gpu_trace_profile.sbatch
```

The launcher verifies that exactly 256 GPU ranks are present, runs five
training steps, and profiles the fifth step after the PyTorch profiler warmup.
All 256 Chrome traces remain under `runs/<job-id>/figure4/profile_trace/`. It
also copies `rank0_trace.json` through `rank3_trace.json` into
`evaluation/deepseek-236b-256/` and runs every panel (a-c) analysis script.
The rerun is complete when these outputs exist there:

- `trace_timing_gap.txt`: measured cross-parallelism windows from the traces
- `timing_gap_model.txt`: modeled computation windows between communication types
- `comm_gap_model.txt`: modeled DP, EP, and PP communication costs
- `transition_cdf_combined.png`: combined transition-window CDF
- `transition_cdf_per_pair.png`: one CDF panel per transition pair
- `transition_cdf_summary.txt`: CDF sample counts and percentiles

For the ASTRA-Sim strong-scaling inputs, run the following inside the Section 3 Docker container after building the simulator:

```bash
for dp in 16 32 64 128; do
  DP="${dp}" TP=32 PP=4 EP=1 BATCH=15360 NS=96 SEQ_LEN=4096 \
  SCALE_OUT_SWEEPS=50 MIXED_PRECISION=0 \
  ./simulation/reconfig_backend/run_stg_exp_fg_pp.sh

  case_dir="simulation/reconfig_backend/examples/stg_dp${dp}_pp4_tp32_batch_15360_mb-1_96stack_seq4096_50BW"
  python3 simulation/reconfig_backend/examples/helpers/run_helper.py "${case_dir}" \
    --reconfig-times 0 --output figure4d_results.txt
done
```

### Paper Figure 5

Figure 5 derives GPU utilization from the Figure 4(d) frontier windows using `Util = Tcompute / (Tcompute + Tnon_overlap_comm + Tstall)`, with `Tstall = sum_i max(0, Treconfig - Twindow_i)`. It evaluates OCS reconfiguration latencies of 0, 1, 5, 10, 50, 100, 250, 500, 750, and 1000 ms for the four frontier scales.

### Paper Figure 10/11

Requirement: Slurm with CUDA-enabled PyTorch, NCCL/RDMA, and 16 NVIDIA GPUs
(4 nodes × 4 GPUs) for the 16-GPU cases or 64 GPUs (16 nodes × 4 GPUs) for
Llama-64. No physical OCS is required: the controller runs in emulation mode
and applies the requested reconfiguration delay. Complete the Section 2 build
and tokenizer setup before submitting these jobs.

Scripts:

- [Figure 10/11 submission helper](scripts/submit_figure10_11.sh)
- [Generic emulation](scripts/run_slurm_emulation.sbatch)
- [Generic provisioning](scripts/run_slurm_provision.sbatch)
- [Perlmutter emulation](scripts/perlmutter/run_opus_emulation.sbatch)
- [Perlmutter provisioning](scripts/perlmutter/run_opus_provision.sbatch)
- [Output checker](scripts/check_run_output.sh)
- [Latency summarizer](scripts/summarize_iteration_latency.py)
- [Llama 3D, 16-GPU plot notebook](evaluation/llama-3-3d-16-latency/provision.ipynb)
- [Llama 3D, 64-GPU plot notebook](evaluation/llama-3-3d-64-latency/provision.ipynb)
- [DeepSeek 2D plot notebook](evaluation/deepseek_v3_16b-2d-16-latency/plot.ipynb)
- [DeepSeek 3D plot notebook](evaluation/deepseek_v3_16b-3d-16-latency/plot.ipynb)

The submission helper selects the matching TorchTitan configuration,
communication-pattern prefix, GPU count, and provisioning callback mode:

| Case argument | Workload and parallelism | Allocation | Archived pattern |
| --- | --- | --- | --- |
| `llama16` | Llama 3 8B; DP=2, PP=2, TP=4 | 4 nodes × 4 GPUs | [pattern](evaluation/llama-3-3d-16-latency/comm_pattern) |
| `llama64` | Llama 3 8B; DP=8, PP=2, TP=4 | 16 nodes × 4 GPUs | [pattern](evaluation/llama-3-3d-64-latency/comm_pattern_no_ctl_issue) |
| `deepseek2d` | DeepSeek 16B; PP=4, TP=4 | 4 nodes × 4 GPUs | [pattern](evaluation/deepseek_v3_16b-2d-16-latency/comm_pattern) |
| `deepseek3d` | DeepSeek 16B; DP=2, PP=2, TP=4 | 4 nodes × 4 GPUs | [pattern](evaluation/deepseek_v3_16b-3d-16-latency/comm_pattern) |

From the repository root, run one EPS, on-demand, and provisioning comparison.
This example uses the 16-GPU Llama case and a 50 ms OCS delay:

```bash
EPS_JOB=$(scripts/submit_figure10_11.sh \
  llama16 baseline 0 --account=YOUR_ACCOUNT)
ONDEMAND_JOB=$(scripts/submit_figure10_11.sh \
  llama16 baseline 50 --account=YOUR_ACCOUNT)
PROVISION_JOB=$(scripts/submit_figure10_11.sh \
  llama16 provision 50 --account=YOUR_ACCOUNT)

echo "EPS=${EPS_JOB} on-demand=${ONDEMAND_JOB} provision=${PROVISION_JOB}"
```

Replace `llama16` with any case argument from the table. For Perlmutter/Shifter,
select the container launcher and pass the normal site options:

```bash
OPUS_SLURM_LAUNCHER=perlmutter \
OPUS_SHIFTER_IMAGE=ericd16/opus:2.0 \
scripts/submit_figure10_11.sh \
  llama16 provision 50 --account=YOUR_ACCOUNT --constraint=gpu --qos=regular
```

To recreate the archived latency sweep for one case, submit both modes at each
delay. The zero-delay baseline is the EPS point:

```bash
for delay in 0 10 50 100 250 500 750 1000; do
  scripts/submit_figure10_11.sh \
    llama16 baseline "${delay}" --account=YOUR_ACCOUNT
  scripts/submit_figure10_11.sh \
    llama16 provision "${delay}" --account=YOUR_ACCOUNT
done
```

Each submission prints its job ID and writes controller and per-node worker logs
to `runs/<job-id>/`. After the three comparison jobs finish, verify completion,
ACK health, topology changes, and final-five-step latency:

```bash
EXPECTED_RANKS=16 scripts/check_run_output.sh \
  "${EPS_JOB}" "${ONDEMAND_JOB}" "${PROVISION_JOB}"

scripts/summarize_iteration_latency.py \
  "${EPS_JOB}" "${ONDEMAND_JOB}" "${PROVISION_JOB}" --last 5
```

Use `EXPECTED_RANKS=64` when checking `llama64`. The plot notebooks listed
above consume the corresponding latency summaries/CSV data.

### Other paper figures and archived artifacts


| Paper figure/material | Artifact pointer | Note |
| --- | --- | --- |
| Figure 9: hardware testbed and link-recovery timeline | [testbed environment README](environment/testbed-env/README.md), [testbed Dockerfile](environment/testbed-env/Dockerfile), [hardware launch script](torchtitan/opus-test/dp-2-tp-2-pp-2-eval/test-6-7-8-9-8gpu.sh), and [Polatis worker](src/opus-controller/config.py) | Overview only. Reproducing the measurements requires the Polatis OCS, compatible NIC/firmware, testbed hosts, and site-approved packages |
