# Opus software-simulation artifact

The root [`README.md`](../README.md) is the authoritative artifact-evaluation guide. This file keeps the simulator-specific command summary close to the code.

## Docker workflow (recommended)

The common image at [`environment/emulation-env/Dockerfile`](../environment/emulation-env/Dockerfile) includes the simulator C++ toolchain, Python dependencies, and the vendored STG inputs. From the Opus repository root:

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

The simulator is CPU-only, so GPU passthrough is not needed. Keep the `nofile` setting because ASTRA-Sim opens one workload trace per simulated GPU. Allow at least 8 GB RAM for `run_example.sh`; 32 GB is recommended for normal sweeps, and 64 GB for the largest EP or strong-scaling trace-generation runs.

## Host fallback dependencies

On Ubuntu-like systems:

```bash
sudo apt-get install build-essential cmake libprotobuf-dev protobuf-compiler graphviz python3-dev python3-venv

# Run from the Opus repository root.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy sympy graphviz pandas matplotlib seaborn tqdm pyyaml protobuf==5.28.2
```

Activate this `.venv` in every shell before running simulator commands or the toy workload generator. The environment is local to the checkout and is ignored by git.

The simulator sources and the symbolic tensor graph generator are vendored in this artifact. No git submodule checkout or author-local branch is required. The generator relies on the checked-in CSV and companion JSON sharding tables under `symbolic_tensor_graph/sharding_spreadsheets/`; the JSON files are runtime inputs, not disposable cache files.

## Build and smoke test

Run from the repository root:

```bash
./simulation/scripts/build_backends.sh
./simulation/scripts/run_example.sh
```

The reconfigurable CMake driver under `reconfig_backend/build/astra_analytical/` builds the simulator used by this artifact.

The build drivers also generate Chakra's `et_def.pb.h` and `et_def.pb.cc` automatically under `reconfig_backend/build/astra_analytical/build/chakra_proto/`. These files come from `extern/graph_frontend/chakra/schema/protobuf/et_def.proto`; no generated source files need to be committed.

## Paper Figure 14: expert-parallel workloads

The copied EP launcher is:

```bash
./simulation/expert_parallel/run_ep.sh
```

It reconstructs the published 256/512-GPU Figure 14 sweep as CPU-only simulated NPUs. By default it covers EP+TP+DP, EP+DP, all paper EP degrees, and 0/0.5/1/10/50/100-ms reconfiguration delays. Use `CLUSTER_SIZES`, `PLACEMENTS`, `EP_SIZES_*`, `EP_LAYERS`, and `LATENCIES_MS` to make a smoke run. Raw Chakra `.et` files and simulator logs are regenerated and excluded by the repository ignore rules; the original paper raw EP traces are not included.

## Paper sweep commands

```bash
cd simulation/scripts/fig12
./run_latency_exps.sh
./run_bw_exps.sh
./plot_fig12.sh

cd ../fig13
./run_latency_exps.sh
./run_bw_exps.sh
./plot_fig13.sh

cd ../fig15
./run_H200_exps.sh
./run_GB200_exps.sh
./plot_fig15.sh
```

The sweep scripts generate traces under `simulation/reconfig_backend/examples/` and render `fig12.pdf`, `fig13.pdf`, or `fig15.pdf` beside the script. Figures 12 and 13 use the analytical backend for EPS/fixed-topology baselines and keep latency and bandwidth results in separate purpose-specific files so that rerunning one sweep does not overwrite the other. The `fig15` directory/output is the implementation for paper Figure 15; paper Figure 14 is the EP experiment above. Narrow the sweep with the documented environment variables before running a full experiment.
