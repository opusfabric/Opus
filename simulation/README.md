# Opus software-simulation artifact

The root [`README.md`](../README.md) is the authoritative artifact-evaluation guide. This file keeps the simulator-specific command summary close to the code.

## Docker workflow (recommended)

The common image at [`environment/emulation-env/Dockerfile`](../environment/emulation-env/Dockerfile) includes the simulator C++ toolchain, Python dependencies, and the vendored STG inputs. From the Opus repository root:

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

The simulator is CPU-only, so GPU passthrough is not needed.

## Host fallback dependencies

On Ubuntu-like systems:

```bash
sudo apt-get install build-essential cmake libprotobuf-dev protobuf-compiler graphviz python3-dev python3-venv

# Run from the Opus repository root.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy sympy graphviz pandas tqdm pyyaml protobuf==5.28.2
```

Activate this `.venv` in every shell before running simulator commands or the toy workload generator. The environment is local to the checkout and is ignored by git.

The simulator sources and the symbolic tensor graph generator are vendored in this artifact. No git submodule checkout or author-local branch is required. The generator relies on the checked-in CSV and companion JSON sharding tables under `symbolic_tensor_graph/sharding_spreadsheets/`; the JSON files are runtime inputs, not disposable cache files.

## Build and smoke test

Run from the repository root:

```bash
./simulation/scripts/build_backends.sh
./simulation/scripts/run_example.sh
```

The tracked CMake drivers under `analytical_backend/build/astra_analytical/` and `reconfig_backend/build/astra_analytical/` replace the generated build directories expected by older upstream ASTRA-sim scripts.

The build drivers also generate Chakra's `et_def.pb.h` and `et_def.pb.cc` automatically under each backend's ignored `build/chakra_proto/` directory. These files come from `extern/graph_frontend/chakra/schema/protobuf/et_def.proto`; no generated source files need to be committed.

## Supplemental expert-parallel workloads

The copied EP launcher is:

```bash
./simulation/expert_parallel/run_ep.sh
```

It generates the DP=2, TP=1, PP=1, EP=2 workload under `simulation/reconfig_backend/examples/`. The additional EP directories include their simulator configuration and topology metadata. Raw Chakra `.et` files and Torchtitan profiler traces are present locally for convenience but are excluded by the repository ignore rules.

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

cd ../fig14
./run_H200_exps.sh
./run_GB200_exps.sh
./plot_fig14.sh
```

The sweep scripts generate traces under `simulation/reconfig_backend/examples/`, write per-run `results_for_sheet_import.txt`, and render `fig12.pdf`, `fig13.pdf`, or `fig14.pdf` beside the script. Narrow the sweep with the documented environment variables before running a full experiment.
