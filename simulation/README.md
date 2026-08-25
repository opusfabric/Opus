# Opus software-simulation artifact

The root [`README.md`](../README.md) is the authoritative artifact-evaluation guide. This file keeps the simulator-specific command summary close to the code.

## Dependencies

On Ubuntu-like systems:

```bash
sudo apt-get install build-essential cmake libprotobuf-dev protobuf-compiler graphviz python3-dev
python3 -m pip install --user numpy sympy graphviz pandas tqdm pyyaml protobuf==5.28.2
```

The simulator sources and the symbolic tensor graph generator are vendored in this artifact. No git submodule checkout or author-local branch is required.

## Build and smoke test

Run from the repository root:

```bash
./simulation/scripts/build_backends.sh
./simulation/scripts/run_example.sh
```

The tracked CMake drivers under `analytical_backend/build/astra_analytical/` and `reconfig_backend/build/astra_analytical/` replace the generated build directories expected by older upstream ASTRA-sim scripts.

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
