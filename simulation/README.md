### Opus Simulation Artifacts

### Setup

1. Build and launch docker container
```
cd simulation/analytical_backend

git submodule update --init --recursive

docker build -t astra-sim:latest -f Dockerfile .

cd ../../

docker run -it --shm-size=8g -v $(pwd):/workspace astra-sim:latest bash

cd /workspace/simulation
```

<!-- 1. Install ASTRA-sim Dependencies
```
apt -y update
apt -y install coreutils wget vim git
apt -y install gcc-11 g++-11 make cmake 
apt -y install clang-format 
apt -y install libboost-dev libboost-program-options-dev
apt -y install python3.10 python3-pip
apt -y install libprotobuf-dev protobuf-compiler
apt -y install openmpi-bin openmpi-doc libopenmpi-dev
``` -->

```
pip3 install --upgrade pip
pip3 install protobuf==5.28.2
pip3 install graphviz pydot
```

2. Install Symbolic Tensor Graph Depedencies
```
pip install numpy sympy graphviz protobuf pandas tqdm
```

<!-- 3. Initialize and update submodules
```
git submodule update --init --recursive
``` -->

3. Build both the analytical and reconfigurable backend
```
./scripts/build_backends.sh
```

4. Run the example as a sanity check
```
./scripts/run_example.sh
```

### Reproducing Experiments

#### Figure 12
```
cd scripts/fig12
./run_bw_exps.sh        # <-- Runs all scale-out bandwidth sweep experiments (Fig 12. right)
./run_latency_exps.sh   # <-- Runs all reconfig latency experiments (Fig 12. Left)
./plot_fig12.sh
```

#### Figure 13
```
cd scripts/fig13
./run_bw_exps.sh        # <-- Runs all scale-out bandwidth sweep experiments (Fig 13. right)
./run_latency_exps.sh   # <-- Runs all reconfig latency experiments (Fig 13. Left)
./plot_fig13.sh
```