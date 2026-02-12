# Opus

**Opus** is a
a control plane that orchestrates parallelism-driven rail reconfiguration for a photonic-rail network. Opus introduces a control layer between ML
training frameworks and collective communication libraries.
Collective communication libraries act as clients to Opus
and issue provisional intents to communicate, while Opus
interfaces with network orchestrators to reconfigure the optical fabric at parallelism phase boundaries. Opus’s design ensures that reconfiguration is both safe—circuits are never torn down while traffic is in flight—and efficient—speculative
provisioning hides reconfiguration latency within natural
idle windows between parallelism phases.

We implement Opus as a custom PyTorch distributed backend. A training job enables Opus by passing backend="opus" to PyTorch’s `init_process_group`, or equivalently, by setting a single configuration flag in frameworks built on PyTorch’s distributed API (e.g., enable_opus_backend in TorchTitan). No changes to model code or parallelism constructs (DP/PP/TP/CP/EP) are required to use Opus. Opus intercepts
collective calls through PyTorch’s ProcessGroup abstraction
and routes them through the Opus shim, which delegates
data transport to NCCL.

Please note that the source code is under active reconfiguration. The experimental scripts contain out-dated commands and configurations. We do not guarantee executable results. 

## Features

- Control plane for in-job topology reconfiguration
- Bind to PyTorch distributed backend
- Support 3D-parallel LLM training with TorchTitan

## Getting Started

1. Clone the repository:
  ```bash
  git clone https://github.com/opusfabric/Opus
  ```
2. Navigate to the project directory:
  ```bash
  cd Opus
  ```
3. Follow the setup instructions under `environment`. We provide two installation methods. One is suitable for private testbeds with circuit-switching hardware, `testbed-env`. The other is useful for public clouds or super-computing platform with Slurm, `emulation-env`. We use docker containers for package and environment configuration.

4. The exeperiments and execution scripts are stored under `torchtitan/opus-test`
 
## Directory Structure

```
.
├── environment
│   ├── emulation-env
│   └── testbed-env
├── evaluation
│   ├── deepseek_v3_16b-2d-16-latency
│   ├── deepseek_v3_16b-3d-16-latency
│   ├── energy-analysis
│   ├── llama-3-3d-16-latency
│   ├── llama-3-3d-64-latency
│   └── llama-3-70b-128
├── nccl
├── README.md
├── src
│   ├── opus-controller
│   └── opus-shim
└── torchtitan
    ...
    ├── opus-test
    ...
    └── torchtitan
```

<!-- ## Contributing

Contributions are welcome! Please fork the repository and submit a pull request. -->

## License

See the `LICENSE` file for details.

## Contact

For questions or feedback, please reach out to opusfabric@gmail.com.
