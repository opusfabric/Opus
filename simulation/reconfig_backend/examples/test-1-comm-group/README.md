# Generate trace from PyTorch model

https://github.com/mlcommons/chakra/wiki/Chakra-Execution-Trace-Collection-%E2%80%90-A-Comprehensive-Guide-on-Merging-PyTorch-and-Kineto-Traces

```
torchrun --nproc_per_node=2 workload.py

chakra_trace_link --chakra-host-trace pytorch_et_0.json --chakra-device-trace kineto_trace_0.json --rank 0 --output-file chakra_host_device_trace_0.json

chakra_converter PyTorch --input chakra_host_device_trace_0.json --output chakra_trace.0.et

chakra_trace_link --chakra-host-trace pytorch_et_1.json --chakra-device-trace kineto_trace_1.json --rank 1 --output-file chakra_host_device_trace_1.json

chakra_converter PyTorch --input chakra_host_device_trace_1.json --output chakra_trace.1.et
```

## Simulation
https://github.com/mlcommons/chakra/wiki/Running-Simulation-with-Chakra

```
docker run --rm --gpus all  -it -v "$(pwd)":/app/astra-sim -w /app/astra-sim astra-env bash

cd examples/test
./run_network_analytical.sh
```
