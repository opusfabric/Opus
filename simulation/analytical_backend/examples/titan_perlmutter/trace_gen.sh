for i in {0..15}; do
    chakra_trace_link --chakra-host-trace pytorch_et_${i}.json --chakra-device-trace kineto_trace_${i}.json --rank ${i} --output-file chakra_host_device_trace_${i}.json
    chakra_converter PyTorch --input chakra_host_device_trace_${i}.json --output chakra_trace.${i}.et
done