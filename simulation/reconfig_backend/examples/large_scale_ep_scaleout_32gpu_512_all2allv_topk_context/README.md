# large_scale_ep_scaleout_32gpu_512_all2allv_topk_context

GB200-style 512-GPU EP sweep with EP mapped only onto scale-out DP rails,
using explicit all2allv send/recv nodes for MoE EP dispatch/combine.

- total GPUs = 512
- TP = 32 GPUs per scale-up domain
- PP = 2
- DP = 8
- microbatches = 4
- scale-up = 900 GB/s
- scale-out = 800 Gbit/s, modeled as 100 GB/s
- EP sizes = 1, 2, 4, 8
- EP communication pattern = all2allv
- MoE routing = top-k context routing

The all2allv generator keeps the same per-rank EP payload scale as the
collective all-to-all baseline (`comm_size`) and redistributes that payload
across EP destinations according to deterministic top-k expert ownership over
context positions. Zero routed pairs are omitted from the p2p trace; positive transfers are issued in a deterministic pair order so matching send/recv nodes are posted without creating network traffic for absent routes.

Validate the setup:

```bash
bash run_sweep.sh validate
```

Run the full sweep and generate plots:

```bash
bash run_sweep.sh all
```
