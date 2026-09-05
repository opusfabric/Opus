# large_scale_ep_32gpu_512_all2allv_topk_context

Original `EP + TP + DP` 512-GPU sweep using explicit all2allv MoE
dispatch/combine with deterministic top-k context routing.

Run:

```bash
python3 examples/large_scale_ep_sweeps/sweep_ep.py --config examples/large_scale_ep_32gpu_512_all2allv_topk_context/config.json all --rerun
```

The matching scaleout `EP + DP` all2allv source is
`examples/large_scale_ep_scaleout_32gpu_512_all2allv_topk_context`.
