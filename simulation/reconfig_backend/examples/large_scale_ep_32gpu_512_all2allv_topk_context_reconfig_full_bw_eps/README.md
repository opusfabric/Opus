# large_scale_ep_32gpu_512_all2allv_topk_context_reconfig_full_bw_eps

Zero-delay reconfiguration-backend EPS baseline for the 512-GPU all2allv
top-k-context EP sweep. For the constrained `EPS` report, the runner reuses the exact source
`opus_diverge` traces and schedules and runs `AstraSim_Analytical_Reconfigurable` with
network `reconfig_time: [ 0 ]`.

This compares only:

- `EPS`: zero-delay constrained baseline
- `OPUS diverged topology`: 500us, 1ms, and 10ms reconfiguration latencies

The merged plot overlays:

- `EP + TP + DP`: `large_scale_ep_32gpu_512_all2allv_topk_context`
- `EP + DP`: `large_scale_ep_scaleout_32gpu_512_all2allv_topk_context`

Run:

```bash
python3 run_full_bw_eps.py all --rerun
```

EPS uses BFS routing over the source OPUS schedules verbatim with `reconfig_time: [ 0 ]` (zero-cost reconfiguration lower bound).
