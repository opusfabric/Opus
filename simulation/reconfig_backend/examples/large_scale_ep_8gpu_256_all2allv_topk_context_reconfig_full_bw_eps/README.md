# large_scale_ep_8gpu_256_all2allv_topk_context_reconfig_full_bw_eps

Zero-delay reconfiguration-backend EPS baseline for the 256-GPU all2allv top-k-context EP sweep.

For constrained `EPS`, this reuses the exact source `opus_diverge` traces and schedules and runs `AstraSim_Analytical_Reconfigurable` with `reconfig_time: [ 0 ]`.

Run:

```bash
python3 run_full_bw_eps.py all --rerun
```

EPS uses BFS routing over the source OPUS schedules verbatim with `reconfig_time: [ 0 ]` (zero-cost reconfiguration lower bound).
