# large_scale_ep_8gpu_256_reconfig_full_bw_eps

Zero-delay reconfiguration-backend EPS baseline for the 256-GPU EP sweep.

The reported `EPS` baseline uses `AstraSim_Analytical_Reconfigurable`, the same
OPUS DP/PP/EP topology constraints, the same reconfigurable routing path, and
`reconfig_time: [ 0 ]`. This isolates the impact of OPUS reconfiguration delay.

The report compares:

- `EPS`: zero-delay constrained baseline
- `OPUS diverged topology`: 500us, 1ms, and 10ms reconfiguration latencies

The merged plot overlays both EP implementations in one axes:

- `EP + TP + DP`: original EP placement
- `EP + DP`: scale-out EP placement

`EPS` uses the same purple triangle styling as the native baseline in
`/home/dd687/Opus/evaluation/plot_provision_combined.py`.

Run everything:

```bash
python3 run_full_bw_eps.py all --rerun
```

Useful subsets:

```bash
python3 run_full_bw_eps.py run --rerun
python3 run_full_bw_eps.py compare
python3 run_full_bw_eps.py plot
```

Outputs:

- `results/summary.csv`: EPS baseline rows
- `results/eps_vs_opus_diverged_max.csv`
- `plots/eps_vs_opus_diverged_max_log.png`
- `plots/eps_vs_opus_diverged_max_log.pdf`
