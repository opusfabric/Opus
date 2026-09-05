# large_scale_ep_sweeps_scaleout

Variant of `examples/large_scale_ep_sweeps` where EP groups are placed only on
the scale-out dimension.

The original generator builds contiguous EP rank blocks inside each PP stage.
That means small EP sizes overlap a TP/scale-up domain, and larger EP sizes are
formed from full TP domains. This generator instead builds each EP group from a
single TP rail across nodes:

```text
EP(stage, rail, group) = ranks with the same local GPU index across nodes
```

TP remains a physical scale-up group within each node. DP remains a full rail
across all nodes in a PP stage. EP is therefore a subset of a DP rail and does
not share ranks across the TP dimension.

Because one EP group can take only one local GPU index per node, valid EP sizes
are limited to divisors of `nodes_per_stage`.
