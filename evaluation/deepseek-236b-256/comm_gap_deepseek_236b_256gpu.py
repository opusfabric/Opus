#!/usr/bin/env python3
"""
Communication gap analysis for DeepSeek-V2 236B on 256 GPUs.
Parallelism: EP=32, DP=8, PP=8, TP=4 (TP excluded per request).

Topology assumption:
  DP × PP × TP = 8 × 8 × 4 = 256  (dense layer parallelism)
  EP = 32 = GPUs per PP stage       (one PP stage spans 32 GPUs = EP group)
  Each PP stage: TP=4 sub-groups, DP=8 data replicas, EP=32 expert shards
  Expert layout: 160 routed experts across EP=32 GPUs -> 5 experts per GPU.
"""

# Model: DeepSeek-V2 236B
DIM            = 5120
INTER_DIM      = 12288
MOE_INTER_DIM  = 1536
N_LAYERS       = 60
N_DENSE_LAYERS = 1
N_MOE_LAYERS   = N_LAYERS - N_DENSE_LAYERS   # 59

N_HEADS        = 128
KV_LORA_RANK   = 512
Q_LORA_RANK    = 1536
QK_NOPE_DIM    = 128
QK_ROPE_DIM    = 64
V_HEAD_DIM     = 128

NUM_EXPERTS    = 160
NUM_SHARED     = 2
TOP_K          = 6

EP = 32; DP = 8; PP = 8; TP = 4
TOTAL_GPUS = 256

N_MICRO_BATCHES     = PP
MICRO_BATCH_SEQS    = 256 // DP // N_MICRO_BATCHES   # 4
SEQ_LEN             = 4096
MICRO_BATCH_TOKENS  = MICRO_BATCH_SEQS * SEQ_LEN     # 16384
BYTES_PER_ELEM      = 2
EXPERTS_PER_EP_RANK = NUM_EXPERTS // EP               # 5

DP_BW = EP_BW = PP_BW = 50e9   # 50 GB/s IB

# Attention params (MLA), TP-sharded
attn_params = (
    DIM * Q_LORA_RANK
    + Q_LORA_RANK * (QK_NOPE_DIM + QK_ROPE_DIM) * N_HEADS
    + DIM * (KV_LORA_RANK + QK_ROPE_DIM)
    + KV_LORA_RANK * (QK_NOPE_DIM + V_HEAD_DIM) * N_HEADS
    + N_HEADS * V_HEAD_DIM * DIM
) // TP
shared_exp_params = NUM_SHARED * 3 * DIM * MOE_INTER_DIM // TP
dense_ffn_params  = 3 * DIM * INTER_DIM // TP
norm_params       = 2 * DIM

dp_params = (
    N_DENSE_LAYERS * (attn_params + dense_ffn_params + norm_params)
    + N_MOE_LAYERS  * (attn_params + shared_exp_params + norm_params)
)

tokens = MICRO_BATCH_TOKENS

ep_dispatch_bytes   = tokens * TOP_K * DIM * BYTES_PER_ELEM
ep_gather_bytes     = ep_dispatch_bytes
ep_per_layer_bytes  = ep_dispatch_bytes + ep_gather_bytes
ep_per_step_bytes   = N_MOE_LAYERS * N_MICRO_BATCHES * ep_per_layer_bytes * 2  # fwd+bwd

dp_allreduce_bytes  = 2 * (DP - 1) / DP * dp_params * BYTES_PER_ELEM
pp_activation_bytes = tokens * DIM * BYTES_PER_ELEM
pp_per_step_bytes   = N_MICRO_BATCHES * 2 * pp_activation_bytes

T_ep = ep_per_step_bytes / EP_BW
T_dp = dp_allreduce_bytes / DP_BW
T_pp = pp_per_step_bytes  / PP_BW

def gb(b): return b / 1e9
def fmt_gb(b): return f"{gb(b):.3f} GB"
def fmt_ms(t): return f"{t*1e3:.2f} ms"

SEP = "=" * 72
sep = "-" * 72

print(SEP)
print("  DeepSeek-V2 236B — Communication Gap Analysis")
print("  256 GPUs | EP=32  DP=8  PP=8  TP=4 (TP comms excluded)")
print(SEP)
print(f"""
Model:  {N_LAYERS} layers ({N_DENSE_LAYERS} dense + {N_MOE_LAYERS} MoE) | dim={DIM}
        {NUM_EXPERTS} routed experts (top-{TOP_K}) + {NUM_SHARED} shared | expert_dim={MOE_INTER_DIM}
Batch:  {256} seqs x {SEQ_LEN} tok, {N_MICRO_BATCHES} micro-batches of {MICRO_BATCH_SEQS} seqs ({MICRO_BATCH_TOKENS:,} tokens)
BW:     {EP_BW/1e9:.0f} GB/s IB (all inter-node comms)
""")

print(sep)
print("DATA PARALLELISM (DP=8) — Gradient All-Reduce — once per step")
print(sep)
print(f"  DP-reducible params/GPU:  {dp_params/1e9:.3f} B  (attn+shared_exp+dense, /TP=4)")
print(f"  Ring all-reduce volume:   {fmt_gb(dp_allreduce_bytes)}")
print(f"  Time:                     {fmt_ms(T_dp)}")

print()
print(sep)
print("EXPERT PARALLELISM (EP=32) — MoE All-to-All — per MoE layer per micro-batch")
print(sep)
print(f"  Dispatch/layer/GPU:       {fmt_gb(ep_dispatch_bytes)}  ({MICRO_BATCH_TOKENS:,} tok x {TOP_K} x {DIM} x BF16)")
print(f"  Gather/layer/GPU:         {fmt_gb(ep_gather_bytes)}")
print(f"  Total/step (59L x 8μb x fwd+bwd): {fmt_gb(ep_per_step_bytes)}")
print(f"  Time:                     {fmt_ms(T_ep)}")

print()
print(sep)
print("PIPELINE PARALLELISM (PP=8) — Activation Send/Recv — per micro-batch boundary")
print(sep)
print(f"  Activation/micro-batch:   {fmt_gb(pp_activation_bytes)}")
print(f"  Total/step (8μb x fwd+bwd): {fmt_gb(pp_per_step_bytes)}")
print(f"  Time:                     {fmt_ms(T_pp)}")

print()
print(SEP)
print("SUMMARY — per training step, per GPU")
print(SEP)
entries = [("DP all-reduce", dp_allreduce_bytes, T_dp), ("EP all-to-all", ep_per_step_bytes, T_ep), ("PP send/recv", pp_per_step_bytes, T_pp)]
print(f"\n{'Type':<18} {'Volume':>12} {'Time':>12}")
print(f"{'':─<18} {'':─>12} {'':─>12}")
for name, vol, t in entries:
    print(f"{name:<18} {fmt_gb(vol):>12} {fmt_ms(t):>12}")

print(f"\n  EP / DP: {T_ep/T_dp:.1f}x  |  EP / PP: {T_ep/T_pp:.1f}x  |  DP / PP: {T_dp/T_pp:.1f}x")
print(f"\n  EP repeats {N_MOE_LAYERS}L x {N_MICRO_BATCHES}μb x 2 = {N_MOE_LAYERS*N_MICRO_BATCHES*2} all-to-alls/step")
print(f"  PP repeats {N_MICRO_BATCHES}μb x 2 = {N_MICRO_BATCHES*2} sends/step per boundary")
print(f"  DP: 1 all-reduce/step")
