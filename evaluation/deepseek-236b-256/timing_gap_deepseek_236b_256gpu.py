#!/usr/bin/env python3
"""
Timing gap analysis: gaps between adjacent collectives of DIFFERENT parallelism
types in a DeepSeek-V2 236B training step (EP=32, DP=8, PP=8, TP=4).

Definition:
  "gap(A→B)" = compute time between the end of a type-A collective and
               the start of the immediately following type-B collective,
               where A != B and no other collective lies in between.

Training step timeline (non-overlapped, 1F1B schedule):
  For each micro-batch in forward pass (M=8 micro-batches):
    [PP_recv] → attn/FFN → [EP_dispatch] → expert → [EP_gather] → shared+combine → [PP_send]

  For each micro-batch in backward pass (reversed layer order):
    [PP_recv_bwd] → shared_bwd+combine_bwd → [EP_bwd_dispatch] → expert_bwd → [EP_bwd_gather]
                 → norm+attn_bwd → ... (repeat per layer) → [PP_send_bwd]

  After all micro-batches:
    [DP_allreduce]

Adjacent cross-type windows identified:
  FWD: PP_recv  → EP_dispatch   (PP→EP): gap = layers-before-first-EP compute
  FWD: EP_gather → PP_send      (EP→PP): gap = shared + combine (post-FFN)
  BWD: PP_recv  → EP_bwd_dispatch (PP→EP): gap = shared_bwd + combine_bwd
  BWD: EP_bwd_gather → PP_send  (EP→PP): gap = attn_bwd + [dense_bwd for stage-0]
  END: PP_send_bwd → DP_allreduce (PP→DP): gap ≈ 0 (optimizer overhead excluded)
  NEXT: DP_allreduce → PP_recv_fwd (DP→PP): gap = 0 (back-to-back steps, excl. optimizer)

Note: EP and DP are never directly adjacent (PP send/recv always intervenes).
"""

# ─────────────────────────────────────────────────────────────────────────────
# Model config: DeepSeek-V2 236B
# ─────────────────────────────────────────────────────────────────────────────
DIM           = 5120
INTER_DIM     = 12288
MOE_INTER_DIM = 1536
N_LAYERS      = 60
N_DENSE       = 1
N_MOE         = N_LAYERS - N_DENSE   # 59

N_HEADS       = 128
KV_LORA_RANK  = 512
Q_LORA_RANK   = 1536
QK_NOPE_DIM   = 128
QK_ROPE_DIM   = 64
V_HEAD_DIM    = 128

NUM_EXPERTS   = 160
NUM_SHARED    = 2
TOP_K         = 6

EP = 32; DP = 8; PP = 8; TP = 4

N_MICRO       = PP          # 1F1B: micro-batches = PP stages
MICRO_SEQS    = 256 // DP // N_MICRO   # 4 sequences per micro-batch
SEQ_LEN       = 4096
TOKENS        = MICRO_SEQS * SEQ_LEN   # 16 384 tokens per micro-batch
B2            = 2                       # BF16

EXPERTS_PER_RANK = NUM_EXPERTS // EP   # 5

# ─────────────────────────────────────────────────────────────────────────────
# Bandwidth
# ─────────────────────────────────────────────────────────────────────────────
IB_BW = 50e9   # 50 GB/s InfiniBand (EP, DP, PP all inter-node)

# ─────────────────────────────────────────────────────────────────────────────
# Hardware: H100 SXM5
# ─────────────────────────────────────────────────────────────────────────────
PEAK_TFLOPS = 989e12   # BF16 with sparsity
MFU         = 0.50
EFF_FLOPS   = PEAK_TFLOPS * MFU   # ~494 TFLOPS effective

def flop_time(flops): return flops / EFF_FLOPS

# ─────────────────────────────────────────────────────────────────────────────
# Collective times (per operation, per GPU)
# ─────────────────────────────────────────────────────────────────────────────
# EP: dispatch + gather (per MoE layer, per micro-batch)
T_ep_dispatch = TOKENS * TOP_K * DIM * B2 / IB_BW
T_ep_gather   = T_ep_dispatch   # symmetric
T_ep_layer    = T_ep_dispatch + T_ep_gather

# PP: one activation send or recv (per micro-batch, per stage boundary)
T_pp_xfer     = TOKENS * DIM * B2 / IB_BW

# DP: ring all-reduce of non-expert params
attn_p   = (DIM*Q_LORA_RANK + Q_LORA_RANK*(QK_NOPE_DIM+QK_ROPE_DIM)*N_HEADS
           + DIM*(KV_LORA_RANK+QK_ROPE_DIM) + KV_LORA_RANK*(QK_NOPE_DIM+V_HEAD_DIM)*N_HEADS
           + N_HEADS*V_HEAD_DIM*DIM) // TP
shared_p = NUM_SHARED * 3 * DIM * MOE_INTER_DIM // TP
dense_p  = 3 * DIM * INTER_DIM // TP
norm_p   = 2 * DIM
dp_params = (N_DENSE*(attn_p+dense_p+norm_p) + N_MOE*(attn_p+shared_p+norm_p))
T_dp      = 2*(DP-1)/DP * dp_params * B2 / IB_BW

# ─────────────────────────────────────────────────────────────────────────────
# Compute times (per layer, per micro-batch, this GPU)
# ─────────────────────────────────────────────────────────────────────────────
heads_tp = N_HEADS // TP   # TP shards attention heads

# MLA attention forward
attn_flops_fwd = (
    2*TOKENS*DIM*Q_LORA_RANK                                            # Q compress
  + 2*TOKENS*Q_LORA_RANK*heads_tp*(QK_NOPE_DIM+QK_ROPE_DIM)           # Q decompress
  + 2*TOKENS*DIM*(KV_LORA_RANK+QK_ROPE_DIM)                           # KV compress (no TP shard)
  + 2*TOKENS*KV_LORA_RANK*heads_tp*(QK_NOPE_DIM+V_HEAD_DIM)           # KV decompress
  + 2*TOKENS*SEQ_LEN*heads_tp*(QK_NOPE_DIM+QK_ROPE_DIM)               # QK^T
  + 2*TOKENS*SEQ_LEN*heads_tp*V_HEAD_DIM                               # AV
  + 2*TOKENS*heads_tp*V_HEAD_DIM*DIM                                   # O proj
)
T_attn_fwd = flop_time(attn_flops_fwd)
T_attn_bwd = 2 * T_attn_fwd

# Dense FFN forward (layer 0, TP-sharded)
T_dense_fwd = flop_time(3 * 2 * TOKENS * DIM * (INTER_DIM // TP))
T_dense_bwd = 2 * T_dense_fwd

# Shared expert forward (NUM_SHARED experts, TP-sharded)
T_shared_fwd = flop_time(NUM_SHARED * 3 * 2 * TOKENS * DIM * (MOE_INTER_DIM // TP))
T_shared_bwd = 2 * T_shared_fwd

# Expert combine: weighted sum of top_k outputs (small)
T_combine_fwd = flop_time(2 * TOKENS * TOP_K * DIM)
T_combine_bwd = 2 * T_combine_fwd

# ─────────────────────────────────────────────────────────────────────────────
# PP stage layout
# ─────────────────────────────────────────────────────────────────────────────
# Layer 0 (dense) lives in stage 0.  Remaining 59 MoE layers distributed round-robin.
# Stage 0 gets: 1 dense + floor(59/8) = 7 MoE layers = 8 layers total
# Stages 1-4:   0 dense + 7 MoE layers
# Stages 5-7:   0 dense + 8 MoE layers  (59 - 7*5 = 59-35 = 24 → 8 each for 3 stages)
# Actually 7*5 + 8*3 = 35+24 = 59 ✓
stage_layout = {}   # stage -> (n_dense, n_moe)
stage_layout[0] = (1, 7)
for s in range(1, 5):
    stage_layout[s] = (0, 7)
for s in range(5, 8):
    stage_layout[s] = (0, 8)

# ─────────────────────────────────────────────────────────────────────────────
# Timeline model: enumerate all adjacent cross-type collective pairs
# ─────────────────────────────────────────────────────────────────────────────
# For each stage, per micro-batch, forward and backward, we find:
#   - (PP→EP) windows: gap between PP_recv and first EP_dispatch
#   - (EP→PP) windows: gap between last EP_gather/dispatch and PP_send
#   - (PP→DP) window: at end of all backward passes
#   - (DP→PP) window: at start of next step's forward
#
# EP→EP transitions (between consecutive MoE layer EP ops) are SAME type → skipped.

windows = {}   # (type_A, type_B) -> list of gap times (seconds)

def add_window(pair, gap_s):
    windows.setdefault(pair, []).append(gap_s)

for stage, (n_dense, n_moe) in stage_layout.items():
    for _ in range(N_MICRO):

        # ── FORWARD micro-batch ───────────────────────────────────────────
        # Sequence:
        #   PP_recv_fwd
        #   [dense: T_attn_fwd + T_dense_fwd] × n_dense
        #   for each MoE layer:
        #       T_attn_fwd
        #       EP_dispatch
        #       [expert compute]
        #       EP_gather
        #       T_shared_fwd + T_combine_fwd
        #   PP_send_fwd

        if n_moe > 0:
            # PP_recv → first EP_dispatch:
            #   gap = dense layers compute + first MoE attention
            gap_pp_to_ep_fwd = (
                n_dense * (T_attn_fwd + T_dense_fwd)   # dense layer(s) before first MoE
                + T_attn_fwd                             # first MoE layer: attn before EP_dispatch
            )
            add_window(("PP", "EP"), gap_pp_to_ep_fwd)

            # last EP_gather → PP_send_fwd:
            #   gap = shared + combine (post-FFN of last MoE layer)
            gap_ep_to_pp_fwd = T_shared_fwd + T_combine_fwd
            add_window(("EP", "PP"), gap_ep_to_pp_fwd)

            # Between consecutive MoE layers: EP_gather[i] → EP_dispatch[i+1]
            # This is EP→EP (same type) — tracked separately for reference
            # gap = (T_shared_fwd + T_combine_fwd) + T_attn_fwd  [post-FFN + next attn]

        # ── BACKWARD micro-batch ──────────────────────────────────────────
        # Backward of MoE layer (reversed within each layer):
        #   shared_bwd + combine_bwd
        #   EP_bwd_dispatch  (backward of EP_gather: send grad to expert GPUs)
        #   expert_bwd
        #   EP_bwd_gather    (backward of EP_dispatch: recv grad from expert GPUs)
        #   norm_bwd + attn_bwd
        #
        # Full backward sequence for one PP stage (last MoE layer first):
        #   PP_recv_bwd
        #   [shared_bwd + combine_bwd]        ← gap before first EP_bwd_dispatch
        #   EP_bwd_dispatch_N                 ← EP collective
        #   [expert_bwd]
        #   EP_bwd_gather_N                   ← EP collective
        #   [norm_bwd + attn_bwd]             ← inter-layer compute
        #   ...  (EP_bwd_dispatch/gather for layers N-1 down to 1)
        #   EP_bwd_gather_1                   ← last EP collective in this micro-batch
        #   [norm_bwd + attn_bwd_layer1]      ← gap before PP_send_bwd
        #   [dense_bwd × n_dense]             ← dense layer backward
        #   PP_send_bwd

        if n_moe > 0:
            # PP_recv_bwd → first EP_bwd_dispatch  (= backward of last MoE EP_gather):
            #   gap = shared_bwd + combine_bwd of last MoE layer
            gap_pp_to_ep_bwd = T_shared_bwd + T_combine_bwd
            add_window(("PP", "EP"), gap_pp_to_ep_bwd)

            # last EP_bwd_gather → PP_send_bwd:
            #   gap = attn_bwd of first MoE layer + all dense layers backward
            gap_ep_to_pp_bwd = (
                T_attn_bwd                              # first MoE layer attn backward
                + n_dense * (T_attn_bwd + T_dense_bwd) # dense layer(s) backward
            )
            add_window(("EP", "PP"), gap_ep_to_pp_bwd)

# ── PP → DP (end of backward, once per step) ─────────────────────────────────
# After last PP_send_bwd, DP all-reduce starts.
# Gap = time between last PP_send and DP start.
# In non-overlapped setup this is 0 (DP starts immediately after last backward).
# In overlapped (reduce-scatter during backward): even tighter.
add_window(("PP", "DP"), 0.0)

# ── DP → PP (start of next step, once per step boundary) ─────────────────────
# After DP all-reduce, next step's first PP_recv starts.
# Gap = optimizer weight update time (not modelled here → 0 for comm-only analysis).
add_window(("DP", "PP"), 0.0)

# ── EP and DP are never directly adjacent ─────────────────────────────────────
# Between last EP collective and DP: there is always a PP_send_bwd in between.
# Between DP and first EP: there is always a PP_recv_fwd in between.
# Hence ("EP","DP") and ("DP","EP") windows do not exist in this timeline.

# ─────────────────────────────────────────────────────────────────────────────
# EP→EP intra-layer gaps (same type, not a cross-type window, but useful reference)
# ─────────────────────────────────────────────────────────────────────────────
ep_ep_fwd_gap = T_shared_fwd + T_combine_fwd + T_attn_fwd   # between MoE layers, fwd
ep_ep_bwd_gap = T_attn_bwd + T_shared_bwd + T_combine_bwd   # between MoE layers, bwd

# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def fmt_ms(t): return f"{t*1e3:.3f} ms"
def fmt_us(t): return f"{t*1e6:.1f} μs"

SEP = "=" * 76
sep = "-" * 76

print(SEP)
print("  Timing Gap Analysis — DeepSeek-V2 236B | EP=32 DP=8 PP=8 TP=4")
print(SEP)

print(f"""
Model:     {N_LAYERS} layers ({N_DENSE} dense + {N_MOE} MoE)  |  dim={DIM}
Micro-batch: {TOKENS:,} tokens  |  {N_MICRO} micro-batches  |  BF16
Hardware:  H100 SXM5, {EFF_FLOPS/1e12:.0f} TFLOPS effective (MFU={MFU*100:.0f}%)
Network:   {IB_BW/1e9:.0f} GB/s InfiniBand (all inter-node)
""")

print(sep)
print("Collective operation times:")
print(sep)
print(f"  EP dispatch (per layer, per μbatch) : {fmt_ms(T_ep_dispatch)}")
print(f"  EP gather   (per layer, per μbatch) : {fmt_ms(T_ep_gather)}")
print(f"  PP send/recv (per μbatch)           : {fmt_ms(T_pp_xfer)}")
print(f"  DP all-reduce (per step)            : {fmt_ms(T_dp)}")

print()
print(sep)
print("Compute times between collectives (per layer, per μbatch):")
print(sep)
print(f"  T_attn_fwd   (MLA attention fwd)    : {fmt_ms(T_attn_fwd)}")
print(f"  T_attn_bwd   (MLA attention bwd)    : {fmt_ms(T_attn_bwd)}")
print(f"  T_dense_fwd  (dense FFN fwd, L0)    : {fmt_ms(T_dense_fwd)}")
print(f"  T_dense_bwd  (dense FFN bwd, L0)    : {fmt_ms(T_dense_bwd)}")
print(f"  T_shared_fwd (shared experts fwd)   : {fmt_ms(T_shared_fwd)}")
print(f"  T_shared_bwd (shared experts bwd)   : {fmt_ms(T_shared_bwd)}")
print(f"  T_combine    (expert output combine) : {fmt_ms(T_combine_fwd)}")

print()
print(sep)
print("PP stage layout (layers per stage):")
print(sep)
for s, (nd, nm) in sorted(stage_layout.items()):
    print(f"  Stage {s}: {nd} dense + {nm} MoE  = {nd+nm} layers")

print()
print(SEP)
print("  Adjacent Cross-Parallelism Timing Windows")
print(SEP)
print("""
A window exists when consecutive collectives A (end) → B (start) are of different
parallelism types (A≠B) with no other collective in between.
Gap = compute time between end of A and start of B.
""")

print(f"{'Pair (A→B)':<14} {'Windows':>8}  {'Avg Gap':>12}  {'Min':>12}  {'Max':>12}  Notes")
print(sep)

pair_notes = {
    ("PP", "EP"): "PP_recv done → EP_dispatch starts (attn/dense compute in between)",
    ("EP", "PP"): "EP_gather done → PP_send starts (shared+combine or attn compute)",
    ("PP", "DP"): "last PP_send_bwd → DP all-reduce begins",
    ("DP", "PP"): "DP all-reduce done → next step PP_recv (excl. optimizer time)",
}

for pair in [("PP","EP"), ("EP","PP"), ("PP","DP"), ("DP","PP")]:
    vals  = windows.get(pair, [])
    n     = len(vals)
    if n == 0:
        print(f"  {pair[0]}→{pair[1]}    {'0':>8}  {'N/A':>12}")
        continue
    avg = sum(vals) / n
    mn  = min(vals)
    mx  = max(vals)
    note = pair_notes.get(pair, "")
    print(f"  {pair[0]}→{pair[1]}    {n:>8}  {fmt_ms(avg):>12}  {fmt_ms(mn):>12}  {fmt_ms(mx):>12}")
    print(f"            {note}")
    print()

# Also report EP→EP intra-layer gaps for reference
print(sep)
print("Reference: EP→EP gaps (same-type, not a cross-parallelism window)")
print(sep)
total_ep_ep_fwd = N_MOE * N_MICRO * PP * (N_MOE - 1) if N_MOE > 1 else 0
# More precisely: per stage, N_moe-1 intra-MoE gaps per micro-batch
ep_ep_count_fwd = sum((nm - 1) * N_MICRO for _, (_, nm) in stage_layout.items() if nm > 1)
ep_ep_count_bwd = ep_ep_count_fwd
print(f"  EP→EP (fwd): {ep_ep_count_fwd} windows, gap = {fmt_ms(ep_ep_fwd_gap)} each")
print(f"               (shared+combine+attn between consecutive MoE layers)")
print(f"  EP→EP (bwd): {ep_ep_count_bwd} windows, gap = {fmt_ms(ep_ep_bwd_gap)} each")
print(f"               (attn+norm+shared+combine between consecutive MoE layers bwd)")

print()
print(SEP)
print("  Summary: total windows and average gaps")
print(SEP)
all_pp_ep  = windows.get(("PP","EP"), [])
all_ep_pp  = windows.get(("EP","PP"), [])
all_pp_dp  = windows.get(("PP","DP"), [])
all_dp_pp  = windows.get(("DP","PP"), [])

total_cross = len(all_pp_ep) + len(all_ep_pp) + len(all_pp_dp) + len(all_dp_pp)
print(f"""
  Total cross-parallelism windows per training step: {total_cross}
    PP→EP : {len(all_pp_ep):3d} windows  (fwd+bwd, all stages, all micro-batches)
    EP→PP : {len(all_ep_pp):3d} windows
    PP→DP :   1 window   (end of backward)
    DP→PP :   1 window   (start of next step)

  Average gaps:
    PP→EP avg : {fmt_ms(sum(all_pp_ep)/len(all_pp_ep))}   range [{fmt_ms(min(all_pp_ep))}, {fmt_ms(max(all_pp_ep))}]
    EP→PP avg : {fmt_ms(sum(all_ep_pp)/len(all_ep_pp))}   range [{fmt_ms(min(all_ep_pp))}, {fmt_ms(max(all_ep_pp))}]
    PP→DP avg : {fmt_ms(0.0)}   (no compute gap; DP starts immediately)
    DP→PP avg : {fmt_ms(0.0)}   (no compute gap; next PP_recv starts immediately)

  EP and DP are NEVER directly adjacent:
    A PP send/recv always intervenes between any EP and DP collective.
    => No (EP→DP) or (DP→EP) windows exist in this timeline.
""")

print("Interpretation:")
print(f"  - PP→EP gaps dominate in FORWARD (avg {fmt_ms(sum(all_pp_ep)/len(all_pp_ep))}):  ")
print(f"    attention compute before each forward MoE EP collective.")
print(f"  - EP→PP gaps dominate in BACKWARD (avg {fmt_ms(sum(all_ep_pp)/len(all_ep_pp))}):  ")
print(f"    attention backward after last EP collective in each backward micro-batch.")
print(f"  - These gaps represent compute windows that *could* overlap EP/PP comms.")
print(f"  - PP→DP and DP→PP gaps are ~0 ms; DP latency ({fmt_ms(T_dp)}) is the")
print(f"    actual bottleneck, not the transition gap.")
