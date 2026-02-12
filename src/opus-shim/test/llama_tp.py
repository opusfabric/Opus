import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import opus

from torch.distributed import init_process_group
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

from log_utils import rank_log, get_logger, verify_min_gpu_count
from llama2 import Transformer, ModelArgs

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel,
)

import torch.cuda.nvtx as nvtx

# ---- Init ----------------------------------------------------------------
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
init_process_group(backend="cuda:opus")

# ---- World Info ---------------------------------------------------------
logger = get_logger()
_rank        = int(os.environ["RANK"])
_world_size  = int(os.environ["WORLD_SIZE"])
_local_rank  = int(os.environ["LOCAL_RANK"])

tp_size = _world_size

device_mesh = init_device_mesh("cuda", (tp_size,), mesh_dim_names=("tp",))
rank_log(_rank, logger, f"Device mesh created: {device_mesh}")

tp_mesh = device_mesh["tp"]
tp_rank = tp_mesh.get_local_rank()
rank_log(_rank, logger, f"TP rank: {tp_rank}")

# ---- Build Model --------------------------------------------------------
simple_llama2_config = ModelArgs(dim=3072, n_layers=24, n_heads=24, vocab_size=32000)
model = Transformer.from_model_args(simple_llama2_config).to("cuda")
model.init_weights()

# TP: Embeddings + Layers
model = parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Replicate()
        ),
    }
)

for layer_id, block in enumerate(model.layers):
    layer_tp_plan = {
        "attention_norm": SequenceParallel(),
        "attention": PrepareModuleInput(
            input_layouts=(Shard(1), Replicate()),
            desired_input_layouts=(Replicate(), Replicate()),
        ),
        "attention.wq": ColwiseParallel(use_local_output=False),
        "attention.wk": ColwiseParallel(use_local_output=False),
        "attention.wv": ColwiseParallel(use_local_output=False),
        "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
        "ffn_norm": SequenceParallel(),
        "feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        "feed_forward.w3": ColwiseParallel(),
    }

    # Shard heads across TP ranks
    # attn = block.attention
    # attn.n_heads    //= tp_mesh.size()
    # attn.n_kv_heads //= tp_mesh.size()
    parallelize_module(block, tp_mesh, layer_tp_plan)


# ---- Dataset ------------------------------------------------------------
# torch.manual_seed(42)
# total_samples = 512
batch_size = 8
# seq_len = 3072

# Create dummy dataset
# tokens = torch.randint(0, 32000, (total_samples, seq_len))
# targets = torch.randint(0, 32000, (total_samples, seq_len))

# dataset = TensorDataset(tokens, targets)
# sampler = DistributedSampler(dataset, num_replicas=dp_size, rank=dp_rank, shuffle=True)
# dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# ---- Optimizer ----------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

# ---- Training -----------------------------------------------------------
print(f"Rank {_rank}: Starting training ...")
num_iterations = 3

# for iter_i, (inp, tgt) in zip(range(num_iterations), dataloader):
for i in range(num_iterations):
    torch.manual_seed(i)
    inp = torch.randint(32000, (batch_size, 256), device="cuda")

    nvtx.range_push(f"Itr {i}")
    optimizer.zero_grad(set_to_none=True)

    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end   = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end   = torch.cuda.Event(enable_timing=True)

    nvtx.range_push("Forward pass")
    forward_start.record()
    output = model(inp)
    # loss = F.cross_entropy(output.view(-1, output.size(-1)), tgt.view(-1))
    output = output.sum()
    forward_end.record()
    torch.cuda.synchronize()
    nvtx.range_pop()

    nvtx.range_push("Backward pass")
    backward_start.record()
    output.backward()
    optimizer.step()
    backward_end.record()
    torch.cuda.synchronize()
    nvtx.range_pop()
    nvtx.range_pop()


    fwd_time = forward_start.elapsed_time(forward_end)
    bwd_time = backward_start.elapsed_time(backward_end)
    print(f"Rank {_rank}: Iter {i} | Loss: {output.item():.4f} | Fwd: {fwd_time:.2f}ms | Bwd: {bwd_time:.2f}ms")

print(f"Rank {_rank}: Training complete.")
