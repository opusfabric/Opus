import os
import torch
import opus
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
import torch.cuda.nvtx as nvtx

# Init env
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])

# Select device
torch.cuda.set_device(local_rank)

# Init default process group
dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

# Create two new groups using same ranks
group1 = dist.new_group(ranks=list(range(world_size)), backend="gloo")
group2 = dist.new_group(ranks=list(range(world_size)), backend="gloo")

# Allocate tensors
size = 32768 # 2 ** 15
x1 = torch.full((size,), rank, dtype=torch.float, device="cpu")
x2 = torch.full((size,), rank + 100, dtype=torch.float, device="cpu")  # different base value

# Define a large linear layer
layer = torch.nn.Linear(size, size)

# Start profiling
with profile(activities=[ProfilerActivity.CPU], 
             on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')) as prof:
    for _ in range(5):
        # with nvtx.range("matrix_multiplication 1"):
            x1 = layer(x1)
            x2 = layer(x2)
            # torch.cuda.synchronize()  # Synchronize after matrix multiplication 1
            
        # with nvtx.range(f"group1_allreduce"):
            dist.all_reduce(layer.weight, group=group1, async_op=False)
            # torch.cuda.synchronize()  # Synchronize after group1 allreduce

        # with nvtx.range("matrix_multiplication 2"):
            x1 = layer(x1)
            x2 = layer(x2)
            #torch.cuda.synchronize()  # Synchronize after matrix multiplication 2

        # with nvtx.range("group2_allreduce"):
            # Perform allreduce in group2
            dist.all_reduce(layer.weight, group=group2, async_op=False)
            # torch.cuda.synchronize()  # Synchronize after group2 allreduce

# torch.cuda.synchronize()
# Export profiler trace to a JSON file
prof.export_chrome_trace("./trace.json")

# Cleanup
dist.destroy_process_group()