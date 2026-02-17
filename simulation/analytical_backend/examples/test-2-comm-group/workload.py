import os
import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity, ExecutionTraceObserver
import time
import torch.cuda.nvtx as nvtx


# Init env
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])

et = ExecutionTraceObserver()
et.register_callback(f"pytorch_et_{rank}.json")
et.start()

def trace_handler(prof):
    rank = int(os.environ["LOCAL_RANK"])
    prof.export_chrome_trace(f"kineto_trace_{rank}.json")

# Select device
torch.cuda.set_device(local_rank)

# Init default process group
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# Create two new groups using same ranks
group1 = dist.new_group(ranks=list(range(world_size)), backend="nccl")
group2 = dist.new_group(ranks=list(range(world_size)), backend="nccl")

# Allocate tensors
size = 32768
x1 = torch.full((size,), rank, dtype=torch.float, device="cuda")
x2 = torch.full((size,), rank + 100, dtype=torch.float, device="cuda")  # different base value

# Define a large linear layer
layer = torch.nn.Linear(size, size).cuda()

start_perf = time.perf_counter()
# Start profiling
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=3, warmup=0, active=1),
            on_trace_ready=trace_handler) as prof:


    for i in range(5):
        # with nvtx.range("matrix_multiplication 1"):
            if i == 3:
                et.start()
            if i == 4:
                et.stop()

            for _ in range(10):
                x1 = layer(x1)
                x2 = layer(x2)
            
            
            # torch.cuda.synchronize()  # Synchronize after matrix multiplication 1
            
        # with nvtx.range(f"group1_allreduce"):
            dist.all_reduce(layer.weight, group=group1)
            # torch.cuda.synchronize()  # Synchronize after group1 allreduce

        # with nvtx.range("matrix_multiplication 2"):
            for _ in range(10):
                x1 = layer(x1)
                x2 = layer(x2)
            #torch.cuda.synchronize()  # Synchronize after matrix multiplication 2

        # with nvtx.range("group2_allreduce"):
            # Perform allreduce in group2
            dist.all_reduce(layer.weight, group=group2)
            # torch.cuda.synchronize()  # Synchronize after group2 allreduce

            prof.step()

torch.cuda.synchronize()
dist.destroy_process_group()

et.stop()
et.unregister_callback()

end_perf = time.perf_counter()

elapsed_perf = end_perf - start_perf
print(f"Elapsed time (perf_counter): {elapsed_perf} seconds")
# Export profiler trace to a JSON file

# Cleanup
# dist.destroy_process_group()
