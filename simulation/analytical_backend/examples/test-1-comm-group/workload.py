import torch
from torch.profiler import ExecutionTraceObserver, profile
import os
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module
from torch.nn.parallel import DistributedDataParallel as DDP

def trace_handler(prof):
    rank = int(os.environ["LOCAL_RANK"])
    prof.export_chrome_trace(f"kineto_trace_{rank}.json")

# Dummy model
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        return self.seq(x)

def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def train(prof, et):
    setup()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")


    model = DummyModel().to(device)
    fsdp_model = FSDP(model, device_id=device)

    # Dummy data
    x = torch.randn(32, 8).to(device)
    y = torch.randn(32, 4).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=0.01)

    fsdp_model.train()
    for step in range(5):
        if step == 3:
            et.start()
        if step == 4:
            et.stop()
        optimizer.zero_grad()
        output = fsdp_model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"[Step {step}] Loss: {loss.item():.4f}")

        prof.step()

    cleanup()


def main():
    rank = int(os.environ["LOCAL_RANK"])
    et = ExecutionTraceObserver()
    et.register_callback(f"pytorch_et_{rank}.json")
    et.start()

    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=3, warmup=0, active=1),
        on_trace_ready=trace_handler
    ) as prof:
        train(prof, et)

    et.stop()
    et.unregister_callback()


if __name__ == "__main__":
    main()


