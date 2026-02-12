import torch

import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor, init_device_mesh
import os
import opus
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])

def run_data_parallel():
    dist.init_process_group(backend='cpu:gloo,cuda:opus', rank=rank, world_size=world_size)

    device_mesh = init_device_mesh("cuda", (2,))
    local_rank = device_mesh.get_local_rank()

    local_tensor = torch.ones(3) * (local_rank + 1)
    print(f"local rank: {local_rank}, local tensor: {local_tensor}")

    dtensor = DTensor.from_local(local_tensor, device_mesh, [Replicate()])
    
    if local_rank == 1:
        print(dtensor)
    result = dtensor.to_local()

    if local_rank == 1:
        print(result)

    dist.all_reduce(result, op=dist.ReduceOp.SUM)

    print(f"local rank: {local_rank}, result after all_reduce: {result}")

    # Perform reduce_scatter operation
    output_tensor = torch.zeros(3, device="cuda")
    input_tensor = torch.ones(3, device="cuda") * (local_rank + 1)

    dist.reduce_scatter(output_tensor, [input_tensor * i for i in range(world_size)], op=dist.ReduceOp.SUM)
    print(f"local rank: {local_rank}, output after reduce_scatter: {output_tensor}")

    # perform all_gather
    output_tensor = [torch.zeros(3, device="cuda") for _ in range(world_size)]
    input_tensor = torch.ones(3, device="cuda") * (local_rank + 1)

    dist.all_gather(output_tensor, input_tensor)
    print(f"local rank: {local_rank}, output after all_gather: {output_tensor}")

    # dist.destroy_process_group()

if __name__ == "__main__":
    run_data_parallel()
