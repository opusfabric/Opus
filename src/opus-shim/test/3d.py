import os
import torch

import torch.distributed as dist
import opus
import time

def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # backend = "nccl" if torch.cuda.is_available() else "gloo"
    backend = "cuda:nccl"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size

def send_recv(rank, world_size):
    """Perform send and recv operations."""
    tensor = torch.zeros(10).cuda()  # Use GPU local rank 0
    if rank == 0:
        # Rank 0 sends a tensor to Rank 1
        tensor += 1
        dist.send(tensor=tensor, dst=1)
        print(f"Rank {rank} sent tensor with size {tensor.size()} to Rank 1")
    elif rank == 1:
        print("Rank 1 sleeping for 5 seconds before recv")
        time.sleep(5) 
        # Rank 1 receives a tensor from Rank 0
        dist.recv(tensor=tensor, src=0)
        print(f"Rank {rank} received tensor with size {tensor.size()} from Rank 0")
        print(f"Rank {rank} tensor content: {tensor}")

    print("Second round of send/recv")

    tensor = torch.zeros(10).cuda()  # Use GPU local rank 0
    if rank == 0:
        # Rank 0 sends a tensor to Rank 1
        tensor += 2
        print("Rank 0 sleeping for 5 seconds before recv")
        time.sleep(5)
        dist.send(tensor=tensor, dst=1)
        print(f"Rank {rank} sent tensor with size {tensor.size()} to Rank 1")
    elif rank == 1:
        # Rank 1 receives a tensor from Rank 0
        dist.recv(tensor=tensor, src=0)
        print(f"Rank {rank} received tensor with size {tensor.size()} from Rank 0")
        print(f"Rank {rank} tensor content: {tensor}")

def three_dimension(rank, world_size):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    


def main():
    rank, world_size = setup_distributed()
    send_recv(rank, world_size)

if __name__ == "__main__":
    main()