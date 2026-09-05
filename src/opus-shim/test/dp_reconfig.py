"""Small data-parallel smoke test that exercises controller reconfiguration."""

import os
import time

import torch
import torch.distributed as dist

import opus  # noqa: F401: registers the cuda:opus backend


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 2:
        raise RuntimeError(f"This demo expects exactly two ranks, got {world_size}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="cuda:opus", rank=rank, world_size=world_size)

    tensor = torch.full((1024,), float(rank + 1), device=f"cuda:{local_rank}")
    for iteration in range(3):
        tensor.fill_(float(rank + iteration + 1))
        torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        expected = float(3 + 2 * iteration)
        if not torch.allclose(tensor, torch.full_like(tensor, expected)):
            raise RuntimeError(
                f"rank {rank}: unexpected all-reduce value {tensor[0].item()}"
            )
        print(
            f"[DP rank {rank}] iteration={iteration} "
            f"sum={tensor[0].item():.0f} elapsed_ms={elapsed_ms:.2f}",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
