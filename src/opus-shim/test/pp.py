import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint
from torch.distributed.pipelining import ScheduleGPipe
import opus


# ----- Simple building blocks -----

class Layer(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.linear(x))


class LMHead(nn.Module):
    def __init__(self, hidden_dim=16, vocab_size=20):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        return self.proj(x)


# ----- Full Model -----

class Model(nn.Module):
    def __init__(self, vocab_size=10, hidden_dim=16, num_layers=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim)

        # Place SplitPoints at boundaries where we want pipeline stages
        layers = []
        for i in range(num_layers):
            # if i == num_layers // 2:
            #     layers.append(SplitPoint.BEGIN)  # Start of stage 2
            layers.append(Layer(hidden_dim))
        self.layers = nn.ModuleList(layers)

        self.lm = LMHead(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.lm(x)
        return x


def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # backend = "nccl" if torch.cuda.is_available() else "gloo"
    backend = "cuda:opus"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size


def main():
    rank, world_size = setup_distributed()
    
    n_microbatches = 4
    batch_size = 16
    seq_len = 5
    vocab_size = 10

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    model = Model(vocab_size=10, hidden_dim=16, num_layers=4)
    model.to(device).eval()

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Wrap in pipeline; this automatically partitions at SplitPoints
    pipelined_model = pipeline(
        module=model,
        mb_args=(x[:4],),
        split_spec={
            "layers.2": SplitPoint.BEGINNING,
        },
    )

    stage = pipelined_model.build_stage(rank, device=device)

    schedule = ScheduleGPipe(stage, n_microbatches)

    # inference
    for _iter in range(5):
        if rank == 0:
            schedule.step(x)
        else:
            output = schedule.step()
            print(f"[Rank {rank}] Output shape: {output.shape}")
        

    # pipelined_model.train()

    # # Simple optimizer (only updates local stage parameters)
    # optimizer = torch.optim.SGD(
    #     filter(lambda p: p.requires_grad, pipelined_model.local_parameters()),
    #     lr=0.01,
    # )

    # # Dummy data
    # batch_size = 16
    # seq_len = 5
    # vocab_size = 10
    # x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # loss_fn = nn.CrossEntropyLoss()

    # for step in range(5):
    #     optimizer.zero_grad()
    #     out = pipelined_model(x)
    #     # out: [batch, seq, vocab]
    #     loss = loss_fn(out.view(-1, vocab_size), y.view(-1))
    #     loss.backward()
    #     optimizer.step()

    #     if rank == 0:
    #         print(f"[Rank {rank}] Step {step+1} | Loss: {loss.item():.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
