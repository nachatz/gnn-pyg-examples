import torch
from torch import Tensor

LISTENER = 0


def star_graph(messages: list[Tensor]) -> tuple[Tensor, Tensor]:
    senders = torch.arange(1, len(messages) + 1)
    # A pure aggregator never reads the listener's own vector; zeros just keep every row the same shape.
    listener = torch.zeros_like(messages[0])
    x = torch.stack([listener, *messages])
    edge_index = torch.stack([senders, torch.full_like(senders, LISTENER)])
    return x, edge_index
