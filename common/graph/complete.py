import torch
from torch import Tensor


def complete_graph(num_nodes: int) -> Tensor:
    pairs = [(sender, receiver) for sender in range(num_nodes) for receiver in range(num_nodes) if sender != receiver]
    return torch.tensor(pairs).t()
