import torch
from torch import Tensor

def add_self_loops(
    edge_index: Tensor,
    edge_attr: Tensor | None = None,
    fill_value: float | Tensor | str | None = None,
    num_nodes: int | None = None,
) -> tuple[Tensor, Tensor | None]: ...
def degree(index: Tensor, num_nodes: int | None = None, dtype: torch.dtype | None = None) -> Tensor: ...

# Only the edge-index-only form is typed: without edge_attr, to_undirected returns just the edge index.
def to_undirected(edge_index: Tensor, *, num_nodes: int | None = None, reduce: str = "add") -> Tensor: ...
