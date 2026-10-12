import torch
from torch import Tensor

def gcn_norm(
    edge_index: Tensor,
    edge_weight: Tensor | None = None,
    num_nodes: int | None = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]: ...
