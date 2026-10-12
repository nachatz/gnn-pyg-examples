from typing import Any

from torch import Tensor
from torch import nn
from torch_geometric.nn.dense.linear import Linear

class MessagePassing(nn.Module):
    def __init__(self, aggr: str | list[str] | None = "sum") -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    def propagate(self, edge_index: Tensor, size: tuple[int, int] | None = None, **kwargs: Any) -> Tensor: ...

class SAGEConv(MessagePassing):
    lin_l: Linear
    lin_r: Linear
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str | list[str] | None = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
    ) -> None: ...
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor: ...

class GCNConv(MessagePassing):
    lin: Linear
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool | None = None,
        normalize: bool = True,
        bias: bool = True,
    ) -> None: ...
    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None) -> Tensor: ...
