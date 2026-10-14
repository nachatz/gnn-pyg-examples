from typing import Any
from typing import Literal
from typing import overload

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

class GATConv(MessagePassing):
    lin: Linear
    att_src: nn.Parameter
    att_dst: nn.Parameter
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
    ) -> None: ...
    # return_attention_weights=True also returns (edge_index with self-loops, attention of shape [E, heads]).
    @overload
    def __call__(self, x: Tensor, edge_index: Tensor) -> Tensor: ...
    @overload
    def __call__(
        self, x: Tensor, edge_index: Tensor, *, return_attention_weights: Literal[True]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]: ...

class TransformerConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        root_weight: bool = True,
    ) -> None: ...
    # return_attention_weights=True also returns (edge_index, attention of shape [E, heads]); no self-loops are added.
    @overload
    def __call__(self, x: Tensor, edge_index: Tensor) -> Tensor: ...
    @overload
    def __call__(
        self, x: Tensor, edge_index: Tensor, *, return_attention_weights: Literal[True]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]: ...
