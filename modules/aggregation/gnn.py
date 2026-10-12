import torch
from jaxtyping import Float
from jaxtyping import Int
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import degree

from common.enums.aggregation import Aggregation


class NeighbourhoodAggregator(MessagePassing):
    """The gather step on its own: each node combines its neighbours' vectors with `aggr`. No weights, no update.

    Each neighbour sends its vector unchanged (`message` returns `x_j`), and the only thing that differs between sum,
    mean and max is the single `aggr` argument passed to MessagePassing.

    The aggregator must give the same answer whatever order the messages arrive in, because a node's neighbours have
    no natural order. Sum, mean and max all qualify, and each forgets something different:

    - sum keeps the number of neighbours, but grows without limit as that number rises.
    - mean stays stable, but can't tell 1 friend from 2 identical friends.
    - max keeps the strongest signal, but can't tell how many neighbours sent it.

    Shapes: N nodes, E directed edges, D features per node.
    """

    def __init__(self, aggr: Aggregation) -> None:
        super().__init__(aggr=str(aggr))

    def forward(self, x: Float[Tensor, "N D"], edge_index: Int[Tensor, "2 E"]) -> Float[Tensor, "N D"]:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: Float[Tensor, "E D"]) -> Float[Tensor, "E D"]:
        return x_j


class AggregatingSAGEConv(SAGEConv):
    """PyG's SAGEConv reduced to its aggregator, so that sum, mean and max can be compared directly.

    `root_weight=False` drops the node's own vector from the output. An identity weight with zero bias then passes the
    aggregate straight through, so the output is exactly what the node heard from its neighbours.
    """

    def __init__(self, aggr: Aggregation) -> None:
        super().__init__(3, 3, aggr=str(aggr), root_weight=False)
        with torch.no_grad():
            self.lin_l.weight.copy_(torch.eye(3))
            self.lin_l.bias.zero_()


class HandGCNConv(MessagePassing):
    """GCN's normalised sum, written out by hand (without the learnable weight, so the arithmetic stays visible).

    `normalise` does the work in three steps:

    1. Add a self-loop to every node, so each node also hears its own message.
    2. Count each node's degree, including that self-loop.
    3. Weight every message j -> i by

        w_ij = 1 / sqrt(deg_i * deg_j)

    `message` scales each neighbour's vector by its edge weight, and `aggr="add"` sums the weighted messages. A message
    from a well-connected neighbour is turned down, so no popular node drowns out the rest. Unlike a plain mean, the
    result still depends on how many neighbours each node has.
    """

    def __init__(self) -> None:
        super().__init__(aggr="add")

    @classmethod
    def normalise(
        cls, edge_index: Int[Tensor, "2 E"], num_nodes: int
    ) -> tuple[Int[Tensor, "2 E_loops"], Float[Tensor, " E_loops"]]:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        senders, receivers = edge_index
        # Counted after the self-loops are added, so every node's degree includes itself.
        degrees = degree(receivers, num_nodes=num_nodes)
        return edge_index, degrees[senders].rsqrt() * degrees[receivers].rsqrt()

    def forward(self, x: Float[Tensor, "N D"], edge_index: Int[Tensor, "2 E"]) -> Float[Tensor, "N D"]:
        edge_index_with_self_loops, weight = self.normalise(edge_index, x.size(0))
        return self.propagate(edge_index_with_self_loops, x=x, weight=weight)

    def message(self, x_j: Float[Tensor, "E D"], weight: Float[Tensor, " E"]) -> Float[Tensor, "E D"]:
        return weight.view(-1, 1) * x_j


class IdentityGCNConv(GCNConv):
    """PyG's GCNConv with an identity weight and no bias, so its output can be checked against HandGCNConv.

    GCNConv normally multiplies every node's vector by a learned weight before the normalised sum. Fixing that weight
    to the identity leaves just the normalised sum, which is exactly what HandGCNConv computes.
    """

    def __init__(self) -> None:
        super().__init__(3, 3, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(torch.eye(3))
