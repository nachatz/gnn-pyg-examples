from collections.abc import Callable

import torch
from jaxtyping import Float
from jaxtyping import Int
from torch import Tensor
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import SAGEConv


class MeanMessagePassing(MessagePassing):
    """One round of message passing: send -> gather (mean) -> update, with no learned weights.

        new_i = self_weight * x_i + (1 - self_weight) * mean(x_j for every neighbour j of i)

    PyG splits a layer into three hooks, and each one is a step of the round:

    - send: `message(x_j)` runs once per edge. `x_j` holds the sender's features, one row per edge, and each friend
      sends their tastes unchanged.
    - gather: `aggr="mean"` tells PyG to average every message arriving at a node.
    - update: `update(mean_of_neighbours, x)` blends the gathered mean with the node's own vector `x`.

    `forward` calls `propagate()`, which runs message -> aggregate -> update for every node at once.
    Shapes: N nodes, E directed edges, D features per node.
    """

    def __init__(self, self_weight: float = 0.5) -> None:
        super().__init__(aggr="mean")
        self.self_weight = self_weight

    def forward(self, x: Float[Tensor, "N D"], edge_index: Int[Tensor, "2 E"]) -> Float[Tensor, "N D"]:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: Float[Tensor, "E D"]) -> Float[Tensor, "E D"]:
        return x_j

    def update(self, mean_of_neighbours: Float[Tensor, "N D"], x: Float[Tensor, "N D"]) -> Float[Tensor, "N D"]:
        return self.self_weight * x + (1 - self.self_weight) * mean_of_neighbours


class FrozenSAGEConv(SAGEConv):
    """MeanMessagePassing's rule written as PyG's learnable SAGEConv, with its weights frozen by hand.

    SAGEConv computes

        new_i = W_self * x_i + W_neigh * mean(x_j) + b

    where `lin_r` is W_self (applied to the node itself) and `lin_l` is W_neigh plus the bias b (applied to the gathered
    neighbours). Normally both weights are learned. Setting W_self = self_weight * I, W_neigh = (1 - self_weight) * I
    and b = 0 recovers MeanMessagePassing exactly, which shows that a "real" GNN layer is the same send-gather-update
    round with trainable weights.
    """

    def __init__(self, self_weight: float = 0.5) -> None:
        super().__init__(3, 3, aggr="mean")
        with torch.no_grad():
            self.lin_l.weight.copy_((1 - self_weight) * torch.eye(3))
            self.lin_l.bias.zero_()
            self.lin_r.weight.copy_(self_weight * torch.eye(3))


class FoodieGNN(nn.Module):
    """A tiny learnable GNN that scores each friend as {not a foodie, foodie}.

        tastes (3) -> SAGEConv -> ReLU -> SAGEConv -> scores (2)

    Each SAGEConv layer is one round of message passing. After layer 1 a friend has heard from their friends; after
    layer 2 they have heard from friends-of-friends. So every prediction draws on everyone within 2 hops. The ReLU
    between the layers lets the model learn non-linear combinations of what it heard.
    """

    # nn.Module types every call as returning Any; this states what calling the model returns.
    __call__: Callable[["FoodieGNN", Tensor, Tensor], Tensor]

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv(3, hidden)
        self.conv2 = SAGEConv(hidden, 2)

    def forward(self, x: Float[Tensor, "N 3"], edge_index: Int[Tensor, "2 E"]) -> Float[Tensor, "N 2"]:
        heard_from_friends = torch.relu(self.conv1(x, edge_index))
        return self.conv2(heard_from_friends, edge_index)


class DeepMeanGNN(nn.Module):
    """`depth` rounds of MeanMessagePassing, returning the vectors after every round.

    MeanMessagePassing has no weights, so stacking `depth` copies of it is the same as applying one copy `depth`
    times. Returning each round's output shows two effects of depth:

    - receptive field: after k rounds, a node has heard from everyone up to k hops away.
    - over-smoothing: every round averages each node with its neighbours. Repeated enough times, all the vectors
      converge to the same blend and the nodes can no longer be told apart.
    """

    def __init__(self, depth: int) -> None:
        super().__init__()
        self.depth = depth
        self.layer = MeanMessagePassing()

    def forward(self, x: Float[Tensor, "N D"], edge_index: Int[Tensor, "2 E"]) -> list[Float[Tensor, "N D"]]:
        vectors_after_each_round: list[Tensor] = []
        for _ in range(self.depth):
            x = self.layer(x, edge_index)
            vectors_after_each_round.append(x)
        return vectors_after_each_round
