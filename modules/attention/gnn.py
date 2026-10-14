import torch
import torch.nn.functional as F
from jaxtyping import Float
from jaxtyping import Int
from torch import Tensor
from torch_geometric.nn import GATConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import softmax

from common.enums.taste import Taste


class TasteAttention(GATConv):
    """PyG's GATConv set up to ask one readable question: how much is each message about `taste`?

    Graph attention replaces fixed edge weights with learned ones. For every message j -> i, GAT computes a score

        score_ij = LeakyReLU(att_src · W x_j + att_dst · W x_i)

    then turns the scores arriving at each node into shares with a softmax, so every node's shares add up to 1. The node
    hears the share-weighted sum of its neighbours' vectors, plus its own through a self-loop.

    Normally W, att_src and att_dst are all learned. Here W is the identity, att_dst is zero (so the score ignores the
    listener), and att_src is `strength` on the `taste` feature only. Each score is then simply

        score_ij = LeakyReLU(strength * taste of j)

    so the friend whose message is most about `taste` gets the largest share.
    """

    def __init__(self, taste: Taste, strength: float = 2.0) -> None:
        super().__init__(3, 3, heads=1, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(torch.eye(3))
            self.att_dst.zero_()
            self.att_src.zero_()
            self.att_src[0, 0, taste] = strength


class HandTasteAttention(MessagePassing):
    """TasteAttention's arithmetic written out by hand, so every step of attention stays visible.

    `shares` does the work in four steps:

    1. Add a self-loop to every node, so each node also hears its own message.
    2. Score each message by how much the sender cares about `taste`: strength * x_j[taste].
    3. Pass each score through LeakyReLU, as GAT does (it leaves positive scores unchanged).
    4. Softmax the scores arriving at each node, so each node's shares add up to 1.

    `message` scales each sender's vector by its share, and `aggr="add"` sums them. Unlike GCN's fixed weights, the
    shares depend on what each message says, so asking about a different taste moves the attention elsewhere.
    """

    def __init__(self, taste: Taste, strength: float = 2.0, negative_slope: float = 0.2) -> None:
        super().__init__(aggr="add")
        self.taste = taste
        self.strength = strength
        self.negative_slope = negative_slope

    def shares(
        self, x: Float[Tensor, "N D"], edge_index: Int[Tensor, "2 E"]
    ) -> tuple[Int[Tensor, "2 E_loops"], Float[Tensor, " E_loops"]]:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        senders, receivers = edge_index
        scores = F.leaky_relu(self.strength * x[senders, self.taste], self.negative_slope)
        return edge_index, softmax(scores, receivers, num_nodes=x.size(0))

    def forward(self, x: Float[Tensor, "N D"], edge_index: Int[Tensor, "2 E"]) -> Float[Tensor, "N D"]:
        edge_index_with_self_loops, shares = self.shares(x, edge_index)
        return self.propagate(edge_index_with_self_loops, x=x, share=shares)

    def message(self, x_j: Float[Tensor, "E D"], share: Float[Tensor, " E"]) -> Float[Tensor, "E D"]:
        return share.view(-1, 1) * x_j
