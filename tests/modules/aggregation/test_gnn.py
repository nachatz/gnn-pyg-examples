import pytest
import torch
from torch import Tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from common.enums.aggregation import Aggregation
from common.enums.friend import Friend
from modules.aggregation.gnn import AggregatingSAGEConv
from modules.aggregation.gnn import HandGCNConv
from modules.aggregation.gnn import IdentityGCNConv
from modules.aggregation.gnn import NeighbourhoodAggregator
from tests.utils import GNNTestBase


class TestNeighbourhoodAggregator(GNNTestBase):
    @pytest.mark.parametrize(
        ("aggr", "expected"),
        [
            (Aggregation.SUM, [1.4, 0.7, 0.9]),
            (Aggregation.MEAN, [0.4667, 0.2333, 0.3]),
            (Aggregation.MAX, [0.9, 0.5, 0.9]),
        ],
    )
    def test_nancy_hears(self, x: Tensor, edge_index: Tensor, aggr: Aggregation, expected: list[float]) -> None:
        out = NeighbourhoodAggregator(aggr)(x, edge_index)[Friend.NANCY]
        assert torch.allclose(out, torch.tensor(expected), atol=1e-4)

    @pytest.mark.parametrize("aggr", list(Aggregation))
    def test_ignores_message_order(self, x: Tensor, edge_index: Tensor, aggr: Aggregation) -> None:
        shuffled = edge_index[:, torch.randperm(edge_index.size(1))]
        layer = NeighbourhoodAggregator(aggr)
        assert torch.equal(layer(x, edge_index), layer(x, shuffled))

    @pytest.mark.parametrize("aggr", list(Aggregation))
    def test_matches_reduced_sage_conv(self, x: Tensor, edge_index: Tensor, aggr: Aggregation) -> None:
        with torch.no_grad():
            expected = AggregatingSAGEConv(aggr)(x, edge_index)
        assert torch.allclose(NeighbourhoodAggregator(aggr)(x, edge_index), expected)


class TestHandGCNConv(GNNTestBase):
    def test_weights_into_nancy(self, x: Tensor, edge_index: Tensor) -> None:
        edge_index, weight = HandGCNConv.normalise(edge_index, len(Friend))
        into_nancy = {
            int(src): round(float(w), 3) for (src, dst), w in zip(edge_index.t(), weight) if dst == Friend.NANCY
        }
        assert into_nancy == {Friend.JOE: 0.354, Friend.BILL: 0.289, Friend.MEI: 0.289, Friend.NANCY: 0.25}

    def test_weights_match_pyg_gcn_norm(self, x: Tensor, edge_index: Tensor) -> None:
        hand = HandGCNConv.normalise(edge_index, len(Friend))
        pyg = gcn_norm(edge_index, num_nodes=len(Friend), add_self_loops=True)
        assert torch.equal(hand[0], pyg[0])
        assert torch.allclose(hand[1], pyg[1])

    def test_output_matches_gcn_conv(self, x: Tensor, edge_index: Tensor) -> None:
        with torch.no_grad():
            expected = IdentityGCNConv()(x, edge_index)
        assert torch.allclose(HandGCNConv()(x, edge_index), expected)
