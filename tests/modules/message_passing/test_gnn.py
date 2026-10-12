import torch
from torch import Tensor

from common.enums.friend import Friend
from common.enums.taste import Taste
from modules.message_passing.gnn import DeepMeanGNN
from modules.message_passing.gnn import FoodieGNN
from modules.message_passing.gnn import FrozenSAGEConv
from modules.message_passing.gnn import MeanMessagePassing
from tests.utils import GNNTestBase


class TestMeanMessagePassing(GNNTestBase):
    def test_nancy_blends_herself_with_the_mean_of_her_friends(self, x: Tensor, edge_index: Tensor) -> None:
        out = MeanMessagePassing()(x, edge_index)
        assert torch.allclose(out[Friend.NANCY], torch.tensor([0.2833, 0.5667, 0.15]), atol=1e-4)

    def test_matches_frozen_sage_conv(self, x: Tensor, edge_index: Tensor) -> None:
        with torch.no_grad():
            for weight in (0.0, 0.3, 0.5):
                expected = FrozenSAGEConv(weight)(x, edge_index)
                assert torch.allclose(MeanMessagePassing(weight)(x, edge_index), expected)


class TestFoodieGNN(GNNTestBase):
    def test_outputs_two_scores_per_friend(self, x: Tensor, edge_index: Tensor) -> None:
        assert FoodieGNN(hidden=4)(x, edge_index).shape == (len(Friend), 2)


class TestDeepMeanGNN(GNNTestBase):
    def test_returns_every_layer(self, x: Tensor, edge_index: Tensor) -> None:
        assert len(DeepMeanGNN(depth=3)(x, edge_index)) == 3

    def test_two_layers_carry_bill_to_joe(self, x: Tensor, edge_index: Tensor) -> None:
        one_hop, two_hop = DeepMeanGNN(depth=2)(x, edge_index)
        assert one_hop[Friend.JOE, Taste.FOOD] == 0
        assert torch.isclose(two_hop[Friend.JOE, Taste.FOOD], torch.tensor(0.075))

    def test_many_layers_over_smooth(self, x: Tensor, edge_index: Tensor) -> None:
        last = DeepMeanGNN(depth=30)(x, edge_index)[-1]
        assert torch.cdist(last, last).max() < 1e-3
