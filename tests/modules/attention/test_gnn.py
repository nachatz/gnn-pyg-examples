import pytest
import torch
from torch import Tensor

from common.enums.friend import Friend
from common.enums.taste import Taste
from common.graph.edges import messages_into
from modules.attention.gnn import HandTasteAttention
from modules.attention.gnn import TasteAttention
from tests.utils import GNNTestBase


class TestTasteAttention(GNNTestBase):
    @pytest.mark.parametrize(
        ("taste", "expected"),
        [
            (Taste.FOOD, {Friend.JOE: 0.111, Friend.BILL: 0.668, Friend.MEI: 0.111, Friend.NANCY: 0.111}),
            (Taste.MUSIC, {Friend.JOE: 0.551, Friend.BILL: 0.091, Friend.MEI: 0.247, Friend.NANCY: 0.111}),
        ],
    )
    def test_nancys_shares_follow_the_question(
        self, x: Tensor, edge_index: Tensor, taste: Taste, expected: dict[Friend, float]
    ) -> None:
        _, (edge_index_with_self_loops, attention) = TasteAttention(taste)(
            x,
            edge_index,
            return_attention_weights=True,
        )
        into_nancy = messages_into(Friend.NANCY, edge_index_with_self_loops, attention[:, 0])
        assert {sender: round(share.item(), 3) for sender, share in into_nancy} == expected

    def test_food_question_pulls_nancy_toward_food(self, x: Tensor, edge_index: Tensor) -> None:
        nancy = TasteAttention(Taste.FOOD)(x, edge_index)[Friend.NANCY]
        assert torch.allclose(nancy, torch.tensor([0.166, 0.233, 0.602]), atol=1e-3)


class TestHandTasteAttention(GNNTestBase):
    @pytest.mark.parametrize("taste", list(Taste))
    def test_shares_match_gatconv(self, x: Tensor, edge_index: Tensor, taste: Taste) -> None:
        _, (gat_edge_index, attention) = TasteAttention(taste)(x, edge_index, return_attention_weights=True)
        hand_edge_index, hand_shares = HandTasteAttention(taste).shares(x, edge_index)
        assert torch.equal(hand_edge_index, gat_edge_index)
        assert torch.allclose(hand_shares, attention[:, 0])

    @pytest.mark.parametrize("taste", list(Taste))
    def test_output_matches_gatconv(self, x: Tensor, edge_index: Tensor, taste: Taste) -> None:
        with torch.no_grad():
            expected = TasteAttention(taste)(x, edge_index)
        assert torch.allclose(HandTasteAttention(taste)(x, edge_index), expected)

    def test_each_nodes_shares_add_up_to_one(self, x: Tensor, edge_index: Tensor) -> None:
        edge_index_with_self_loops, shares = HandTasteAttention(Taste.FOOD).shares(x, edge_index)
        totals = torch.zeros(len(Friend)).index_add_(0, edge_index_with_self_loops[1], shares)
        assert torch.allclose(totals, torch.ones(len(Friend)))
