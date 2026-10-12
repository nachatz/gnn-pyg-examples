import pytest
import torch
from pydantic import ValidationError
from torch import Tensor

from common.enums.friend import Friend
from common.graph.factories.profile import FriendGraphSpecFactory
from common.graph.factories.profile import TastesFactory
from common.graph.models.profile import FriendGraphSpec
from common.graph.models.profile import Tastes
from tests.utils import GNNTestBase


class TestTastes:
    def test_as_list_orders_music_outdoors_food(self) -> None:
        assert Tastes(music=0.1, outdoors=0.2, food=0.3).as_list() == [0.1, 0.2, 0.3]

    def test_factory_builds_values_in_unit_range(self) -> None:
        assert all(0.0 <= value <= 1.0 for value in TastesFactory.build().as_list())

    @pytest.mark.parametrize("food", [-0.1, 1.1])
    def test_rejects_values_outside_unit_range(self, food: float) -> None:
        with pytest.raises(ValidationError):
            Tastes(music=0.0, outdoors=0.0, food=food)


class TestFriendGraphSpec(GNNTestBase):
    def test_rejects_missing_profiles(self) -> None:
        with pytest.raises(ValidationError, match="Omar"):
            FriendGraphSpec(profiles={}, friendships=[])

    def test_neighbours_are_read_in_both_directions(self, spec: FriendGraphSpec) -> None:
        assert spec.neighbours(Friend.NANCY) == [Friend.JOE, Friend.BILL, Friend.MEI]
        assert spec.neighbours(Friend.JOE) == [Friend.NANCY]

    def test_x_has_one_row_of_tastes_per_friend(self, x: Tensor) -> None:
        assert x.shape == (len(Friend), 3)
        assert torch.equal(x[Friend.BILL], torch.tensor([0.0, 0.1, 0.9]))

    def test_edge_index_lists_every_friendship_in_both_directions(self, edge_index: Tensor) -> None:
        assert edge_index.shape == (2, 12)

    def test_x_works_for_any_valid_spec(self) -> None:
        assert FriendGraphSpecFactory.build().x.shape == (len(Friend), 3)
