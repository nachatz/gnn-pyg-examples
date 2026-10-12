from collections.abc import Generator

import pytest
import torch
from torch import Tensor

from common.graph.friends import six_friends
from common.graph.models.profile import FriendGraphSpec


class GNNTestBase:
    @pytest.fixture(autouse=True)
    def seeded(self) -> Generator[None]:
        torch.manual_seed(0)
        yield

    @pytest.fixture
    def spec(self) -> FriendGraphSpec:
        return six_friends()

    @pytest.fixture
    def x(self, spec: FriendGraphSpec) -> Tensor:
        return spec.x

    @pytest.fixture
    def edge_index(self, spec: FriendGraphSpec) -> Tensor:
        return spec.edge_index
