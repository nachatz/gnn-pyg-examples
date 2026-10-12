import torch

from common.graph.star import LISTENER
from common.graph.star import star_graph


class TestStarGraph:
    def test_every_message_points_at_the_listener(self) -> None:
        x, edge_index = star_graph([torch.ones(3), torch.full((3,), 2.0)])

        assert torch.equal(x[LISTENER], torch.zeros(3))
        assert torch.equal(edge_index, torch.tensor([[1, 2], [LISTENER, LISTENER]]))
