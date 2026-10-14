import torch

from common.graph.complete import complete_graph


class TestCompleteGraph:
    def test_every_node_hears_every_other_node_once(self) -> None:
        senders, receivers = complete_graph(3)

        assert sorted(zip(senders.tolist(), receivers.tolist(), strict=True)) == [
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
        ]
        assert not torch.any(senders == receivers)
