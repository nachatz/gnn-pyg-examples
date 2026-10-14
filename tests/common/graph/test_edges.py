import torch

from common.enums.friend import Friend
from common.graph.edges import messages_into
from common.graph.edges import sender_label


class TestMessagesInto:
    def test_keeps_only_messages_arriving_at_the_receiver(self) -> None:
        edge_index = torch.tensor([[Friend.JOE, Friend.BILL, Friend.NANCY], [Friend.NANCY, Friend.OMAR, Friend.NANCY]])
        values = torch.tensor([0.1, 0.2, 0.3])

        incoming = messages_into(Friend.NANCY, edge_index, values)

        assert [(sender, round(value.item(), 1)) for sender, value in incoming] == [
            (Friend.JOE, 0.1),
            (Friend.NANCY, 0.3),
        ]


class TestSenderLabel:
    def test_names_the_sender_or_the_self_loop(self) -> None:
        assert sender_label(Friend.JOE, Friend.NANCY) == "Joe"
        assert sender_label(Friend.NANCY, Friend.NANCY) == "herself"
