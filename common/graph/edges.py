from torch import Tensor

from common.enums.friend import Friend


def messages_into(receiver: Friend, edge_index: Tensor, values: Tensor) -> list[tuple[Friend, Tensor]]:
    senders, receivers = edge_index
    into_receiver = receivers == receiver
    return [
        (Friend(sender), value)
        for sender, value in zip(senders[into_receiver].tolist(), values[into_receiver], strict=True)
    ]


def sender_label(sender: Friend, receiver: Friend) -> str:
    # A self-loop carries the receiver's own message. Every receiver in these stories is Nancy, hence "herself".
    return "herself" if sender is receiver else sender.label
