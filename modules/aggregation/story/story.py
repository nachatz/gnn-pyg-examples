import torch
from torch import Tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import degree

from common.enums.aggregation import Aggregation
from common.enums.friend import Friend
from common.enums.taste import Taste
from common.graph.star import LISTENER
from common.graph.star import star_graph
from common.narrator import Cell
from common.story import Story
from common.story import chapter
from modules.aggregation.gnn import AggregatingSAGEConv
from modules.aggregation.gnn import HandGCNConv
from modules.aggregation.gnn import IdentityGCNConv
from modules.aggregation.gnn import NeighbourhoodAggregator


class AggregationStory(Story):
    """How a node combines its neighbours' messages, and what each way forgets, told on the six friends.

    The cast, and who is friends with whom:

        Joe ── Nancy ── Bill
                 │        │
                Mei      Omar
                 │        │
               Priya ─────┘

    - Nancy is the listener in most chapters. She hears from Joe (music), Bill (food) and Mei (half music, half
      outdoors), so sum, mean and max each give her a visibly different answer.
    - Omar's taste is half music, half food; a crowd of fans who all share it shows sum growing without limit.
    - Mei is the friend with a twin: two identical Meis average to exactly one Mei.
    - Bill is the only food lover among Bill, Joe and Nancy (Joe and Nancy have no taste for food), so max hears 0.9
      food whether one Bill sends it or three.
    - Joe has one friend, Bill and Mei have two, and Nancy has three. GCN counts each friend's self-loop too, so
      the degrees into Nancy's weights are Joe 2, Bill 3, Mei 3 and Nancy 4; Joe's message weighs the most.

    The middle chapters use a star graph instead of the friends: an anonymous listener who hears only the messages
    listed in each table.

    From Story: `self.x` holds each friend's tastes (one row per `Friend`, one column per `Taste`), `self.edge_index`
    lists every friendship in both directions (senders in row 0, receivers in row 1), `self.generator` makes the shuffle
    repeatable, and `self.narrator` prints the sections and tables.
    """

    @chapter("Aggregation must ignore the order of the messages")
    def order_does_not_matter(self) -> None:
        self._show_shuffling_changes_nothing(Friend.NANCY)
        self._show_concatenation_depends_on_order()

    @chapter("Same messages, different answers: SAGEConv with sum, mean and max")
    def same_messages_different_answers(self) -> None:
        self.narrator.table(
            ["aggr", "what Nancy hears from Joe, Bill and Mei"],
            [[aggr, AggregatingSAGEConv(aggr)(self.x, self.edge_index)[Friend.NANCY]] for aggr in Aggregation],
        )

    @chapter("Sum grows with the crowd: Omar's taste arrives from 1, 10, then 1000 fans")
    def sum_grows_with_the_crowd(self) -> None:
        crowds = [(size, [self.x[Friend.OMAR]] * size) for size in (1, 10, 1000)]
        self.narrator.table(
            ["fans", "sum", "mean"],
            [[size, self._hear(fans, Aggregation.SUM), self._hear(fans, Aggregation.MEAN)] for size, fans in crowds],
        )
        self.narrator.note("Sum grows with the number of fans; mean stays put.")

    @chapter("Mean can't tell 1 friend from 2 identical friends")
    def mean_misses_a_twin(self) -> None:
        mei = self.x[Friend.MEI]
        senders = (("Mei", [mei]), ("Mei and her twin", [mei, mei]))
        self.narrator.table(
            ["messages from", "mean", "sum"],
            [
                [label, self._hear(messages, Aggregation.MEAN), self._hear(messages, Aggregation.SUM)]
                for label, messages in senders
            ],
        )

    @chapter("Max can't tell 1 food lover from 3")
    def max_misses_extra_foodies(self) -> None:
        bill, joe, nancy = self.x[Friend.BILL], self.x[Friend.JOE], self.x[Friend.NANCY]
        senders = (("Bill, Joe and Nancy", [bill, joe, nancy]), ("three Bills", [bill] * 3))
        self.narrator.table(
            ["messages from", "max food", "sum food"],
            [
                [label, self._food_heard(messages, Aggregation.MAX), self._food_heard(messages, Aggregation.SUM)]
                for label, messages in senders
            ],
        )

    @chapter("GCN normalisation: weight each message by 1 / sqrt(deg_i * deg_j)")
    def gcn_normalisation(self) -> None:
        edge_index_with_self_loops, weights = HandGCNConv.normalise(self.edge_index, len(Friend))
        self._show_weights_into(Friend.NANCY, edge_index_with_self_loops, weights)
        self._show_same_weights_as_pyg(edge_index_with_self_loops, weights)
        self._show_gcn_conv_matches_hand_gcn_conv(Friend.NANCY)

    def _show_shuffling_changes_nothing(self, listener: Friend) -> None:
        shuffle = torch.randperm(self.edge_index.size(1), generator=self.generator)
        shuffled_edge_index = self.edge_index[:, shuffle]

        rows: list[list[Cell]] = []
        for aggr in Aggregation:
            aggregator = NeighbourhoodAggregator(aggr)
            heard_in_order = aggregator(self.x, self.edge_index)[listener]
            heard_shuffled = aggregator(self.x, shuffled_edge_index)[listener]
            rows.append([aggr, heard_in_order, heard_shuffled, torch.allclose(heard_in_order, heard_shuffled)])
        self.narrator.table(
            ["aggr", f"{listener.label}, original order", f"{listener.label}, shuffled order", "the same in both?"],
            rows,
        )

    def _show_concatenation_depends_on_order(self) -> None:
        as_list = torch.cat([self.x[Friend.JOE], self.x[Friend.BILL], self.x[Friend.MEI]])
        as_list_reversed = torch.cat([self.x[Friend.MEI], self.x[Friend.BILL], self.x[Friend.JOE]])
        self.narrator.fact(
            "concatenated instead, is [Joe, Bill, Mei] the same as [Mei, Bill, Joe]?",
            torch.allclose(as_list, as_list_reversed),
        )

    def _show_weights_into(self, receiver: Friend, edge_index_with_self_loops: Tensor, weights: Tensor) -> None:
        senders, receivers = edge_index_with_self_loops
        # Counted after the self-loops are added, so every friend's degree includes themselves.
        degrees = degree(receivers, num_nodes=len(Friend)).long().tolist()
        into_receiver = receivers == receiver
        senders_into_receiver = senders[into_receiver].tolist()

        self.narrator.table(
            ["message from", "formula", f"weight into {receiver.label}"],
            [
                self._weight_row(Friend(sender), receiver, degrees, weight)
                for sender, weight in zip(senders_into_receiver, weights[into_receiver], strict=True)
            ],
        )

    def _show_same_weights_as_pyg(self, edge_index_with_self_loops: Tensor, weights: Tensor) -> None:
        pyg_edge_index, pyg_weights = gcn_norm(self.edge_index, num_nodes=len(Friend))
        same_edges = torch.equal(edge_index_with_self_loops, pyg_edge_index)
        self.narrator.fact(
            "HandGCNConv's weights the same as PyG's gcn_norm?",
            same_edges and torch.allclose(weights, pyg_weights),
        )

    def _show_gcn_conv_matches_hand_gcn_conv(self, friend: Friend) -> None:
        by_hand = HandGCNConv()(self.x, self.edge_index)[friend]
        by_pyg = IdentityGCNConv()(self.x, self.edge_index)[friend]
        self.narrator.table(
            ["layer", f"{friend.label}'s new vector"],
            [["HandGCNConv", by_hand], ["GCNConv", by_pyg]],
        )
        self.narrator.fact("GCNConv's output the same as HandGCNConv's?", torch.allclose(by_pyg, by_hand))

    @classmethod
    def _weight_row(cls, sender: Friend, receiver: Friend, degrees: list[int], weight: Tensor) -> list[Cell]:
        message_from = "herself" if sender is receiver else sender.label
        return [message_from, f"1/sqrt({degrees[sender]}*{degrees[receiver]})", weight]

    @classmethod
    def _food_heard(cls, messages: list[Tensor], aggr: Aggregation) -> Tensor:
        return cls._hear(messages, aggr)[Taste.FOOD]

    @classmethod
    def _hear(cls, messages: list[Tensor], aggr: Aggregation) -> Tensor:
        x, edge_index = star_graph(messages)
        return NeighbourhoodAggregator(aggr)(x, edge_index)[LISTENER]
