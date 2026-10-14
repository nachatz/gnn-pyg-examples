import torch
from torch import Tensor
from torch_geometric.nn import GATConv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from common.enums.friend import Friend
from common.enums.taste import Taste
from common.graph.complete import complete_graph
from common.graph.edges import messages_into
from common.graph.edges import sender_label
from common.narrator import Cell
from common.story import Story
from common.story import chapter
from modules.attention.gnn import HandTasteAttention
from modules.attention.gnn import TasteAttention


class AttentionStory(Story):
    """How graph attention lets each node learn how much to listen to each neighbour, told on the six friends.

    The cast, and who is friends with whom:

        Joe ── Nancy ── Bill
                 │        │
                Mei      Omar
                 │        │
               Priya ─────┘

    - Nancy is the listener. She hears from Joe, Bill and Mei, plus herself through a self-loop.
    - GCN's fixed weights only count friends: Joe has one friend, Bill and Mei two, Nancy three, so with self-loops
      Joe's message weighs most (0.354) whatever anyone says.
    - Bill is the only one of them whose message is about food (0.9), so asking about food sends most of Nancy's
      attention to Bill.
    - Joe loves music most (0.9), Mei likes it moderately (0.5), Nancy barely (0.1) and Bill not at all, so asking
      about music moves Nancy's attention from Bill to Joe.

    The last chapter swaps the friendships for a complete graph, where every friend hears every other friend.

    From Story: `self.x` holds each friend's tastes (one row per `Friend`, one column per `Taste`), `self.edge_index`
    lists every friendship in both directions (senders in row 0, receivers in row 1), and `self.narrator` prints the
    sections and tables.
    """

    @chapter("Fixed weights: GCN decides how much Nancy hears from each friend")
    def fixed_weights(self) -> None:
        edge_index_with_self_loops, weights = gcn_norm(self.edge_index, num_nodes=len(Friend))
        self._show_shares_into(Friend.NANCY, edge_index_with_self_loops, weights, "GCN weight")
        self.narrator.note("These weights only count friends; they never look at what anyone says.")

    @chapter("Attention asks a question: how much is each message about food?")
    def attention_on_food(self) -> None:
        self._show_attention(Friend.NANCY, Taste.FOOD)

    @chapter("A different question: how much is each message about music?")
    def attention_on_music(self) -> None:
        self._show_attention(Friend.NANCY, Taste.MUSIC)
        self.narrator.note("Same friends, same messages, different shares.")

    @chapter("Multi-head attention: four heads ask four questions at once")
    def multi_head(self) -> None:
        self.narrator.table(
            ["GATConv(3, 16, heads=4)", "output shape", "attention shape"],
            [
                self._shape_row("concat=True", GATConv(3, 16, heads=4), self.edge_index),
                self._shape_row("concat=False", GATConv(3, 16, heads=4, concat=False), self.edge_index),
            ],
        )
        self.narrator.note("18 attention rows = 12 messages + 6 self-loops, with one column per head.")
        self.narrator.note("concat=True joins the 4 heads into 4 x 16 = 64 features; concat=False averages them.")

    @chapter("Everyone hears everyone: the attention pattern of a Transformer")
    def everyone_hears_everyone(self) -> None:
        # In a complete graph every friend attends to every other friend, just as every token attends to every token
        # in a Transformer. Transformers score with dot products (query · key); PyG's TransformerConv uses that style.
        everyone = complete_graph(len(Friend))
        self.narrator.table(
            ["heads=4 on the complete graph", "output shape", "attention shape"],
            [
                self._shape_row("GATConv", GATConv(3, 16, heads=4), everyone),
                self._shape_row("TransformerConv", TransformerConv(3, 16, heads=4), everyone),
            ],
        )
        self.narrator.note("30 messages: each of the 6 friends hears the other 5.")
        self.narrator.note("GATConv adds 6 self-loops; TransformerConv keeps each friend's own vector separately.")

    def _show_attention(self, listener: Friend, taste: Taste) -> None:
        heard, (edge_index_with_self_loops, attention) = TasteAttention(taste)(
            self.x,
            self.edge_index,
            return_attention_weights=True,
        )
        shares = attention[:, 0]  # a single head
        self._show_shares_into(listener, edge_index_with_self_loops, shares, "attention share")
        self._show_same_shares_as_by_hand(taste, edge_index_with_self_loops, shares)
        self.narrator.fact(f"{listener.label}'s new vector", heard[listener])

    def _show_shares_into(self, receiver: Friend, edge_index: Tensor, shares: Tensor, column: str) -> None:
        largest_first = sorted(messages_into(receiver, edge_index, shares), key=lambda message: -message[1].item())
        self.narrator.table(
            ["message from", f"{column} into {receiver.label}"],
            [[sender_label(sender, receiver), share] for sender, share in largest_first],
        )

    def _show_same_shares_as_by_hand(self, taste: Taste, edge_index: Tensor, shares: Tensor) -> None:
        hand_edge_index, hand_shares = HandTasteAttention(taste).shares(self.x, self.edge_index)
        same_edges = torch.equal(edge_index, hand_edge_index)
        self.narrator.fact(
            "GATConv's shares the same as by hand?",
            same_edges and torch.allclose(shares, hand_shares),
        )

    def _shape_row(self, label: str, layer: GATConv | TransformerConv, edge_index: Tensor) -> list[Cell]:
        output, (_, attention) = layer(self.x, edge_index, return_attention_weights=True)
        return [label, self._shape(output), self._shape(attention)]

    @classmethod
    def _shape(cls, tensor: Tensor) -> str:
        return str(list(tensor.shape))
