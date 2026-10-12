import torch
from torch import Tensor

from common.config import settings
from common.enums.friend import Friend
from common.enums.taste import Taste
from common.narrator import Cell
from common.story import Story
from common.story import chapter
from modules.message_passing.gnn import DeepMeanGNN
from modules.message_passing.gnn import FoodieGNN
from modules.message_passing.gnn import FrozenSAGEConv
from modules.message_passing.gnn import MeanMessagePassing
from modules.message_passing.story.trainer import FoodieTrainer


class MessagePassingStory(Story):
    """How one GNN layer works, told on the six friends.

    The cast, and who is friends with whom:

        Joe ── Nancy ── Bill
                 │        │
                Mei      Omar
                 │        │
               Priya ─────┘

    - Nancy is the hub: her three friends Joe, Bill and Mei are who she hears from in the first two chapters.
    - Joe loves music and has no taste for food at all; his only friend is Nancy.
    - Bill loves food more than anyone (0.9). He is two hops from Joe, via Nancy, which is how Joe picks up a taste
      for food without being Bill's friend.
    - Bill, Priya and Omar are the foodies (food taste of at least 0.5); Joe, Nancy and Mei have none.
    - Joe the music lover and Bill the foodie start furthest apart, which makes them the clearest witnesses to
      over-smoothing.

    From Story: `self.x` holds each friend's tastes (one row per `Friend`, one column per `Taste`), `self.edge_index`
    lists every friendship in both directions, `self.spec.neighbours(friend)` names a friend's friends, and
    `self.narrator` prints the sections and tables.
    """

    @chapter("One round of message passing, by hand, for Nancy")
    def one_round_by_hand(self) -> None:
        self.narrator.table(
            ["Nancy hears from", "tastes"],
            [[friend.label, self.x[friend]] for friend in self.spec.neighbours(Friend.NANCY)],
        )
        self.narrator.fact("mean of what she heard", self._mean_heard_by(Friend.NANCY))
        self.narrator.fact("her new vector (half her old self + half the mean)", self._updated_by_hand(Friend.NANCY))

    @chapter("The same round with PyG layers")
    def same_round_with_pyg(self) -> None:
        nancy_by_hand = self._updated_by_hand(Friend.NANCY)

        rows: list[list[Cell]] = []
        for layer in (MeanMessagePassing(), FrozenSAGEConv()):
            nancy_by_layer = layer(self.x, self.edge_index)[Friend.NANCY]
            rows.append([type(layer).__name__, nancy_by_layer, torch.allclose(nancy_by_layer, nancy_by_hand)])
        self.narrator.table(["layer", "Nancy's new vector", "the same as by hand?"], rows)

    @chapter("Stacking layers lets information travel further")
    def two_hops(self) -> None:
        after_one_layer, after_two_layers = DeepMeanGNN(depth=2)(self.x, self.edge_index)
        joes_food = (Friend.JOE, Taste.FOOD)

        self.narrator.table(
            ["when", "Joe's food taste"],
            [
                ["at the start", self.x[joes_food]],
                ["after 1 layer (hears Nancy)", after_one_layer[joes_food]],
                ["after 2 layers (hears Bill via Nancy)", after_two_layers[joes_food]],
            ],
        )
        self.narrator.note("Joe picked up Bill's love of food without being his friend.")

    @chapter("Training a 2-layer GNN to spot the foodies")
    def train_foodie_detector(self) -> None:
        is_foodie = self.x[:, Taste.FOOD] >= settings.foodie.threshold
        trainer = FoodieTrainer(FoodieGNN(settings.foodie.hidden), settings.foodie)

        losses = trainer.fit(self.x, self.edge_index, is_foodie)
        self._show_losses(losses)

        self._show_predictions(trainer.foodie_probability(self.x, self.edge_index), is_foodie)

    @chapter("Over-smoothing: apply the averaging layer again and again")
    def over_smoothing(self) -> None:
        depth = 30
        vectors_after_layers = [self.x, *DeepMeanGNN(depth)(self.x, self.edge_index)]

        self.narrator.table(
            ["layers", "largest distance between any two friends"],
            [[layers, self._largest_distance(vectors_after_layers[layers])] for layers in (0, 1, 2, 5, 10, depth)],
            precision=4,
        )

        after_all_layers = vectors_after_layers[depth]
        self.narrator.table(
            ["friend", f"vector after {depth} layers"],
            [[friend.label, after_all_layers[friend]] for friend in Friend],
        )
        self.narrator.note("Joe the music lover and Bill the foodie now have nearly identical vectors.")

    def _show_losses(self, losses: list[float]) -> None:
        epochs_to_show = [1, *range(15, len(losses) + 1, 15)]
        self.narrator.table(
            ["epoch", "loss"],
            [[epoch, losses[epoch - 1]] for epoch in epochs_to_show],
            precision=4,
        )

    def _show_predictions(self, foodie_probability: Tensor, is_foodie: Tensor) -> None:
        self.narrator.table(
            ["friend", "P(foodie)", "prediction", "truth"],
            [self._prediction_row(friend, foodie_probability, is_foodie) for friend in Friend],
        )

    def _prediction_row(self, friend: Friend, foodie_probability: Tensor, is_foodie: Tensor) -> list[Cell]:
        predicted_foodie = foodie_probability[friend] > 0.5
        return [
            friend.label,
            foodie_probability[friend],
            self._foodie_label(predicted_foodie),
            self._foodie_label(is_foodie[friend]),
        ]

    def _mean_heard_by(self, friend: Friend) -> Tensor:
        return self.x[self.spec.neighbours(friend)].mean(dim=0)

    def _updated_by_hand(self, friend: Friend) -> Tensor:
        return 0.5 * self.x[friend] + 0.5 * self._mean_heard_by(friend)

    @classmethod
    def _largest_distance(cls, vectors: Tensor) -> Tensor:
        return torch.cdist(vectors, vectors).max()

    @classmethod
    def _foodie_label(cls, is_foodie: Tensor) -> str:
        return "foodie" if is_foodie else "not a foodie"
