from torch import Tensor

from common.config import FoodieSettings
from common.enums.taste import Taste
from modules.message_passing.gnn import FoodieGNN
from modules.message_passing.story.trainer import FoodieTrainer
from tests.utils import GNNTestBase


class TestFoodieTrainer(GNNTestBase):
    def test_learns_who_is_a_foodie(self, x: Tensor, edge_index: Tensor) -> None:
        foodie_settings = FoodieSettings()
        is_foodie = x[:, Taste.FOOD] >= foodie_settings.threshold
        trainer = FoodieTrainer(FoodieGNN(foodie_settings.hidden), foodie_settings)

        losses = trainer.fit(x, edge_index, is_foodie)

        assert len(losses) == foodie_settings.epochs
        assert losses[-1] < losses[0] / 10
        assert ((trainer.foodie_probability(x, edge_index) > 0.5) == is_foodie).all()
