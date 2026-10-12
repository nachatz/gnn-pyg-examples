import torch
import torch.nn.functional as F
from torch import Tensor

from common.config import FoodieSettings
from modules.message_passing.gnn import FoodieGNN


class FoodieTrainer:
    def __init__(self, model: FoodieGNN, settings: FoodieSettings) -> None:
        self._model = model
        self._settings = settings

    def fit(self, x: Tensor, edge_index: Tensor, is_foodie: Tensor) -> list[float]:
        optimizer = self._settings.optimizer(self._model.parameters())
        losses: list[float] = []

        self._model.train()
        for _ in range(self._settings.epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(self._model(x, edge_index), is_foodie.long())
            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._settings.grad_clip)
            optimizer.step()
            losses.append(loss.item())
        return losses

    def foodie_probability(self, x: Tensor, edge_index: Tensor) -> Tensor:
        self._model.eval()
        with torch.no_grad():
            # Column 1 is the "foodie" class, because fit() trains on is_foodie as 0/1 labels.
            return self._model(x, edge_index).softmax(dim=1)[:, 1]
