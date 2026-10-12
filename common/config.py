from collections.abc import Iterable

from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
from torch import nn
from torch import optim


class FoodieSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FOODIE_")

    hidden: int = 8
    lr: float = 0.05
    epochs: int = 60
    grad_clip: float = 1.0
    threshold: float = 0.5

    def optimizer(self, params: Iterable[nn.Parameter]) -> optim.Optimizer:
        return optim.Adam(params, lr=self.lr)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GNN_")

    seed: int = 0
    log_level: str = "INFO"
    foodie: FoodieSettings = FoodieSettings()


settings = Settings()
