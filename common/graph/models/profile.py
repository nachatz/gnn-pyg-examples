from typing import Annotated
from typing import Self

import torch
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from torch import Tensor
from torch_geometric.utils import to_undirected

from common.enums.friend import Friend

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


class Tastes(BaseModel):
    music: UnitFloat
    outdoors: UnitFloat
    food: UnitFloat

    def as_list(self) -> list[float]:
        return [self.music, self.outdoors, self.food]


class Friendship(BaseModel):
    first: Friend
    second: Friend


class FriendGraphSpec(BaseModel):
    profiles: dict[Friend, Tastes]
    friendships: list[Friendship]

    @model_validator(mode="after")
    def _every_friend_has_a_profile(self) -> Self:
        missing = set(Friend) - set(self.profiles)
        if missing:
            raise ValueError(f"missing profiles for {sorted(f.label for f in missing)}")
        return self

    def neighbours(self, friend: Friend) -> list[Friend]:
        neighbours: list[Friend] = []
        for pair in self.friendships:
            if pair.first == friend:
                neighbours.append(pair.second)
            if pair.second == friend:
                neighbours.append(pair.first)
        return neighbours

    @property
    def x(self) -> Tensor:
        return torch.tensor([self.profiles[friend].as_list() for friend in Friend])

    @property
    def edge_index(self) -> Tensor:
        one_way_edges = torch.tensor([[pair.first, pair.second] for pair in self.friendships]).t()
        return to_undirected(one_way_edges, num_nodes=len(Friend))
