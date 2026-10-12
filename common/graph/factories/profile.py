from polyfactory import Use
from polyfactory.factories.pydantic_factory import ModelFactory

from common.enums.friend import Friend
from common.graph.models.profile import FriendGraphSpec
from common.graph.models.profile import Friendship
from common.graph.models.profile import Tastes


class TastesFactory(ModelFactory[Tastes]):
    __model__ = Tastes


class FriendshipFactory(ModelFactory[Friendship]):
    __model__ = Friendship


class FriendGraphSpecFactory(ModelFactory[FriendGraphSpec]):
    __model__ = FriendGraphSpec

    profiles = Use(lambda: {friend: TastesFactory.build() for friend in Friend})
    friendships = Use(lambda: FriendshipFactory.batch(size=4))
