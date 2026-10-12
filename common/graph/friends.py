from common.enums.friend import Friend
from common.graph.models.profile import FriendGraphSpec
from common.graph.models.profile import Friendship
from common.graph.models.profile import Tastes


def six_friends() -> FriendGraphSpec:
    return FriendGraphSpec(
        profiles={
            Friend.JOE: Tastes(music=0.9, outdoors=0.1, food=0.0),
            Friend.NANCY: Tastes(music=0.1, outdoors=0.9, food=0.0),
            Friend.BILL: Tastes(music=0.0, outdoors=0.1, food=0.9),
            Friend.MEI: Tastes(music=0.5, outdoors=0.5, food=0.0),
            Friend.PRIYA: Tastes(music=0.0, outdoors=0.5, food=0.5),
            Friend.OMAR: Tastes(music=0.5, outdoors=0.0, food=0.5),
        },
        friendships=[
            Friendship(first=Friend.JOE, second=Friend.NANCY),
            Friendship(first=Friend.NANCY, second=Friend.BILL),
            Friendship(first=Friend.NANCY, second=Friend.MEI),
            Friendship(first=Friend.BILL, second=Friend.OMAR),
            Friendship(first=Friend.OMAR, second=Friend.PRIYA),
            Friendship(first=Friend.PRIYA, second=Friend.MEI),
        ],
    )
