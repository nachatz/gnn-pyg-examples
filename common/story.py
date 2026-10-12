from collections.abc import Callable

import torch

from common.config import settings
from common.graph.friends import six_friends
from common.graph.models.profile import FriendGraphSpec
from common.logger import configure_logging
from common.narrator import Narrator

CHAPTER_TITLE = "__chapter_title__"


def chapter[ChapterMethod: Callable[..., None]](title: str) -> Callable[[ChapterMethod], ChapterMethod]:
    def mark_as_chapter(method: ChapterMethod) -> ChapterMethod:
        setattr(method, CHAPTER_TITLE, title)
        return method

    return mark_as_chapter


class Story:
    """A sequence of chapters told on the six friends.

    Every chapter can rely on:

    - `self.spec`: who the friends are, their tastes, and who is friends with whom.
    - `self.x`: each friend's tastes, one row per `Friend` and one column per `Taste`.
    - `self.edge_index`: every friendship as two directed edges, sender in row 0 and receiver in row 1.
    - `self.generator`: a seeded random number generator, so shuffles repeat exactly.
    - `self.narrator`: prints numbered sections, tables and facts.

    Subclasses mark each chapter with `@chapter("title")`; chapters are told in the order they are defined.
    """

    def __init__(self, spec: FriendGraphSpec) -> None:
        self.spec = spec
        self.x = spec.x
        self.edge_index = spec.edge_index
        self.generator = torch.Generator().manual_seed(settings.seed)
        self.narrator = Narrator()

    @classmethod
    def main(cls) -> None:
        configure_logging()
        cls(six_friends()).tell()

    def tell(self) -> None:
        # Layer weights are initialised from torch's global generator, so seed it too.
        torch.manual_seed(settings.seed)
        for title, tell_chapter in self.chapters():
            self.narrator.section(title)
            tell_chapter()

    def chapters(self) -> list[tuple[str, Callable[[], None]]]:
        return [
            (getattr(member, CHAPTER_TITLE), getattr(self, name))
            for name, member in vars(type(self)).items()
            if hasattr(member, CHAPTER_TITLE)
        ]
