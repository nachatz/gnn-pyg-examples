import pytest

from common.enums.friend import Friend
from common.story import Story
from common.story import chapter


class TestStory:
    class TwoChapterStory(Story):
        """Uses `self.x` (each friend's tastes) from Story."""

        @chapter("Who is Bill?")
        def introduce_bill(self) -> None:
            self.narrator.fact("Bill", self.x[Friend.BILL])

        def not_a_chapter(self) -> None:
            self.narrator.note("never told")

        @chapter("Who is Joe?")
        def introduce_joe(self) -> None:
            self.narrator.fact("Joe", self.x[Friend.JOE])

    def test_main_tells_marked_chapters_in_definition_order(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO", logger="gnn_pyg_examples"):
            self.TwoChapterStory.main()

        assert caplog.messages == [
            "1. Who is Bill?",
            "   Bill: [0.000, 0.100, 0.900]",
            "",
            "2. Who is Joe?",
            "   Joe: [0.900, 0.100, 0.000]",
        ]
