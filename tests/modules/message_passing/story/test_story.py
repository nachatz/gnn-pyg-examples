import pytest

from modules.message_passing.story.story import MessagePassingStory


class TestMessagePassingStory:
    def test_tells_the_whole_story(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO", logger="gnn_pyg_examples"):
            MessagePassingStory.main()

        assert "   her new vector (half her old self + half the mean): [0.283, 0.567, 0.150]" in caplog.messages
        assert "   MeanMessagePassing  [0.283, 0.567, 0.150]  yes" in caplog.messages
        assert "   FrozenSAGEConv      [0.283, 0.567, 0.150]  yes" in caplog.messages
        assert "   after 2 layers (hears Bill via Nancy)  0.075" in caplog.messages
        assert sum(message.endswith("foodie        foodie") for message in caplog.messages) == 3
        assert "   30      0.0003" in caplog.messages
