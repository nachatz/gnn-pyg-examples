import pytest

from modules.attention.story.story import AttentionStory


class TestAttentionStory:
    def test_tells_the_whole_story(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO", logger="gnn_pyg_examples"):
            AttentionStory.main()

        assert "   Joe           0.354" in caplog.messages
        assert "   Bill          0.668" in caplog.messages
        assert "   Joe           0.551" in caplog.messages
        assert caplog.messages.count("   GATConv's shares the same as by hand? yes") == 2
        assert "   Nancy's new vector: [0.166, 0.233, 0.602]" in caplog.messages
        assert "   concat=True              [6, 64]       [18, 4]" in caplog.messages
        assert "   concat=False             [6, 16]       [18, 4]" in caplog.messages
        assert "   TransformerConv                [6, 64]       [30, 4]" in caplog.messages
