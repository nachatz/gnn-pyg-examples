import pytest

from modules.aggregation.story.story import AggregationStory


class TestAggregationStory:
    def test_tells_the_whole_story(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO", logger="gnn_pyg_examples"):
            AggregationStory.main()

        assert sum(message.endswith("  yes") for message in caplog.messages[:6]) == 3
        assert "   concatenated instead, is [Joe, Bill, Mei] the same as [Mei, Bill, Joe]? no" in caplog.messages
        assert "   1000  [500.000, 0.000, 500.000]  [0.500, 0.000, 0.500]" in caplog.messages
        assert "   Joe           1/sqrt(2*4)  0.354" in caplog.messages
        assert "   HandGCNConv's weights the same as PyG's gcn_norm? yes" in caplog.messages
        assert "   GCNConv's output the same as HandGCNConv's? yes" in caplog.messages
