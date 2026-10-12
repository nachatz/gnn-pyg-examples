import pytest
import torch

from common.narrator import Cell
from common.narrator import Narrator


class TestNarrator:
    @pytest.mark.parametrize(
        ("cell", "expected"),
        [
            (True, "yes"),
            (False, "no"),
            (torch.tensor([0.28333, 1.0, 0.0]), "[0.283, 1.000, 0.000]"),
            (torch.tensor(0.0754), "0.075"),
            (0.5, "0.500"),
            (7, "7"),
            ("Nancy", "Nancy"),
        ],
    )
    def test_formats_cells(self, cell: Cell, expected: str) -> None:
        assert Narrator.format(cell) == expected

    def test_precision_applies_to_scalars(self) -> None:
        assert Narrator.format(torch.tensor(0.00029), precision=4) == "0.0003"

    def test_tells_sections_tables_facts_and_questions(self, caplog: pytest.LogCaptureFixture) -> None:
        narrator = Narrator()
        with caplog.at_level("INFO", logger="gnn_pyg_examples"):
            narrator.section("First")
            narrator.table(["friend", "food"], [["Bill", 0.9], ["Joe", 0.0]])
            narrator.section("Second")
            narrator.fact("Bill's food", 0.9)
            narrator.fact("the same?", False)
            narrator.note("done")

        assert caplog.messages == [
            "1. First",
            "   friend  food",
            "   ------  -----",
            "   Bill    0.900",
            "   Joe     0.000",
            "",
            "2. Second",
            "   Bill's food: 0.900",
            "   the same? no",
            "   done",
        ]
