from torch import Tensor

from common.logger import logger

Cell = str | int | float | bool | Tensor


class Narrator:
    def __init__(self) -> None:
        self._sections_told = 0

    def section(self, title: str) -> None:
        if self._sections_told:
            logger.info("")
        self._sections_told += 1
        logger.info("%d. %s", self._sections_told, title)

    def table(self, header: list[str], rows: list[list[Cell]], precision: int = 3) -> None:
        lines = [header] + [[self.format(cell, precision) for cell in row] for row in rows]
        widths = [max(len(line[column]) for line in lines) for column in range(len(header))]
        lines.insert(1, ["-" * width for width in widths])

        for line in lines:
            padded = [cell.ljust(width) for cell, width in zip(line, widths, strict=True)]
            logger.info("   %s", "  ".join(padded).rstrip())

    def fact(self, label: str, value: Cell) -> None:
        separator = " " if label.endswith("?") else ": "
        logger.info("   %s%s%s", label, separator, self.format(value))

    def note(self, text: str) -> None:
        logger.info("   %s", text)

    @classmethod
    def format(cls, cell: Cell, precision: int = 3) -> str:
        if isinstance(cell, bool):
            return "yes" if cell else "no"
        if isinstance(cell, Tensor) and cell.dim() == 1:
            # Vectors always use 3 decimals so they line up across tables; precision only applies to scalars.
            return "[" + ", ".join(f"{value:.3f}" for value in cell.tolist()) + "]"
        if isinstance(cell, Tensor):
            return cls.format(cell.item(), precision)
        if isinstance(cell, float):
            return f"{cell:.{precision}f}"
        return str(cell)
