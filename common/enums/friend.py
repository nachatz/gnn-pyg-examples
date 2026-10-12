from enum import IntEnum


class Friend(IntEnum):
    JOE = 0
    NANCY = 1
    BILL = 2
    MEI = 3
    PRIYA = 4
    OMAR = 5

    @property
    def label(self) -> str:
        return self.name.title()
