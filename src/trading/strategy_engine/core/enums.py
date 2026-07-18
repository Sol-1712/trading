from __future__ import annotations
from enum import Enum, auto


class SignalDirection(Enum):
    LONG  =  1
    SHORT = -1
    FLAT  =  0


class Side(Enum):
    BUY  = auto()
    SELL = auto()

    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

