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


class MarketType(Enum):
    SPOT   = auto()
    PERP   = auto()
    FUTURE = auto()
    OPTION = auto()


class OrderType(Enum):
    MARKET      = auto()
    LIMIT       = auto()
    STOP_MARKET = auto()
    STOP_LIMIT  = auto()


class OrderStatus(Enum):
    PENDING    = auto()   # Created locally, not yet sent
    OPEN       = auto()   # Resting on exchange book
    PARTIAL    = auto()   # Partially filled, still open
    FILLED     = auto()   # Fully filled
    CANCELLED  = auto()
    REJECTED   = auto()
    EXPIRED    = auto()