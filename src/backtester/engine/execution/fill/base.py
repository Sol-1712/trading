from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


class FillModel(ABC):

    @abstractmethod
    def fill(self, order: Order) -> Fill:
        """
        Given an order intent and bar data, return the actual fill.
        Determines fill price and whether order was fully executed.
        """
        ...


@dataclass(frozen=True)
class Fill:
    placed_at:    datetime
    filled_at:    datetime
    units_filled: float    # may be < order.delta_units for partial fills
    fill_price:   float    # may differ from close for slippage models



@dataclass(frozen=True)
class Order:
    timestamp:       datetime
    delta_units:     float
