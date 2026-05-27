from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


class FillModel(ABC):

    @abstractmethod
    def attempt_fill(self, order: Order, bar: pd.Series) -> Fill:
        """
        Given an order intent and bar data, return the actual fill.
        Determines fill price and whether order was fully executed.
        """


@dataclass(frozen=True)
class Fill:
    placed_at:    datetime
    filled_at:    datetime
    units_filled: float    # may be < order.delta_units for partial fills
    fill_price:   float    # may differ from close for slippage models



@dataclass(frozen=True)
class Order:
    placed_at:          datetime
    exec_bar:           int
    delta_notional:     float    # signed dollar amount committed — fixed at submission
    remaining_notional: float    # decreases as partially filled
