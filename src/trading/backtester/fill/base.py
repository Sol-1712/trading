from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar
import logging
import pandas as pd

from trading.data_utils.core.enums import PriceType

logger = logging.getLogger(__name__)

class FillModel(ABC):
    """
    Abstract interface for order fill simulation models.

    Implementations decide fill price, partial-fill behaviour, and whether
    an order executes on a given bar (slippage, liquidity, delays, etc.).

    Parameters
    ----------
    fee_rate : float
        Proportional fee applied to filled notional; must be in ``[0, 1)``.
    """

    def __init__(self, fee_rate: float) -> None:
        self.fee_rate = fee_rate
        if not 0.0 <= self.fee_rate < 1.0:
            raise ValueError(f"fee_rate {self.fee_rate} out of range [0, 1.0)")

    price_type: ClassVar[PriceType]

    @abstractmethod
    def attempt_fill(self, order: Order, bar: pd.Series) -> Fill:
        """
        Simulate execution of an order against the current bar.

        Parameters
        ----------
        order : Order
            Order to attempt filling.
        bar : pd.Series
            Current bar data (OHLC + market fields required by the model).

        Returns
        -------
        Fill
            Fill result with units executed, fill price, and fees.
            Units may be less than the remaining order notional for
            partial fills.
        """


@dataclass(frozen=True)
class Fill:
    """
    Immutable record of a completed (or partial) order execution.

    Attributes
    ----------
    placed_at : datetime
        Timestamp when the parent order was placed.
    filled_at : datetime
        Timestamp when this fill occurred.
    units_filled : float
        Signed units filled. Magnitude may be less than requested when
        the fill is partial or spans multiple bars.
    fill_price : float
        Execution price per unit (may include modelled slippage).
    fees : float
        Fees charged for this fill.
    """
    placed_at:    datetime
    filled_at:    datetime
    units_filled: float
    fill_price:   float
    fees:         float
    
    def __post_init__(self):
        if self.fill_price <= 0:
            raise ValueError(f"fill_price must be positive, got {self.fill_price}")
        if self.fill_price > 1e10:
            raise ValueError(f"fill_price seems unrealistic: {self.fill_price}")


@dataclass(frozen=True)
class Order:
    """
    Immutable order submitted to the execution engine.

    Represents a signed notional delta to trade. The engine drives fills
    via a FillModel until ``remaining_notional`` is exhausted.

    Attributes
    ----------
    placed_at : datetime
        Timestamp when the order was created.
    exec_bar : int
        Bar index at which fill attempts become eligible (after delay).
    delta_notional : float
        Signed notional in quote currency fixed at submission.
        Positive = buy, negative = sell.
    remaining_notional : float
        Notional still to fill; shrinks as partial fills accumulate.
    """
    placed_at:          datetime
    exec_bar:           int
    delta_notional:     float
    remaining_notional: float
    
    def __post_init__(self):
        if abs(self.delta_notional) > 1e15:
            raise ValueError(f"delta_notional seems unrealistic: {self.delta_notional}")
        if not (-abs(self.delta_notional) <= self.remaining_notional <= abs(self.delta_notional)):
            raise ValueError(
                f"remaining_notional {self.remaining_notional} outside [{-abs(self.delta_notional)}, {abs(self.delta_notional)}]"
            )
