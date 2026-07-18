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
    
    Implementations determine fill price, partial fill behavior, and execution
    timing given an order and bar data. Can model slippage, partial fills,
    and execution delays.
    """

    def __init__(self, fee_rate: float) -> None:
        self.fee_rate = fee_rate
        if not 0.0 <= self.fee_rate < 1.0:
            raise ValueError(f"fee_rate {self.fee_rate} out of range [0, 1.0)")

    price_type: ClassVar[PriceType]

    @abstractmethod
    def attempt_fill(self, order: Order, bar: pd.Series) -> Fill:
        """
        Simulate execution of an order against current bar data.
        
        Parameters
        ----------
        order : Order
            Order to attempt execution on.
        bar : pd.Series
            Current bar data (OHLC + market data).
            
        Returns
        -------
        Fill
            Fill result with actual units executed and fill price.
            Units may be less than order amount for partial fills.
        """


@dataclass(frozen=True)
class Fill:
    """
    Executed fill record.
    
    Immutable record of a successful (or partial) order execution.
    
    Attributes
    ----------
    placed_at : datetime
        Timestamp when order was placed.
    filled_at : datetime
        Timestamp when fill was executed.
    units_filled : float
        Number of units actually filled. May be less than requested for
        partial fills or orders that span multiple bars.
    fill_price : float
        Execution price per unit (may include slippage).
    fees : float
        Fees paid for the fill.
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
    Order submitted to execution engine.
    
    Immutable record representing a position target request.
    Execution engine converts this to actual fills via fill model.
    
    Attributes
    ----------
    placed_at : datetime
        Timestamp when order was created.
    exec_bar : int
        Bar index when execution should attempt (based on delay).
    delta_notional : float
        Signed notional amount in quote currency (fixed at submission).
        Positive = buy, negative = sell.
    remaining_notional : float
        Remaining notional to fill. Decreases as order is partially filled
        across multiple bars.
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
