from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trading.strategy_engine.core import SignalDirection
from trading.backtester.fill      import Fill


class OpenTrade:
    """Mutable, accumulates fills while the position stays open."""

    def __init__(
        self, 
        entry_time:  datetime, 
        direction:   SignalDirection, 
        entry_price: float, 
        units:       float, 
        fees:        float = 0.0, 
        funding_pnl: float = 0.0
        ) -> None:

        self.entry_time      = entry_time
        self.direction       = direction
        self.avg_entry_price = entry_price
        self.units           = units
        self.fees            = fees
        self.funding_pnl     = funding_pnl

    def accrue_bar(self, funding_pnl: float) -> None:
        """Called every bar the trade is open, regardless of fills."""
        self.funding_pnl += funding_pnl


    def add_fill(self, fill: Fill) -> None:
        """Called only on bars where a rebalancing order executes."""
        # update weighted-average entry price, units, fees
        self.fees  += fill.fee
        self.units += fill.units
        ...

    def close(
        self, 
        exit_time: datetime, 
        exit_price: float
        ) -> Trade:

        position_pnl = self.direction * self.units * (exit_price - self.avg_entry_price)

        return Trade(
            entry_time=self.entry_time, 
            exit_time=exit_time, 
            direction=self.direction, 
            avg_entry_price=self.avg_entry_price, 
            exit_price=exit_price, 
            units=self.units, 
            position_pnl=position_pnl, 
            fees=self.fees, 
            funding_pnl=self.funding_pnl,
            net_pnl=position_pnl + self.funding_pnl- self.fees
            )


@dataclass(frozen=True)
class Trade:
    """Immutable, finalized record — only exists once a trade is closed."""
    entry_time:       datetime
    exit_time:        datetime
    direction:        SignalDirection
    avg_entry_price:  float
    exit_price:       float
    units:            float
    position_pnl:     float
    fees:             float
    funding_pnl:      float
    net_pnl:          float



class TradeLog:
    pass