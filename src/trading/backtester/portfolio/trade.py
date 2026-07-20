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
        entry_price: float, # weighted-average entry price
        units:       float, 
        ) -> None:

        self.entry_time      = entry_time
        self.direction       = direction
        self.avg_entry_price = entry_price
        self.units           = units
        self.fees            = 0.0
        self.funding_pnl     = 0.0
        self.realized_pnl    = 0.0

        self.max_units_held    = abs(units)
        self.max_notional_held = abs(units) * entry_price

    @property
    def sign(self) -> int:
        return 1 if self.direction is SignalDirection.LONG else -1

    def accrue_bar(self, funding_pnl: float) -> None:
        """Called every bar the trade is open, regardless of fills."""
        self.funding_pnl += funding_pnl


    def add_fill(self, fill: Fill) -> None:
        """Called only on bars where a rebalancing order executes against
        this trade. Caller (TradeLog) is responsible for splitting a fill
        that would cross zero into a closing portion and a new-trade portion.
        """
        self.fees += fill.fees

        is_increase = (fill.units_filled > 0) == (self.units > 0)

        if is_increase:
            total_cost = (
                self.avg_entry_price * abs(self.units)
                + fill.fill_price * abs(fill.units_filled)
            )
            self.units += fill.units_filled
            self.avg_entry_price = total_cost / abs(self.units)
        else:
            units_closed = min(abs(fill.units_filled), abs(self.units))
            self.realized_pnl += self.sign * units_closed * (fill.fill_price - self.avg_entry_price)
            self.units += fill.units_filled

        max_h = max(abs(self.max_units_held), abs(self.units))
        self.max_units_held    = max_h if self.sign == 1 else -max_h
        max_n = max(abs(self.max_notional_held), abs(self.units) * fill.fill_price)
        self.max_notional_held = max_n if self.sign == 1 else -max_n


    def close(
        self, 
        exit_time: datetime, 
        ) -> Trade:

        return Trade(
            entry_time=self.entry_time,
            exit_time=exit_time,
            direction=self.direction,
            avg_entry_price=self.avg_entry_price,
            position_pnl=self.realized_pnl,
            fees=self.fees,
            funding_pnl=self.funding_pnl,
            net_pnl=self.realized_pnl + self.funding_pnl - self.fees,
            max_units_held=self.max_units_held,
            max_notional_held=self.max_notional_held,
        )


@dataclass(frozen=True)
class Trade:
    """Immutable, finalized record — only exists once a trade is closed."""
    entry_time:        datetime
    exit_time:         datetime
    direction:         SignalDirection
    avg_entry_price:   float
    position_pnl:      float
    fees:              float
    funding_pnl:       float
    net_pnl:           float
    max_units_held:    float
    max_notional_held: float


class TradeLog:
    """
    Tracks open and closed trades, keyed by instrument.
    Consumes live Fill objects and per-bar funding as the backtest runs —
    mirrors how a live system would track positions, not a post-hoc
    reconstruction from portfolio_history.
    """

    def __init__(self) -> None:
        self._open_trade:    OpenTrade | None = None
        self._closed_trades: list[Trade] = []

    @property
    def open_trade(self) -> OpenTrade | None:
        return self._open_trade

    @property
    def closed_trades(self) -> list[Trade]:
        return self._closed_trades

    def accrue_bar(self, funding_pnl: float) -> None:
        """Called every bar, regardless of fills."""
        if self._open_trade is not None:
            self._open_trade.accrue_bar(funding_pnl)

    def on_fill(self,fill: Fill, timestamp: datetime) -> None:

        if self._open_trade is None:
            self._open_trade = self._open_new_trade(timestamp, fill)
            return

        same_direction = (fill.units_filled > 0) == (self._open_trade.units > 0)

        if same_direction:
            self._open_trade.add_fill(fill)
            return

        # opposite direction: partial reduce, exact close, or reversal
        if abs(fill.units_filled) < abs(self._open_trade.units):
            self._open_trade.add_fill(fill)

        elif abs(fill.units_filled) == abs(self._open_trade.units):
            closed = self._open_trade.close(timestamp)
            self._closed_trades.append(closed)
            self._open_trade = None

        else:  # reversal — fill overshoots current position
            closing_fill, opening_fill = self._split_reversal_fill(fill, self._open_trade.units)
            self._open_trade.add_fill(closing_fill)
            closed = self._open_trade.close(timestamp)
            self._closed_trades.append(closed)
            self._open_trade = self._open_new_trade(timestamp, opening_fill)


    def _open_new_trade(self, timestamp: datetime, fill: Fill) -> OpenTrade:
        direction = SignalDirection.LONG if fill.units_filled > 0 else SignalDirection.SHORT
        trade = OpenTrade(
            entry_time=timestamp,
            direction=direction,
            entry_price=fill.fill_price,
            units=fill.units_filled,
        )
        trade.fees += fill.fees
        return trade

    
    def _split_reversal_fill(self, fill: Fill, current_units: float) -> tuple[Fill, Fill]:
        """Splits a fill that crosses zero into a closing portion (sized
        exactly to flatten the current position) and an opening portion
        (the remainder, in the new direction)."""
        closing_units = -current_units
        opening_units = fill.units_filled - closing_units

        total_units = abs(fill.units_filled)
        closing_fees = fill.fees * (abs(closing_units) / total_units)
        opening_fees = fill.fees - closing_fees

        closing_fill = Fill(
            placed_at=fill.placed_at,
            filled_at=fill.filled_at,
            units_filled=closing_units,
            fill_price=fill.fill_price,
            fees=closing_fees,
        )
        opening_fill = Fill(
            placed_at=fill.placed_at,
            filled_at=fill.filled_at,
            units_filled=opening_units,
            fill_price=fill.fill_price,
            fees=opening_fees,
        )
        return closing_fill, opening_fill