import numpy as np
from functools import cached_property

from .base import MetricsGroup
from .core_stats import CoreStats
from trading.backtester.portfolio import TradeLog
from trading.backtester.portfolio.trade import Trade



class TradeMetrics(MetricsGroup):
    """
    Trade-log and position-state metrics.

    Parameters
    ----------
    core : CoreStats
        Shared portfolio history substrate.
    trade_log : TradeLog
        Source of closed round-trip trades.
    """

    def __init__(self, core: CoreStats, trade_log: TradeLog) -> None:
        super().__init__(core)
        self.trade_log = trade_log

    @cached_property
    def _long_mask(self) -> np.ndarray:
        return self.core.position_fraction > 0

    @cached_property
    def _short_mask(self) -> np.ndarray:
        return self.core.position_fraction < 0

    @cached_property
    def _flat_mask(self) -> np.ndarray:
        return self.core.position_fraction == 0

    @cached_property
    def _long_returns(self) -> np.ndarray:
        return self.core.returns[self._long_mask]

    @cached_property
    def _short_returns(self) -> np.ndarray:
        return self.core.returns[self._short_mask]

    @cached_property
    def _closed_trades(self) -> list[Trade]:
        return self.trade_log.closed_trades

    @cached_property
    def _trade_returns(self) -> np.ndarray:
        """Per-trade return as net PnL divided by peak notional held."""
        returns: list[float] = []
        for trade in self._closed_trades:
            notional = abs(trade.max_notional_held)
            if notional == 0:
                continue
            returns.append(trade.net_pnl / notional)
        return np.asarray(returns, dtype=np.float64)

    @property
    def num_trades(self) -> float:
        """Number of completed round-trip trades."""
        return float(len(self._closed_trades))

    @property
    def avg_position_size(self) -> float:
        """Mean absolute position as a fraction of equity."""
        return float(np.mean(np.abs(self.core.position_fraction)))

    @property
    def max_position_size(self) -> float:
        """Maximum absolute position as a fraction of equity."""
        return float(np.max(np.abs(self.core.position_fraction)))

    @property
    def avg_long_size(self) -> float:
        """Mean signed position fraction while long."""
        long_pos = self.core.position_fraction[self._long_mask]
        return float(np.mean(long_pos)) if long_pos.size > 0 else 0.0

    @property
    def avg_short_size(self) -> float:
        """Mean signed position fraction while short."""
        short_pos = self.core.position_fraction[self._short_mask]
        return float(np.mean(short_pos)) if short_pos.size > 0 else 0.0

    @property
    def time_long(self) -> float:
        """Fraction of bars spent long."""
        return float(np.mean(self._long_mask))

    @property
    def time_short(self) -> float:
        """Fraction of bars spent short."""
        return float(np.mean(self._short_mask))

    @property
    def time_flat(self) -> float:
        """Fraction of bars with no position."""
        return float(np.mean(self._flat_mask))

    @property
    def hit_rate_trade(self) -> float:
        """Fraction of closed trades with positive net PnL."""
        if not self._closed_trades:
            return float("nan")
        wins = sum(1 for t in self._closed_trades if t.net_pnl > 0)
        return float(wins / len(self._closed_trades))

    @property
    def expectancy(self) -> float:
        """Probability-weighted expected return per trade."""
        returns = self._trade_returns
        returns = returns[returns != 0]
        if returns.size == 0:
            return float("nan")

        wins = returns[returns > 0]
        losses = returns[returns < 0]
        if wins.size == 0 or losses.size == 0:
            return float("nan")

        hit_rate = wins.size / returns.size
        return float(
            (hit_rate * np.mean(wins)) + ((1 - hit_rate) * np.mean(losses))
        )

    @property
    def profit_factor(self) -> float:
        """Ratio of gross winning trade returns to gross losing trade returns."""
        returns = self._trade_returns
        if returns.size == 0:
            return float("nan")

        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        if gross_loss == 0:
            return float("nan")
        return float(gross_profit / gross_loss)

    @property
    def time_in_market(self) -> float:
        """Fraction of bars in any non-flat position."""
        return 1.0 - self.time_flat

    @property
    def avg_holding_period(self) -> float:
        """Average length in bars of contiguous in-market segments."""
        in_market = self.core.position_fraction != 0
        if not np.any(in_market):
            return 0.0

        lengths: list[int] = []
        run = 0
        for active in in_market:
            if active:
                run += 1
            elif run:
                lengths.append(run)
                run = 0
        if run:
            lengths.append(run)

        return float(np.mean(lengths))

    @property
    def largest_win(self) -> float:
        """Largest single-bar return."""
        return float(np.max(self.core.returns))

    @property
    def largest_loss(self) -> float:
        """Largest single-bar loss (most negative return)."""
        return float(np.min(self.core.returns))

    @property
    def long_pnl_pct(self) -> float:
        """Share of total bar returns earned while long."""
        total = np.sum(self.core.returns)
        if total == 0:
            return float("nan")
        return float(np.sum(self._long_returns) / abs(total))

    @property
    def short_pnl_pct(self) -> float:
        """Share of total bar returns earned while short."""
        total = np.sum(self.core.returns)
        if total == 0:
            return float("nan")
        return float(np.sum(self._short_returns) / abs(total))

