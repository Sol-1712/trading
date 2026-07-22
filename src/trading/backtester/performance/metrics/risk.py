import numpy as np
from functools import cached_property

from .base import MetricsGroup
from .utils import compute_sharpe


class RiskMetrics(MetricsGroup):
    """Drawdown, volatility, and risk-adjusted return metrics."""

    @cached_property
    def _long_mask(self) -> np.ndarray:
        return self.core.position_fraction > 0

    @cached_property
    def _short_mask(self) -> np.ndarray:
        return self.core.position_fraction < 0

    @cached_property
    def _long_excess(self) -> np.ndarray:
        return self.core.excess_returns[self._long_mask]

    @cached_property
    def _short_excess(self) -> np.ndarray:
        return self.core.excess_returns[self._short_mask]

    @cached_property
    def _drawdown_durations(self) -> np.ndarray:
        """Lengths in bars of consecutive drawdown periods."""
        in_drawdown = self.core.drawdown < 0
        if not np.any(in_drawdown):
            return np.array([], dtype=int)

        padded = np.r_[0, in_drawdown.astype(int), 0]
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        return ends - starts

    @cached_property
    def _drawdown_episode_depths(self) -> np.ndarray:
        """Trough drawdown for each distinct drawdown episode."""
        dd = self.core.drawdown
        in_drawdown = dd < 0
        if not np.any(in_drawdown):
            return np.array([], dtype=np.float64)

        padded = np.r_[False, in_drawdown, False]
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        return np.array([dd[start:end].min() for start, end in zip(starts, ends)])

    @property
    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown."""
        return float(self.core.drawdown.min())

    @property
    def max_drawdown_duration(self) -> float:
        """Longest drawdown episode in bars."""
        durations = self._drawdown_durations
        return float(durations.max()) if durations.size > 0 else 0.0

    @property
    def avg_drawdown_duration(self) -> float:
        """Average drawdown episode length in bars."""
        durations = self._drawdown_durations
        return float(durations.mean()) if durations.size > 0 else 0.0

    @property
    def avg_drawdown(self) -> float:
        """Average trough drawdown across drawdown episodes."""
        depths = self._drawdown_episode_depths
        return float(depths.mean()) if depths.size > 0 else 0.0

    @property
    def time_in_drawdown(self) -> float:
        """Fraction of bars spent in drawdown."""
        return float(np.mean(self.core.drawdown < 0))

    @property
    def sharpe(self) -> float:
        """Per-period Sharpe of excess bar returns."""
        return compute_sharpe(self.core.excess_returns)

    @property
    def annualised_sharpe(self) -> float:
        """Annualised Sharpe: per-period Sharpe scaled by sqrt(periods_per_year)."""
        return self.sharpe * self.core.ann_sqrt

    @property
    def long_sharpe(self) -> float:
        """Annualised Sharpe of excess returns while long."""
        if self._long_excess.size == 0:
            return float("nan")
        return compute_sharpe(self._long_excess) * self.core.ann_sqrt

    @property
    def short_sharpe(self) -> float:
        """Annualised Sharpe of excess returns while short."""
        if self._short_excess.size == 0:
            return float("nan")
        return compute_sharpe(self._short_excess) * self.core.ann_sqrt

    @property
    def sortino(self) -> float:
        """Annualised Sortino ratio (Excess returns)."""
        if self.core.returns.size == 0:
            return float("nan")

        downside = np.minimum(0.0, self.core.excess_returns)
        dd_std = np.sqrt(np.mean(downside ** 2))
        
        if np.isclose(dd_std, 0):
            return np.nan


        return float((np.mean(self.core.excess_returns) / dd_std) * self.core.ann_sqrt)

    @property
    def calmar(self) -> float:
        """CAGR divided by absolute maximum drawdown."""
        if self.max_drawdown == 0:
            return float("nan")

        n_years = self.core.n_obs / self.core.ann_factor
        if n_years == 0:
            return float("nan")

        cagr = (self.core.equity[-1] / self.core.equity[0]) ** (1 / n_years) - 1
        return float(cagr / abs(self.max_drawdown))

    @property
    def var_95(self) -> float:
        """95% Value at Risk (positive loss magnitude)."""
        return self._var(0.05)

    @property
    def var_99(self) -> float:
        """99% Value at Risk (positive loss magnitude)."""
        return self._var(0.01)

    @property
    def cvar_95(self) -> float:
        """95% Conditional Value at Risk."""
        return self._cvar(0.05)

    @property
    def cvar_99(self) -> float:
        """99% Conditional Value at Risk."""
        return self._cvar(0.01)

    @property
    def downside_deviation(self) -> float:
        """Annualised downside deviation of returns below zero."""
        downside = np.minimum(0.0, self.core.returns)
        dd_std = np.sqrt(np.mean(downside ** 2))
        return float(dd_std * self.core.ann_sqrt)

    @property
    def volatility(self) -> float:
        """Annualised volatility of log returns."""
        sd = float(np.std(self.core.log_returns, ddof=1))
        return float(sd * self.core.ann_sqrt)

    @property
    def longest_losing_streak(self) -> float:
        """Longest run of consecutive losing bars."""
        returns = self.core.returns
        if returns.size == 0:
            return 0.0

        losses = returns < 0
        if not np.any(losses):
            return 0.0

        padded = np.r_[0, losses.astype(int), 0]
        diff = np.diff(padded)
        run_starts = np.where(diff == 1)[0]
        run_ends = np.where(diff == -1)[0]
        streak_lengths = run_ends - run_starts
        return float(streak_lengths.max())

    def _var(self, alpha: float) -> float:
        """
        Historical VaR on log returns as a positive loss magnitude.

        Uses the ``alpha`` percentile of log returns, then negates so
        larger losses are reported as larger positive values.
        """
        if self.core.log_returns.size == 0:
            return float("nan")
        return float(-np.percentile(self.core.log_returns, alpha * 100))

    def _cvar(self, alpha: float) -> float:
        """
        Historical CVaR: mean of log returns at or below the VaR threshold.

        Returned as a positive loss magnitude (negated mean of the tail).
        Empty tail yields 0.0; empty series yields NaN.
        """
        if self.core.log_returns.size == 0:
            return float("nan")

        var_alpha = np.percentile(self.core.log_returns, alpha * 100)
        tail_losses = self.core.log_returns[self.core.log_returns <= var_alpha]
        if tail_losses.size == 0:
            return 0.0
        return float(-np.mean(tail_losses))
