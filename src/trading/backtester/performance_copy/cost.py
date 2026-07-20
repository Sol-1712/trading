import numpy as np
from functools import cached_property

from .base import MetricsGroup
from .utils import compute_sharpe


class CostMetrics(MetricsGroup):

    @cached_property
    def _net_return_sum(self) -> float:
        """Arithmetic sum of per-bar net returns (for ratio denominators)."""
        return float(np.sum(self.core.returns))

    @cached_property
    def _gross_return_sum(self) -> float:
        """Arithmetic sum of per-bar position returns."""
        return float(np.sum(self.core.position_returns))

    @cached_property
    def _sharpe(self) -> float:
        return compute_sharpe(
            self.core.returns,
            rf=self.core.rf,
            ann_factor=self.core.ann_factor,
        )

    @cached_property
    def _position_fraction_delta(self) -> np.ndarray:
        return np.abs(np.diff(self.core.position_fraction, prepend=0.0))

    @property
    def total_fee_return(self) -> float:
        """Cumulative fee drag as a fraction of equity."""
        return float(np.sum(self.core.fee_returns))

    @property
    def total_funding_return(self) -> float:
        """Cumulative funding impact as a fraction of equity."""
        return float(np.sum(self.core.funding_returns))

    @property
    def total_cost_return(self) -> float:
        """Cumulative cost drag: fees minus funding received."""
        return float(np.sum(self.core.fee_returns - self.core.funding_returns))

    @property
    def total_fee_pct_of_net(self) -> float:
        """Fees as a fraction of absolute net return."""
        if self._net_return_sum == 0:
            return float("nan")
        return float(self.total_fee_return / abs(self._net_return_sum))

    @property
    def funding_pct_of_net(self) -> float:
        """Funding as a fraction of absolute net return."""
        if self._net_return_sum == 0:
            return float("nan")
        return float(self.total_funding_return / abs(self._net_return_sum))

    @property
    def total_cost_pct_of_net(self) -> float:
        """Total costs as a fraction of absolute net return."""
        if self._net_return_sum == 0:
            return float("nan")
        return float(self.total_cost_return / abs(self._net_return_sum))

    @property
    def cost_to_gross_ratio(self) -> float:
        """Total costs relative to gross return."""
        if self._gross_return_sum == 0:
            return float("nan")
        return float(self.total_cost_return / abs(self._gross_return_sum))

    @property
    def fee_drag_sharpe(self) -> float:
        """Sharpe reduction attributable to fees."""
        returns_no_fee = self.core.returns + self.core.fee_returns
        sharpe_no_fee = compute_sharpe(
            returns_no_fee,
            rf=self.core.rf,
            ann_factor=self.core.ann_factor,
        )
        return float(sharpe_no_fee - self._sharpe)

    @property
    def funding_drag_sharpe(self) -> float:
        """Sharpe impact of funding payments."""
        returns_no_funding = self.core.returns - self.core.funding_returns
        sharpe_no_funding = compute_sharpe(
            returns_no_funding,
            rf=self.core.rf,
            ann_factor=self.core.ann_factor,
        )
        return float(sharpe_no_funding - self._sharpe)

    @property
    def avg_fee_per_bar(self) -> float:
        """Average fee per bar as a fraction of equity."""
        return float(np.mean(self.core.fee_returns))

    @property
    def avg_fee_per_trade(self) -> float:
        """Average fee per rebalance as a fraction of equity."""
        n_trades = int(np.sum(self.core.trade_occurred))
        if n_trades == 0:
            return float("nan")
        return float(np.sum(self.core.fee_returns) / n_trades)

    @property
    def annualised_turnover(self) -> float:
        """Annualised sum of absolute position-fraction changes."""
        return float(np.mean(self._position_fraction_delta) * self.core.ann_factor)

    @property
    def pct_bars_paying_funding(self) -> float:
        """Fraction of bars where funding was a cost."""
        return float(np.mean(self.core.funding_returns < 0))
