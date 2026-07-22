import logging
import numpy as np
from functools import cached_property
from scipy.stats import skew, kurtosis

from .base import MetricsGroup

logger = logging.getLogger(__name__)

class ReturnMetrics(MetricsGroup):
    """Bar-level return summary metrics derived from CoreStats."""

    @cached_property
    def _wins(self) -> np.ndarray:
        r = self.core.returns
        return r[r > 0]


    @cached_property
    def _losses(self) -> np.ndarray:
        r = self.core.returns
        return r[r < 0]


    @property
    def mean_bar_return(self) -> float:
        """Mean per-bar net return."""
        return float(np.mean(self.core.returns))


    @property
    def net_return(self) -> float:
        """Total net return: equity[-1] / equity[0] - 1."""
        if self.core.equity[-1] < 0:
            logger.warning("Portfolio liquidated (negative equity at end)")
            
        return float((self.core.equity[-1] / self.core.equity[0]) - 1)
    

    @property
    def gross_return(self) -> float:
        """Compounded return from position PnL only (excludes fees/funding)."""

        if not hasattr(self.core, 'position_returns') or self.core.position_returns is None:
            raise ValueError("core.position_returns is required")
            
        gross_equity = np.cumprod(1.0 + self.core.position_returns)
        return float(gross_equity[-1] - 1)


    @property
    def cagr(self) -> float:
        """Compound annual growth rate of equity; NaN if fewer than 2 bars."""

        if len(self.core.equity) < 2:
            logger.warning("Cannot compute CAGR with fewer than 2 observations")
            return np.nan
            
        n_years = self.core.n_obs / self.core.ann_factor

        return float((self.core.equity[-1] / self.core.equity[0]) ** (1 / n_years) - 1)    


    @property
    def skew(self) -> float:
        """Sample skewness of log returns (bias=False)."""
        return float(skew(self.core.log_returns, bias=False))
    

    @property
    def kurtosis(self) -> float:
        """Sample excess kurtosis of log returns (bias=False)."""
        return float(kurtosis(self.core.log_returns, bias=False))


    @property
    def largest_bar_return(self) -> float:
        """Largest single-bar position return (not net of fees/funding)."""
        return float(np.max(self.core.position_returns))


    @property
    def smallest_bar_return(self) -> float:
        """Smallest single-bar position return (not net of fees/funding)."""
        return float(np.min(self.core.position_returns))


    @property
    def hit_rate_all(self) -> float:
        """Fraction of bars with positive net return."""
        return float(len(self._wins) / len(self.core.returns))


    @property
    def avg_bar_win(self) -> float:
        """Mean of positive net bar returns; 0.0 if none."""
        return float(np.mean(self._wins)) if self._wins.size > 0 else 0.0


    @property
    def avg_bar_loss(self) -> float:
        """Mean of negative net bar returns; 0.0 if none."""
        return float(np.mean(self._losses)) if self._losses.size > 0 else 0.0


    @property
    def avg_bar_win_loss_ratio(self) -> float:
        """|avg win / avg loss|; NaN if avg loss is zero."""
        if self.avg_bar_loss == 0:
            return float("nan")
        return float(abs(self.avg_bar_win / self.avg_bar_loss))


    @property
    def median_bar_return(self) -> float:
        """Median per-bar net return."""
        return float(np.median(self.core.returns))
