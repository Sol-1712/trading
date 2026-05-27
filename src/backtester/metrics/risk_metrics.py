import logging
import numpy as np
from functools import cached_property

logger = logging.getLogger(__name__)

class RiskMetrics:
    """
    Computes risk-based metrics from a CoreStats object.
    
    Quantifies volatility, drawdown, tail risk, and related metrics.
    Provides annualized and cumulative risk statistics for portfolio analysis.
    
    Parameters
    ----------
    core : CoreStats
        Precomputed core statistics with equity series and returns.
        
    Raises
    ------
    ValueError
        If core is None or missing required attributes.
    """

    def __init__(self, core):
        if core is None:
            raise ValueError("core cannot be None")
        if not hasattr(core, 'equity') or core.equity is None:
            raise ValueError("core must have 'equity' attribute")
        if not hasattr(core, 'log_returns') or core.log_returns is None:
            raise ValueError("core must have 'log_returns' attribute")
            
        self.core = core
        logger.debug("RiskMetrics initialized with %d observations", core.n_obs)


    @property
    def sd(self) -> float:
        """ Per bar standard deviation. """
        return float(np.std(self.core.log_returns, ddof=1))


    @property
    def volatility(self) -> float:
        """
        Annualized volatility of returns.

        Returns:
            float: Annualized standard deviation of returns.

        Metrics:
        - Volatility: Standard deviation of returns scaled to annualized units.
        """

        return float(self.sd * self.core.ann_sqrt)


    @property
    def max_drawdown(self) -> float:
        return float(self.core.drawdown.min())


    @cached_property
    def drawdown_durations(self) -> np.ndarray:
        """
        Lengths of consecutive drawdown periods (number of bars).

        Returns:
            np.ndarray: Array of drawdown durations.
        """
        dd = self.core.drawdown < 0  # True when in drawdown

        if not dd.any():
            return np.array([], dtype=int)

        # Convert boolean array to int (1=in drawdown, 0=flat)
        dd_int = dd.astype(int)

        # Pad with zeros to detect start/end of drawdowns
        padded = np.r_[0, dd_int, 0]
        diff = np.diff(padded)

        # Start and end indices of each drawdown
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]

        # Length of each drawdown period
        durations = ends - starts

        return durations
    

    @property
    def time_in_drawdown(self) -> float:
        """
        Fraction of bars spent in drawdown.

        Returns:
            float: Value between 0 and 1 representing % of time in drawdown.
        """
        return float(np.mean(self.core.drawdown < 0))
    

    @property
    def max_drawdown_duration(self) -> float:
        """
        Maximum consecutive drawdown duration (number of bars).

        Returns:
            float: Maximum drawdown duration. Returns 0 if no drawdowns.
        """
        durations = self.drawdown_durations
        return float(durations.max()) if len(durations) > 0 else 0.0


    @property
    def avg_drawdown_duration(self) -> float:
        """
        Average consecutive drawdown duration (number of bars).

        Returns:
            float: Average drawdown duration. Returns 0 if no drawdowns.
        """
        durations = self.drawdown_durations
        return float(durations.mean()) if len(durations) > 0 else 0.0
    

    @property
    def calmar(self) -> float:
        """
        Calmar Ratio (cagr / mdd)

        Returns:
            float: Calmar Ratio.
        """
        if self.max_drawdown == 0:
            return np.nan
        
        n_years = self.core.n_obs / self.core.ann_factor

        if n_years == 0:
            return np.nan

        cagr = float((self.core.equity[-1] / self.core.equity[0]) ** (1 / n_years) - 1)

        return float(cagr / abs(self.max_drawdown))
    

    @property
    def downside_deviation(self) -> float:
        """
        Calculates downside deviation. Current threshold is 0.

        Returns:
            - float: Annualised downside deviation
        """
        downside = np.minimum(0, self.core.returns - self.core.mar)
        dd_std = np.sqrt(np.mean(downside ** 2))

        return float(dd_std * self.core.ann_sqrt)
    

    @cached_property
    def longest_losing_streak(self) -> int:
        """
        Maximum number of consecutive losing periods.
    
        Returns:
            - int: longest streak of returns < 0
        """
        returns = self.core.returns
        if len(returns) == 0:
            return 0

        # Boolean array: True where return < 0
        losses = returns < 0

        if not losses.any():
            return 0

        # Find the boundaries where streaks start/end
        # Convert boolean array to int
        losses_int = losses.astype(int)

        # Use run-length encoding trick: zero-pad at both ends
        padded = np.r_[0, losses_int, 0]
        diff = np.diff(padded)

        # Starts (+1) and ends (-1) of streaks
        run_starts = np.where(diff == 1)[0]
        run_ends   = np.where(diff == -1)[0]

        # Lengths of each streak
        streak_lengths = run_ends - run_starts

        return int(streak_lengths.max())
    

    def var(self, alpha: float = 0.05) -> float:
        """
        Value at Risk (VaR) at given confidence level.
        Returns positive loss magnitude.
        """
        if len(self.core.log_returns) == 0:
            return np.nan
        

        return float(-np.percentile(self.core.log_returns, alpha * 100))


    def cvar(self, alpha: float = 0.05) -> float:
        """
        Conditional VaR (CVaR) / Expected Shortfall at given confidence.
        Returns positive average loss beyond VaR.
        """
        if len(self.core.log_returns) == 0:
            return np.nan

        var_alpha = np.percentile(self.core.log_returns, alpha * 100)
        tail_losses = self.core.log_returns[self.core.log_returns <= var_alpha]

        if len(tail_losses) == 0:
            return 0.0

        return float(-np.mean(tail_losses))
    

    @property
    def var_95(self) -> float:
        return self.var(0.05)


    @property
    def var_99(self) -> float:
        return self.var(0.01)
    

    @property
    def cvar_95(self) -> float:
        return self.cvar(0.05)

    @property
    def cvar_99(self) -> float:
        return self.cvar(0.01)