import numpy as np
from functools import cached_property


class RiskMetrics:
    """
    Computes risk-based metrics from a CoreStats object.

    This class provides key metrics:
    - Time in drawdown
    - Max drawdown duration
    - Avg drawdown duration
    - Downside deviation
    - Longest losing streak
    - VaR (95%, 99%)
    - CVaR (95%, 995)

    Attributes:
        core (CoreStats): Precomputed core statistics and returns from PnL.
        return_metrics (ReturnMetrics): Precomputed return metrics.
    """

    def __init__(self, core, return_metrics):
        """
        Initializes RiskMetrics with CoreStats and ReturnMetrics objects.

        Args:
            core (CoreStats): Object containing primitive statistics and returns.
            return_metrics (ReturnMetrics): Object containing calculated return metrics.
        """
        self.core = core
        self.return_metrics = return_metrics

    @cached_property
    def drawdown_durations(self) -> np.ndarray:
        """
        Computes the lengths of consecutive drawdown periods.

        Returns:
            np.ndarray: Array of drawdown durations in number of bars.
        """
        dd = self.core.drawdown < 0  # boolean array: True if in drawdown
        if not dd.any():
            return np.array([])

        durations = []
        count = 0
        for in_dd in dd:
            if in_dd:
                count += 1
            elif count > 0:
                durations.append(count)
                count = 0
        # Add final run if it ends in drawdown
        if count > 0:
            durations.append(count)

        return np.array(durations)
    
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
        mdd = self.core.mdd
        if mdd == 0:
            return np.nan
        return float(self.return_metrics.cagr / abs(mdd))
    
    @property
    def downside_deviation(self) -> float:
        """
        Calculates downside deviation. Current threshold is 0.

        Returns:
            - float: Annualised downside deviation
        """
        threshold = 0
        downside = np.minimum(0, self.core.returns - threshold)
        dd_std = np.sqrt(np.mean(downside ** 2))

        return float(dd_std * self.core.ann_sqrt)
    
    @property
    def longest_losing_streak(self) -> int:
        """
        Maximum number of consecutive losing periods.
    
        Returns:
            - int: longest streak of returns < 0
        """
        returns = self.core.returns

        if len(returns) == 0:
            return 0
        losses = returns < 0

        max_streak = 0
        current_streak = 0

        for is_loss in losses:
            if is_loss:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak
    
    def var(self, alpha: float = 0.05) -> float:
        """
        Value at Risk (VaR) at given confidence level.
        Returns positive loss magnitude.
        """
        if len(self.returns) == 0:
            return np.nan
        return float(-np.percentile(self.returns, alpha * 100))

    @property
    def var_95(self) -> float:
        return self.var(0.05)

    @property
    def var_99(self) -> float:
        return self.var(0.01)

    def cvar(self, alpha: float = 0.05) -> float:
        """
        Conditional VaR (CVaR) / Expected Shortfall at given confidence.
        Returns positive average loss beyond VaR.
        """
        if len(self.returns) == 0:
            return np.nan

        var_alpha = np.percentile(self.returns, alpha * 100)
        tail_losses = self.returns[self.returns <= var_alpha]

        if len(tail_losses) == 0:
            return 0.0

        return float(-np.mean(tail_losses))

    @property
    def cvar_95(self) -> float:
        return self.cvar(0.05)

    @property
    def cvar_99(self) -> float:
        return self.cvar(0.01)