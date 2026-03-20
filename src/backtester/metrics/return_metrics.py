import numpy as np
from functools import cached_property
from scipy.stats import skew, kurtosis
from backtester.utils import _compute_sharpe

class ReturnMetrics:
    """
    Computes return-based metrics from a CoreStats object.

    This class provides key metrics:
    - Net return
    - Gross return
    - CAGR (Compound Annual Growth Rate)
    - Sharpe ratio (Per bar/Annualised)
    - Sortino ratio
    - Average win/loss ratio
    - Expectancy
    - Hit rate
    - Profit factor
    - Skew
    - Kurtosis

    Attributes:
        core (CoreStats): Precomputed core statistics and returns from PnL.
    """

    def __init__(self, core):
        """
        Initializes ReturnMetrics with a CoreStats object.

        Args:
            core (CoreStats): Object containing primitive statistics and returns.
        """
        self.core       = core


    @property
    def mean(self) -> float:
        """ Per bar mean return. """
        return float(np.mean(self.core.returns))

    



    @property
    def wins(self) -> np.ndarray:
        """Array of positive returns."""
        r = self.core.returns
        return r[r > self.core.mar]


    @property
    def losses(self) -> np.ndarray:
        """Array of negative returns."""
        r = self.core.returns
        return r[r < self.core.mar]


    @property
    def net_return(self) -> float:
        """
        Compunded net return of the strategy as a fraction of starting capital.
        Capital-agnostic.

        Returns
        -------
        float
            Net return: (final equity - initial equity) / initial equity
        """

        return float((self.core.equity[-1] / self.core.equity[0]) - 1)
    
    @property
    def gross_return(self) -> float:
        """
        Compounded gross return of the strategy as a fraction of starting capital.
        Capital-agnostic.

        Returns
        -------
        float
            Gross return
        """
        gross_equity = np.cumprod(1.0 + self.core.position_returns)
        return float(gross_equity[-1] - 1)
    

    @property
    def cagr(self) -> float:
        """
        Compound Annual Growth Rate (CAGR).

        Returns:
            float: CAGR as a decimal (e.g., 0.12 = 12%).
            Returns np.nan if there are no periods.

        Metrics:
        - CAGR: Compound annual growth rate of the equity curve.
        """
        n_years = self.core.n_obs / self.core.ann_factor

        # DEFENSE
        if n_years == 0:
            return np.nan
       
        return float((self.core.equity[-1] / self.core.equity[0]) ** (1 / n_years) - 1)


    @cached_property
    def sharpe(self) -> float:
        """Sharpe ratio of the strategy returns."""
        return _compute_sharpe(self.core.returns, self.core.rf)
    

    @property
    def annualised_sharpe(self) -> float:
        """Annualised Sharpe ratio of the strategy returns."""
        return self.sharpe * self.core.ann_sqrt
    

    @property
    def sortino(self) -> float:
        """
        Annualized Sortino ratio (downside risk focused).

        Returns:
            float: Sortino ratio. Returns np.nan if downside standard deviation is zero.

        Metrics:
        - Sortino ratio: (mean return / downside standard deviation) * sqrt(annualization factor)
        """
        log_mar = np.log1p(self.core.mar)
        excess = self.core.log_returns - log_mar
        downside = np.minimum(0, excess)
        dd_std = np.sqrt(np.mean(downside ** 2))

        if dd_std == 0:
            return np.nan
        
        excess = self.core.returns - self.core.mar
        
        return float(((np.mean(excess)) / dd_std) * self.core.ann_sqrt)
            

    @property
    def avg_win_loss_ratio_expectancy(self) -> tuple[float, float, float]:
        """
        Calculates average win/loss ratio, expectancy, and hit rate.

        Returns:
            tuple[float, float, float]:
            - avg_win_loss_ratio: Average size of winning trades relative to losing trades.
            - expectancy: Average expected return per trade.
            - hit_rate: Fraction of trades that are winning.
            Returns (np.nan, np.nan, np.nan) if there are no wins or losses.

        Metrics:
        - Average Win/Loss Ratio
        - Expectancy
        - Hit Rate
        """
        returns = self.core.returns
        returns = returns[returns != 0] 

        if len(returns) == 0:
            return (np.nan, np.nan, np.nan)

        wins = self.wins
        losses = self.losses

        if len(wins) == 0 or len(losses) == 0:
            return (np.nan, np.nan, np.nan)

        hit_rate = len(wins) / len(returns)
        avg_win_loss_ratio = np.mean(wins) / abs(np.mean(losses))
        expectancy = (hit_rate * np.mean(wins)) + ((1 - hit_rate) * np.mean(losses))

        hit_rate = float(hit_rate)
        avg_win_loss_ratio = float(avg_win_loss_ratio)
        expectancy = float(expectancy)

        return (avg_win_loss_ratio, expectancy, hit_rate)
    

    @property
    def profit_factor(self) -> float:
        """
        Ratio of total gross profits to total gross losses.

        Returns:
            float: Profit factor. Returns np.nan if there are no losses.

        Metrics:
        - Profit Factor
        """
        returns = self.core.returns
        gross_profit = returns[returns > 0].sum()
        gross_loss   = abs(returns[returns < 0].sum())

        if gross_loss == 0:
            return np.nan
        
        return float(gross_profit / gross_loss)
    

    @property
    def skew(self) -> float:
        """
        Measure of return asymmetry (skewness).

        Positive skew indicates a distribution with larger or more frequent positive returns;
        negative skew indicates larger or more frequent negative returns.

        Returns:
            float: Skew of returns (dimensionless).

        Metrics:
        - Skew
        """
        return float(skew(self.core.log_returns, bias=False))
    

    @property
    def kurtosis(self) -> float:
        """
        Measure of return tail heaviness (kurtosis).

        High kurtosis indicates more frequent extreme returns (fat tails), low kurtosis indicates
        lighter tails compared to a normal distribution (std normal dist. has kurtosis of 3).

        Returns:
            float: Kurtosis of returns (dimensionless).

        Metrics:
        - Kurtosis
        """
        return float(kurtosis(self.core.log_returns, bias=False))
    
