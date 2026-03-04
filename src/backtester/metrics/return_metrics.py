import numpy as np
from scipy.stats import skew, kurtosis

class ReturnMetrics:
    """
    Computes return-based metrics from a CoreStats object.

    This class provides key metrics:
    - CAGR (Compound Annual Growth Rate)
    - Sharpe ratio
    - Sortino ratio
    - Volatility
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
        self.core = core

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
       
        return float(self.core.equity[-1] ** (1 / n_years) - 1)

    @property
    def sharpe(self) -> float:
        """
        Annualized Sharpe ratio.

        Returns:
            float: Sharpe ratio. Returns 0.0 if standard deviation is zero.

        Metrics:
        - Sharpe ratio: (mean return / standard deviation) * sqrt(annualization factor)
        """
        if self.core.sd == 0:
            return 0.0
        
        return float((self.core.mean / self.core.sd) * self.core.ann_sqrt)
    
    @property
    def sortino(self) -> float:
        """
        Annualized Sortino ratio (downside risk focused).

        Returns:
            float: Sortino ratio. Returns np.nan if downside standard deviation is zero.

        Metrics:
        - Sortino ratio: (mean return / downside standard deviation) * sqrt(annualization factor)
        """
        threshold = 0
        downside = np.minimum(0, self.core.returns - threshold)
        dd_std = np.sqrt(np.mean(downside ** 2))

        if dd_std == 0:
            return np.nan
        
        return float((self.core.mean / dd_std) * self.core.ann_sqrt)
    
    @property
    def volatility(self) -> float:
        """
        Annualized volatility of returns.

        Returns:
            float: Annualized standard deviation of returns.

        Metrics:
        - Volatility: Standard deviation of returns scaled to annualized units.
        """

        return float(self.core.sd * self.core.ann_sqrt)
        
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

        wins = returns[returns > 0]
        losses = returns[returns < 0]

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
        return float(skew(self.core.returns, bias=False))
    
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
        return float(kurtosis(self.core.returns, bias=False))