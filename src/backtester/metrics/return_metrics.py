import logging
import numpy as np
from functools import cached_property
from scipy.stats import skew, kurtosis
from backtester.utils import compute_sharpe

logger = logging.getLogger(__name__)

class ReturnMetrics:
    """
    Computes return-based metrics from a CoreStats object.
    
    Provides comprehensive performance metrics including net/gross return, CAGR,
    Sharpe ratio, Sortino ratio, win/loss statistics, profit factor, skew, and
    kurtosis. Uses cached properties to avoid redundant computations.
    
    Parameters
    ----------
    core : CoreStats
        Precomputed core statistics with equity series, returns, and derivatives.
        
    Raises
    ------
    ValueError
        If core is None or missing required attributes.
    """

    def __init__(self, core):
        if core is None:
            raise ValueError("core cannot be None")
        if not hasattr(core, 'returns') or core.returns is None:
            raise ValueError("core must have 'returns' attribute")
        if not hasattr(core, 'equity') or core.equity is None:
            raise ValueError("core must have 'equity' attribute")
            
        self.core = core
        logger.debug("ReturnMetrics initialized with %d observations", core.n_obs)


    @property
    def mean(self) -> float:
        """
        Per-bar arithmetic mean return.
        
        Returns
        -------
        float
            Mean of returns array.
        """
        return float(np.mean(self.core.returns))

    



    @property
    def wins(self) -> np.ndarray:
        """
        Positive return periods above minimum acceptable return (MAR).
        
        Returns
        -------
        np.ndarray
            1D array of return values where return > MAR.
        """
        r = self.core.returns
        return r[r > self.core.mar]


    @property
    def losses(self) -> np.ndarray:
        """
        Negative return periods below minimum acceptable return (MAR).
        
        Returns
        -------
        np.ndarray
            1D array of return values where return < MAR.
        """
        r = self.core.returns
        return r[r < self.core.mar]


    @property
    def net_return(self) -> float:
        """
        Compounded net return of the strategy as a fraction of starting capital.
        
        Capital-agnostic. Accounts for all costs (fees, funding, slippage).
        
        Returns
        -------
        float
            Net return: (final_equity - initial_equity) / initial_equity
            
        Raises
        ------
        ValueError
            If equity array is empty or contains zero/negative values at start.
        """
        if len(self.core.equity) == 0:
            raise ValueError("Equity array is empty")
        if self.core.equity[0] <= 0:
            raise ValueError(f"Initial equity must be positive, got {self.core.equity[0]}")
        if self.core.equity[-1] < 0:
            logger.warning("Portfolio liquidated (negative equity at end)")
            
        return float((self.core.equity[-1] / self.core.equity[0]) - 1)
    
    @property
    def gross_return(self) -> float:
        """
        Compounded gross return before costs (fees, funding).
        
        Capital-agnostic. Represents position PnL only (not including cost drag).
        
        Returns
        -------
        float
            Gross return as a decimal.
            
        Raises
        ------
        ValueError
            If position_returns array is empty.
        """
        if not hasattr(self.core, 'position_returns') or self.core.position_returns is None:
            raise ValueError("core.position_returns is required")
            
        gross_equity = np.cumprod(1.0 + self.core.position_returns)
        return float(gross_equity[-1] - 1)
    

    @property
    def cagr(self) -> float:
        """
        Compound Annual Growth Rate (CAGR).
        
        Annualizes total return over the backtest period.
        
        Returns
        -------
        float
            CAGR as a decimal (e.g., 0.12 = 12%).
            Returns np.nan if backtest duration is zero.
            
        Raises
        ------
        ValueError
            If equity array is empty or has zero length.
        """
        if len(self.core.equity) < 2:
            logger.warning("Cannot compute CAGR with fewer than 2 observations")
            return np.nan
            
        n_years = self.core.n_obs / self.core.ann_factor

        if n_years == 0:
            return np.nan
       
        return float((self.core.equity[-1] / self.core.equity[0]) ** (1 / n_years) - 1)


    @cached_property
    def sharpe(self) -> float:
        """
        Sharpe ratio of strategy returns.
        
        Per-bar excess return (over risk-free rate) divided by volatility.
        Annualized via compute_sharpe helper.
        
        Returns
        -------
        float
            Annualized Sharpe ratio. Returns np.nan if volatility is zero.
        """
        return compute_sharpe(self.core.returns, self.core.rf, self.core.ann_factor)
    

    @property
    def annualised_sharpe(self) -> float:
        """
        Annualized Sharpe ratio.
        
        Sharpe ratio scaled by sqrt(annualization_factor) for consistency.
        
        Returns
        -------
        float
            Annualized Sharpe ratio.
        """
        if np.isnan(self.sharpe):
            return np.nan
        return self.sharpe * self.core.ann_sqrt
    

    @property
    def sortino(self) -> float:
        """
        Sortino ratio (downside risk focused).
        
        Similar to Sharpe but uses only downside volatility (returns below MAR).
        Penalizes downside volatility more than upside volatility.
        
        Returns
        -------
        float
            Annualized Sortino ratio. Returns np.nan if downside std is zero
            (strategy never underperforms MAR).
            
        Raises
        ------
        ValueError
            If log_returns is None or empty.
        """
        if len(self.core.log_returns) == 0:
            raise ValueError("log_returns array is empty")
            
        log_mar = np.log1p(self.core.mar)
        excess = self.core.log_returns - log_mar
        downside = np.minimum(0, excess)
        dd_std = np.sqrt(np.mean(downside ** 2))

        if dd_std == 0:
            logger.debug("Downside volatility is zero, returning np.nan for Sortino")
            return np.nan
        
        excess = self.core.returns - self.core.mar
        
        return float(((np.mean(excess)) / dd_std) * self.core.ann_sqrt)
            

    @property
    def avg_win_loss_ratio_expectancy(self) -> tuple[float, float, float]:
        """
        Win/loss ratio, expectancy, and hit rate.
        
        Calculates the ratio of average winning returns to average losing returns,
        expected value per trade, and fraction of positive returns.
        
        Returns
        -------
        tuple[float, float, float]
            (avg_win_loss_ratio, expectancy, hit_rate)
            
            - avg_win_loss_ratio: Mean(wins) / |Mean(losses)|
            - expectancy: Probability-weighted expected return per trade
            - hit_rate: Fraction of trades that are winning
            
            Returns (np.nan, np.nan, np.nan) if no wins or no losses.
            
        Raises
        ------
        ValueError
            If returns array is empty.
        """
        returns = self.core.returns
        if len(returns) == 0:
            raise ValueError("returns array is empty")
            
        returns = returns[returns != 0]

        if len(returns) == 0:
            logger.debug("No non-zero returns, returning NaN for win/loss metrics")
            return (np.nan, np.nan, np.nan)

        wins = self.wins
        losses = self.losses

        if len(wins) == 0 or len(losses) == 0:
            logger.debug("No wins or losses, returning NaN for win/loss metrics")
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
        
        Useful for risk/reward assessment. Ratios > 2.0 indicate strong edge.
        
        Returns
        -------
        float
            Gross profits / |Gross losses|. Returns np.nan if no losses.
            
        Raises
        ------
        ValueError
            If returns array is empty.
        """
        returns = self.core.returns
        if len(returns) == 0:
            raise ValueError("returns array is empty")
            
        gross_profit = returns[returns > 0].sum()
        gross_loss   = abs(returns[returns < 0].sum())

        if gross_loss == 0:
            logger.debug("No losses in returns, returning np.nan for profit_factor")
            return np.nan
        
        return float(gross_profit / gross_loss)
    

    @property
    def skew(self) -> float:
        """
        Skewness of the return distribution.
        
        Positive skew: distribution favors upside (right tail).
        Negative skew: distribution favors downside (left tail).
        
        Returns
        -------
        float
            Skewness coefficient (dimensionless).
            
        Raises
        ------
        ValueError
            If log_returns array is empty.
        """
        if len(self.core.log_returns) == 0:
            raise ValueError("log_returns array is empty")
        return float(skew(self.core.log_returns, bias=False))
    

    @property
    def kurtosis(self) -> float:
        """
        Kurtosis (tail heaviness) of the return distribution.
        
        High kurtosis: extreme returns more frequent (fat tails).
        Low kurtosis: lighter tails than normal distribution.
        Normal distribution kurtosis ≈ 3 (Fisher kurtosis = 0 with bias=False).
        
        Returns
        -------
        float
            Kurtosis coefficient (dimensionless, excess kurtosis via bias=False).
            
        Raises
        ------
        ValueError
            If log_returns array is empty.
        """
        if len(self.core.log_returns) == 0:
            raise ValueError("log_returns array is empty")
        return float(kurtosis(self.core.log_returns, bias=False))
    
