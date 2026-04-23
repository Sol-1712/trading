import numpy as np
from functools import cached_property
from backtester.utils import compute_sharpe


# Avg Leverage Used ? 
# Long PnL vs Short PnL split — tells you which side of the book is generating alpha



class PositionMetrics:
    """
    Computes position-based metrics from a CoreStats object.

    This class provides key metrics:
    - Avg position size
    - Max position size
    - Time short (%)
    - Time long (%) 
    - Time flat (%)
    - Time in market (%)
    - Avg long size
    - Avg short size
    - Largest Loss
    - Largest Win
    - Avg holding period
    - Long sharpe
    - Short sharpe
    - Long pnl
    - Short pnl


    Attributes:
        core (CoreStats): Precomputed core statistics and returns from PnL.
    """

    def __init__(self, core):
        """
        Initializes PositionMetrics with a CoreStats object.

        Args:
            core (CoreStats): Object containing primitive statistics and returns.
        """
        self.core = core


    @cached_property
    def long_mask(self) -> np.ndarray:
        return self.core.held_pos > 0


    @cached_property
    def short_mask(self) -> np.ndarray:
        return self.core.held_pos < 0


    @cached_property
    def flat_mask(self) -> np.ndarray:
        return self.core.held_pos == 0
    

    @cached_property
    def _long_returns(self) -> np.ndarray:
        return self.core.returns[self.long_mask]
    

    @cached_property
    def _short_returns(self) -> np.ndarray:
        return self.core.returns[self.short_mask]


    @property
    def avg_long_size(self) -> float:
        long_pos = self.core.held_pos[self.long_mask]
        return float(np.mean(long_pos)) if len(long_pos) > 0 else 0.0
    

    @property
    def avg_short_size(self) -> float:
        short_pos = self.core.held_pos[self.short_mask]
        return float(np.mean(short_pos)) if len(short_pos) > 0 else 0.0


    @property
    def avg_position_size(self) -> float:
        """ Average absolute trade size as a fraction of equity. """
        return float(np.mean(np.abs(self.core.held_pos)))
    

    @property
    def max_position_size(self) -> float:
        """ Max absolute trade size as a fraction of equity. """
        return float(np.max(np.abs(self.core.held_pos)))
    

    @property
    def time_long(self) -> float:
        """ % of time long. """
        return float(np.mean(self.long_mask))


    @property
    def time_short(self) -> float:
        """ % of time short. """
        return float(np.mean(self.short_mask))


    @property
    def time_flat(self) -> float:
        """ % of time flat. """
        return float(np.mean(self.flat_mask))


    @property
    def time_in_market(self) -> float:
        """ % of time in the market. """
        return 1.0 - self.time_flat
    

    @property
    def largest_win(self) -> float:
        """Largest single bar return (%)."""
        return float(np.max(self.core.returns))


    @property
    def largest_loss(self) -> float:
        """Largest single bar loss (negative value) (%)."""
        return float(np.min(self.core.returns))
    

    @cached_property
    def avg_holding_period(self) -> float:
        """
        Average number of bars between position changes.
        
        Returns:
            float: Average holding period in bars. 0 if fewer than 2 trades.
        """
        trade_bars = np.where(self.core.trade != 0)[0]
        if len(trade_bars) < 2:
            return 0.0
        return float(np.mean(np.diff(trade_bars)))
    

    @property
    def long_sharpe(self) -> float:
        """Annualised Sharpe ratio on long bars only."""
        if len(self._long_returns) == 0:
            return np.nan
    
        sharpe = compute_sharpe(
            self._long_returns,
            rf = self.core.rf, 
            ann_factor=self.core.ann_factor
            )
        
        return float(sharpe)


    @property
    def short_sharpe(self) -> float:
        """Annualised Sharpe ratio on short bars only."""
        if len(self._short_returns) == 0:
            return np.nan
        sharpe = compute_sharpe(
            self._short_returns,
            rf = self.core.rf, 
            ann_factor=self.core.ann_factor
            )
        
        return float(sharpe)


    @property
    def long_pnl_pct(self) -> float:
        """ Fraction of total pnl generated from long positions."""
        total = np.sum(self.core.returns)
        if total == 0:
            return np.nan
        return float(np.sum(self._long_returns) / abs(total))
    

    @property
    def short_pnl_pct(self) -> float:
        """Fraction of total pnl generated from short positions."""
        total = np.sum(self.core.returns)
        if total == 0:
            return np.nan
        return float(np.sum(self._short_returns) / abs(total))