import logging
import numpy as np
from functools import cached_property
from crypto_quant.backtester.utils import compute_sharpe

logger = logging.getLogger(__name__)

class PositionMetrics:
    """
    Computes position-based metrics from a CoreStats object.
    
    Analyzes position behavior including time spent long/short/flat, average
    position sizes, holding periods, and decomposed Sharpe ratios by side.
    
    Parameters
    ----------
    core : CoreStats
        Precomputed core statistics with position data and returns.
        
    Raises
    ------
    ValueError
        If core is None or missing required attributes.
    """

    def __init__(self, core):
        if core is None:
            raise ValueError("core cannot be None")
        if not hasattr(core, 'held_pos') or core.held_pos is None:
            raise ValueError("core must have 'held_pos' attribute")
        if not hasattr(core, 'returns') or core.returns is None:
            raise ValueError("core must have 'returns' attribute")
            
        self.core = core
        logger.debug("PositionMetrics initialized with %d observations", core.n_obs)


    @cached_property
    def long_mask(self) -> np.ndarray:
        """Boolean mask where position > 0."""
        return self.core.held_pos > 0


    @cached_property
    def short_mask(self) -> np.ndarray:
        """Boolean mask where position < 0."""
        return self.core.held_pos < 0


    @cached_property
    def flat_mask(self) -> np.ndarray:
        """Boolean mask where position == 0."""
        return self.core.held_pos == 0
    

    @cached_property
    def _long_returns(self) -> np.ndarray:
        """Returns during long positions."""
        return self.core.returns[self.long_mask]
    

    @cached_property
    def _short_returns(self) -> np.ndarray:
        """Returns during short positions."""
        return self.core.returns[self.short_mask]


    @property
    def avg_long_size(self) -> float:
        """
        Average position size during long periods.
        
        Returns
        -------
        float
            Mean signed position when held_pos > 0. 0.0 if no long periods.
        """
        long_pos = self.core.held_pos[self.long_mask]
        return float(np.mean(long_pos)) if len(long_pos) > 0 else 0.0
    

    @property
    def avg_short_size(self) -> float:
        """
        Average position size during short periods.
        
        Returns
        -------
        float
            Mean signed position when held_pos < 0. 0.0 if no short periods.
        """
        short_pos = self.core.held_pos[self.short_mask]
        return float(np.mean(short_pos)) if len(short_pos) > 0 else 0.0


    @property
    def avg_position_size(self) -> float:
        """
        Average absolute position size.
        
        Returns
        -------
        float
            Mean of |held_pos| across all bars.
        """
        return float(np.mean(np.abs(self.core.held_pos)))
    

    @property
    def max_position_size(self) -> float:
        """
        Maximum absolute position size.
        
        Returns
        -------
        float
            Max of |held_pos| across all bars.
        """
        return float(np.max(np.abs(self.core.held_pos)))
    

    @property
    def time_long(self) -> float:
        """
        Fraction of bars spent in long positions.
        
        Returns
        -------
        float
            Mean of long_mask (0.0 to 1.0).
        """
        return float(np.mean(self.long_mask))


    @property
    def time_short(self) -> float:
        """
        Fraction of bars spent in short positions.
        
        Returns
        -------
        float
            Mean of short_mask (0.0 to 1.0).
        """
        return float(np.mean(self.short_mask))


    @property
    def time_flat(self) -> float:
        """
        Fraction of bars with no position.
        
        Returns
        -------
        float
            Mean of flat_mask (0.0 to 1.0).
        """
        return float(np.mean(self.flat_mask))


    @property
    def time_in_market(self) -> float:
        """
        Fraction of bars in any position (long or short).
        
        Returns
        -------
        float
            1.0 - time_flat.
        """
        return 1.0 - self.time_flat
    

    @property
    def largest_win(self) -> float:
        """
        Largest single-bar gain.
        
        Returns
        -------
        float
            Maximum return value across all bars.
        """
        return float(np.max(self.core.returns))


    @property
    def largest_loss(self) -> float:
        """
        Largest single-bar loss.
        
        Returns
        -------
        float
            Minimum return value (negative).
        """
        return float(np.min(self.core.returns))
    

    @cached_property
    def avg_holding_period(self) -> float:
        """
        Average bars between position changes.
        
        Computes mean interval between trade signals.
        
        Returns
        -------
        float
            Average holding period in bars. 0.0 if fewer than 2 trades.
        """
        trade_bars = np.where(self.core.trade != 0)[0]
        if len(trade_bars) < 2:
            logger.debug("Fewer than 2 trades, avg_holding_period = 0.0")
            return 0.0
        return float(np.mean(np.diff(trade_bars)))
    

    @property
    def long_sharpe(self) -> float:
        """
        Sharpe ratio during long positions only.
        
        Returns
        -------
        float
            Annualized Sharpe ratio computed from returns when held_pos > 0.
            Returns np.nan if no long periods.
        """
        if len(self._long_returns) == 0:
            logger.debug("No long returns, long_sharpe = nan")
            return np.nan
    
        sharpe = compute_sharpe(
            self._long_returns,
            rf = self.core.rf, 
            ann_factor=self.core.ann_factor
            )
        
        return float(sharpe)


    @property
    def short_sharpe(self) -> float:
        """
        Sharpe ratio during short positions only.
        
        Returns
        -------
        float
            Annualized Sharpe ratio computed from returns when held_pos < 0.
            Returns np.nan if no short periods.
        """
        if len(self._short_returns) == 0:
            logger.debug("No short returns, short_sharpe = nan")
            return np.nan
        sharpe = compute_sharpe(
            self._short_returns,
            rf = self.core.rf, 
            ann_factor=self.core.ann_factor
            )
        
        return float(sharpe)


    @property
    def long_pnl_pct(self) -> float:
        """
        Fraction of total PnL from long positions.
        
        Returns
        -------
        float
            Sum(long_returns) / |Total_returns|. Returns np.nan if total is zero.
        """
        total = np.sum(self.core.returns)
        if total == 0:
            logger.debug("Total returns = 0, long_pnl_pct = nan")
            return np.nan
        return float(np.sum(self._long_returns) / abs(total))
    

    @property
    def short_pnl_pct(self) -> float:
        """
        Fraction of total PnL from short positions.
        
        Returns
        -------
        float
            Sum(short_returns) / |Total_returns|. Returns np.nan if total is zero.
        """
        total = np.sum(self.core.returns)
        if total == 0:
            logger.debug("Total returns = 0, short_pnl_pct = nan")
            return np.nan
        return float(np.sum(self._short_returns) / abs(total))