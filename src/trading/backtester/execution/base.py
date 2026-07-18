from __future__ import annotations
import logging
from abc import ABC, abstractmethod
import pandas as pd

from trading.backtester.engine.config_bases import ExecutionConfig
from trading.backtester.portfolio.base import PortfolioSnapshot
from trading.backtester.fill import Order, Fill

logger = logging.getLogger(__name__)


class ExecutionEngine(ABC):
    """
    Abstract execution engine for order processing and fill simulation.
    
    Manages order queuing, fill model integration, and state tracking across
    the backtest. Concrete implementations (e.g., PerpDirectionalEngine)
    handle strategy-specific order generation logic.
    
    Each backtest uses exactly one ExecutionEngine instance, selected based
    on strategy type. The runner delegates all execution to this engine.
    
    Parameters
    ----------
    config : ExecutionConfig
        Execution configuration including fees, delays, fill model, and limits.
    """

    def __init__(self, config: ExecutionConfig) -> None:


        self.config                     = config
        self._fill_model                = config.fill_model_cls(fee_rate=config.fee_rate)
        self._queue:   dict[int, Order] = {}
        self._pending: list[Order]      = []
        self._pending_notional: float   = 0.0
        self._current_bar: int          = 0
        self._active: list[Order]       = []
        
        logger.debug("ExecutionEngine initialized with config: %s", type(config).__name__)

    @property
    def price_type(self):
        return self._fill_model.price_type

    @abstractmethod
    def submit(self, 
               target_frac: float,
               state: PortfolioSnapshot,
               bar: pd.Series) -> None:
        """
        Submit a position target to the execution engine.
        
        Concrete implementations convert target to an Order and queue it.
        May be delayed (delay_bars) before attempting execution.
        
        Parameters
        ----------
        target_frac : float
            Desired signed position fraction of equity (e.g., 0.5 = 50% long).
        state : PortfolioSnapshot
            Current portfolio state (used for position delta calculation).
        bar : pd.Series
            Current bar data.
        """


    @abstractmethod
    def execute_pending(self, bar: pd.Series, t: int) -> list[Fill]:
        """
        Attempt to fill all pending orders against current bar.
        
        Delegates to fill model for each pending order. Removes fully filled
        orders from queue. Returns all fills executed this bar.
        
        Parameters
        ----------
        bar : pd.Series
            Current bar OHLC data and market information.
        t : int
            Current bar index.
            
        Returns
        -------
        list[Fill]
            List of fills executed this bar (may be empty).
        """

    def _cancel_all_pending(self) -> None:
        """Clear all pending and queued orders. Called on liquidation or error."""
        self._queue.clear()
        self._pending.clear()
        self._pending_notional = 0.0
        logger.debug("Cancelled all pending orders")