from __future__ import annotations
import logging
from abc import ABC, abstractmethod
import pandas as pd

from trading.backtester.engine.config_bases import ExecutionConfig
from trading.backtester.fill import Order, Fill
from trading.data_utils.core import PriceType

logger = logging.getLogger(__name__)


class ExecutionEngine(ABC):
    """
    Abstract execution engine for order processing and fill simulation.

    Manages order queuing, fill-model integration, and execution-layer state
    across the backtest. Concrete implementations (e.g. PerpDirectionalEngine)
    handle strategy-specific order generation.

    Each backtest uses exactly one ExecutionEngine instance. The runner
    delegates all execution to this engine.

    Parameters
    ----------
    config : ExecutionConfig
        Execution configuration including fees, delay_bars, and fill model.
    """

    def __init__(self, config: ExecutionConfig) -> None:

        self.config                     = config
        self._fill_model                = config.fill_model_cls(fee_rate=config.fee_rate)
        self._queue:                    dict[int, list[Order]] = {}
        self._pending_notional: float   = 0.0
        self._current_bar: int          = 0
        self._active: list[Order]       = []
        
        logger.debug("ExecutionEngine initialized with config: %s", type(config).__name__)

    @property
    def price_type(self) -> PriceType:
        """
        Price series required by the configured fill model.
        """
        return self._fill_model.price_type
    
    @property
    def pending_notional(self) -> float:
        """
        The total notional value of all pending orders.
        """
        return self._pending_notional

    @abstractmethod
    def submit(
        self,
        timestamp: pd.Timestamp,
        delta_notional: float,
        immediate: bool = False,
    ) -> None:
        """Add a new order to the queue."""

    @abstractmethod
    def execute_pending(self, bar: pd.Series, t: int) -> list[Fill]:
        """
        Attempt to fill all pending orders against the current bar.

        Delegates to the fill model for each pending order. Fully filled
        orders are removed from the active set.

        Parameters
        ----------
        bar : pd.Series
            Current bar OHLC and market data.
        t : int
            Current bar index.

        Returns
        -------
        list[Fill]
            Fills executed this bar (may be empty).
        """

    def cancel_all_pending(self) -> None:
        """Clear all pending and queued orders. Called on flatten or error."""
        self._queue.clear()
        self._active.clear()
        self._pending_notional = 0.0
        logger.debug("Cancelled all pending orders")