from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

from backtester.engine.execution import ExecutionConfig
from backtester.portfolio.base import Portfolio, PortfolioSnapshot
from backtester.engine.execution.fill import Order, Fill, MarketFillModel


class ExecutionEngine(ABC):
    """
    Abstract execution engine.

    Each strategy type requires a different simulation model.
    The runner holds an ExecutionEngine and calls run() — it 
    does not need to know which concrete engine it has.
    """

    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config
        self._fill_model = config.fill_model or MarketFillModel()
        self._queue:   dict[int, Order] = {}  # bar_index → Order (awaiting submission to fill model)
        self._pending: list[Order]      = []  # orders actively being attempted by fill model
        self._pending_fraction: float = 0.0   # Needs to be a list for multi asset



    @abstractmethod
    def submit(self, 
               target_frac: float,
               state: PortfolioSnapshot,
               bar: pd.Series) -> None:
        """Add order to pending queue."""


    @abstractmethod
    def execute_pending(self, bar: pd.Series) -> list[Fill]:
        """
        Attempt fills on all pending orders against current bar.
        Removes filled orders from queue.
        Returns all fills that occurred this bar.
        Also needs to cancel 'stuck' orders (not filling)
        i.e for directional, if previous order did not fill, cancel as new order units will = desired units.
        """

    def _cancel_all_pending(self) -> None:
        self._queue.clear()
        self._pending.clear()
        self._pending_fraction = 0.0