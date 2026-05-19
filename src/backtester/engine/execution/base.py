from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

from backtester.engine.execution import ExecutionConfig
from backtester.portfolio.base import Portfolio


class ExecutionEngine(ABC):
    """
    Abstract base for execution simulation engines.

    Each strategy type (directional, market making, arbitrage) 
    requires a different execution model. Concrete engines implement 
    this interface so the runner can treat them uniformly.
    """

    @abstractmethod
    def run(
        self,
        targets:   pd.Series,
        data:      pd.DataFrame,
        config:    ExecutionConfig,
        capital:   float,
        price_col: str,
    ) -> Portfolio:
        """
        Simulate execution and return a Portfolio.

        Parameters
        ----------
        targets : pd.Series
            Desired position at each bar, in base asset units.
            Signed: positive = long, negative = short.
        data : pd.DataFrame
            Full enriched market data including funding_rate if available.
        config : ExecutionConfig
            Fee rate, delay bars, leverage limits.
        capital : float
            Initial capital in quote currency.
        price_col : str
            Column in data to use as execution price.
        """
        ...