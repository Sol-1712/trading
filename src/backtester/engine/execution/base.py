from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

from backtester.engine.execution import ExecutionConfig
from backtester.portfolio.base import Portfolio


class ExecutionEngine(ABC):
    """
    Abstract execution engine.

    Each strategy type requires a different simulation model.
    The runner holds an ExecutionEngine and calls run() — it 
    does not need to know which concrete engine it has.
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
        Simulate execution over the full data period.

        Parameters
        ----------
        targets : pd.Series
            Sized position targets as signed fractions of equity.
            Aligned to data.index. Pre-delay is NOT applied — 
            the engine applies delay_bars from config.
        data : pd.DataFrame
            Enriched market data. Must contain price_col and 
            optionally funding_rate.
        config : ExecutionConfig
            Fee rate, delay bars, leverage limits.
        capital : float
            Initial capital in quote currency.
        price_col : str
            Column in data to use as execution price.
            Resolved by the runner before calling this method.
        """
        ...