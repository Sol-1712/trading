from backtester.engine.execution.base import ExecutionEngine
from backtester.engine.execution import ExecutionConfig
from backtester.portfolio.base import Portfolio

import pandas as pd


class PerpDirectionalEngine(ExecutionEngine):
    """
    Execution engine for directional perpetual futures strategies.

    Responsible for:
    - Validating inputs
    - Applying delay_bars to targets
    - Constructing Portfolio with correct config
    - Driving the bar-by-bar loop
    - Extracting funding_rate from data if present

    NOT responsible for:
    - MTM, fee, or funding arithmetic (Portfolio.step())
    - Price column resolution (runner)
    - Signal generation or position sizing (strategy / sizer)
    """

    def run(
        self,
        targets:   pd.Series,
        data:      pd.DataFrame,
        config:    ExecutionConfig,
        capital:   float,
        price_col: str,
    ) -> Portfolio:
        self._validate(targets, data, price_col)

        pass


    def _validate(
        self,
        targets:   pd.Series,
        data:      pd.DataFrame,
        price_col: str,
    ) -> None:
        """
        Validate alignment, price column existence, 
        positive prices, and non-empty inputs.
        """
        ...

        pass