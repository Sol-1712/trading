from backtester.engine.execution.base import ExecutionEngine
from backtester.engine.execution import ExecutionConfig
from backtester.portfolio.base import Portfolio

import pandas as pd


class PerpDirectionalEngine(ExecutionEngine):
    """
    Execution engine for directional perpetual futures strategies.

    Delegates all bar-level state management to Portfolio.step(),
    which enforces the correct order of operations:
        MTM → funding → target sizing → trade → fee
    """

    def run(
        self,
        targets:   pd.Series,
        data:      pd.DataFrame,
        config:    ExecutionConfig,
        capital:   float,
        price_col: str,
    ) -> Portfolio:

        self._validate_inputs(targets, data, price_col)

        price        = data[price_col]
        funding_rate = (
            data["funding_rate"]
            if "funding_rate" in data.columns
            else pd.Series(0.0, index=data.index)
        )

        # Apply execution delay before the loop —
        # the portfolio receives already-delayed targets.
        delayed = targets.shift(config.delay_bars).fillna(0.0)

        portfolio = Portfolio(
            initial_capital = capital,
            fee_rate        = config.fee_rate,
        )

        for ts in data.index:
            portfolio.step(
                timestamp       = ts,
                price           = price[ts],
                target_fraction = delayed[ts],
                funding_rate    = funding_rate[ts],
            )

        return portfolio

    def _validate_inputs(
        self,
        targets:   pd.Series,
        data:      pd.DataFrame,
        price_col: str,
    ) -> None:
        if price_col not in data.columns:
            raise ValueError(
                f"Execution price column '{price_col}' not in data. "
                f"Available: {list(data.columns)}"
            )
        if not targets.index.equals(data.index):
            raise ValueError(
                "targets and data indices do not align. "
                "Ensure position sizing runs on the same data used for features."
            )
        if data[price_col].le(0).any():
            raise ValueError(
                f"Column '{price_col}' contains non-positive prices. "
                "Check data quality."
            )