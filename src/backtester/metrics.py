import numpy as np
import pandas as pd


# Performance

    # Cumulative Return: did it make money overall?

    # CAGR: comparable growth rate across different test lengths.

# Risk

    # Annualised Volatility: how bumpy is the ride?

    # Max Drawdown (MDD): worst peak-to-trough loss (survivability).

# Risk-adjusted

    # Sharpe: return per unit risk (baseline comparator).

# Trade quality

    # Hit Rate: % of bars positive (sanity / style).

    # Profit Factor: wins vs losses (quick health check).

# Trading intensity / cost sensitivity

    # Turnover (avg, annualised): how much you trade (fees sensitivity).

# Perps-specific decomposition (since you’re doing funding)

    # Total Funding Return: carry contribution.

    # Total Fees Return: cost drag.


def compute_metrics(
        pnl_df: pd.DataFrame
        
    ):

    if pnl_df is None:
        raise ValueError("`df` argument is required and cannot be None.")

    if not isinstance(pnl_df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")

    required_cols = {"strategy_ret", "equity", "held_pos",
                     "trade", "fees", "funding_pnl"}
    missing_cols = required_cols - set(pnl_df.columns)

    if missing_cols:
        raise ValueError(
            f"`df` is missing required columns: {missing_cols}"
        )
    print("All checks passed.")
    
    return 'hi'
