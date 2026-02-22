import numpy as np
import pandas as pd




DEFAULT_METRICS = [
    "sharpe", "cum_ret", "cagr", "volatility",
    "max_drawdown", "hit_rate",
    "avg_win_loss_ratio", "profit_factor",
    "max_consecutive_losses", "avg_drawdown_duration",
    "total_funding_return", "total_fees"
]

def compute_metrics(
        pnl_df: pd.DataFrame
    ):

    if pnl_df is None:
        raise ValueError("`df` argument is required and cannot be None.")

    if not isinstance(pnl_df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")


    return
