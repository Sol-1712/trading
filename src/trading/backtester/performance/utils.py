import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

SECONDS_TO_PERIODS_24_7: dict[int, tuple[str, int]] = {
    60:     ("1min",  365 * 24 * 60),
    300:    ("5min",  365 * 24 * 12),
    900:    ("15min", 365 * 24 * 4),
    3600:   ("1H",    365 * 24),      
    86400:  ("1D",    365),
    604800: ("1W",    52),
}

METRICS = {
    "returns": [
        "mean_bar_return","cagr", "net_return", "gross_return", "skew",
         "kurtosis", "hit_rate_all", "median_bar_return", "avg_bar_win", "avg_bar_loss",
         "avg_bar_win_loss_ratio", "largest_bar_return", "smallest_bar_return",
    ],
    "risk": [
        "max_drawdown", "max_drawdown_duration", "avg_drawdown_duration", "avg_drawdown",
        "time_in_drawdown", "sharpe", "annualised_sharpe", "long_sharpe", "short_sharpe",
        "sortino", "calmar", "var_95", "var_99", "cvar_95",
        "cvar_99", "downside_deviation","volatility", "longest_losing_streak",
    ],
    "cost": [
        "total_fee_return", "total_funding_return", "total_cost_return",
        "total_fee_pct_of_net", "funding_pct_of_net", "total_cost_pct_of_net",
        "cost_to_gross_ratio", "fee_drag_sharpe", "funding_drag_sharpe",
        "avg_fee_per_bar", "avg_fee_per_trade", "annualised_turnover", "pct_bars_paying_funding",
    ],
    "trade": [
        "num_trades", "avg_position_size", "max_position_size", "avg_long_size",
        "avg_short_size", "time_long", "time_short", "time_flat",
        "expectancy", "profit_factor", "hit_rate_trade", "time_in_market",
        "avg_holding_period", "largest_win", "largest_loss",
        "long_pnl_pct", "short_pnl_pct",
    ],
}

def infer_ann_factor(
    dtindex: pd.DatetimeIndex
    ) -> tuple[str, float]:
    """
    Infer annualization factor from datetime index.
    
    Computes median time difference between consecutive bars and maps to
    a standard frequency. Returns the frequency string and periods per year
    for use in annualization of volatility and Sharpe ratios.
    
    Parameters
    ----------
    dtindex : pd.DatetimeIndex
        Index of bar timestamps (assumed in ascending order).
        
    Returns
    -------
    tuple[str, float]
        (frequency_string, periods_per_year)
        
        Frequency string (e.g., "1H", "1D") maps to known market conventions.
        Periods per year is used for scaling (e.g., sharpe *= sqrt(periods)).
        
        Falls back to ("1H", 365*24) if median difference not recognized.
        
    Raises
    ------
    ValueError
        If dtindex is None or has fewer than 2 elements.
    """
    if dtindex is None or len(dtindex) < 2:
        raise ValueError("dtindex must have at least 2 elements")
    
    median_diff = pd.Series(dtindex).diff().dt.total_seconds().median()
    
    if np.isnan(median_diff):
        logger.warning("Could not compute median time difference, assuming 1H")
        return ("1H", 365 * 24)
    
    freq, periods = SECONDS_TO_PERIODS_24_7.get(int(median_diff), ("1H", 365 * 24))
    logger.debug("Inferred frequency: %s (median_diff=%.0f seconds)", freq, median_diff)
    return freq, float(periods)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """
    Element-wise division with zero denominator protection.

    Divides numerator by denominator element-wise, returning 0.0 for any
    position where the denominator is zero rather than producing inf or nan.

    Parameters
    ----------
    numerator : np.ndarray
        Array of values to divide.
    denominator : np.ndarray
        Array of values to divide by. Zero elements are handled safely.

    Returns
    -------
    np.ndarray
        Result of numerator / denominator, with 0.0 where denominator == 0.
    """
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0
    )


def compute_sharpe(returns: np.ndarray) -> float:
    """
    Compute per-period Sharpe ratio: mean(returns) / std(returns, ddof=1).

    Does not subtract a risk-free rate — pass excess returns if needed.
    Does not annualise — multiply by sqrt(periods_per_year) externally.

    Parameters
    ----------
    returns : np.ndarray
        Per-period returns (decimal form, e.g. 0.01 = 1%).

    Returns
    -------
    float
        Per-period Sharpe ratio. np.nan if volatility is zero.

    Raises
    ------
    ValueError
        If returns is None, empty, or contains NaN.
    """

    if returns is None or len(returns) == 0:
        raise ValueError("returns array cannot be None or empty")

    if np.isnan(returns).any():
        raise ValueError("returns contain NaN values")

    std = np.std(returns, ddof=1)

    if np.isclose(std, 0):
        return np.nan

    sharpe = np.mean(returns) / std

    return float(sharpe)