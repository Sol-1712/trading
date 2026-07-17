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
        "mean_return","cagr", "net_return", "gross_return", "avg_win_loss", 
         "skew", "kurtosis", "hit_rate_all", "median_bar_return", "avg_bar_win", "avg_bar_loss",
         "avg_bar_win_loss_ratio", "largest_bar_return", "smallest_bar_return",
    ],
    "risk": [
        "max_drawdown", "max_drawdown_duration", "avg_drawdown_duration", "avg_drawdon",
        "time_in_drawdown", "sharpe", "annualised_sharpe",
        "sortino","calmar", "var_95", "var_99", "cvar_95",
        "cvar_99", "downside_deviation","volatility", "longest_losing_streak",
    ],
    "cost": [
        "total_fee_return", "total_funding_return", "total_cost_return",
        "total_fee_pct_of_net", "funding_pct_of_net", "total_cost_pct_of_net",
        "cost_to_gross_ratio", "fee_drag_sharpe", "funding_drag_sharpe",
        "expectancy", "profit_factor",
        "avg_fee_per_bar", "avg_fee_per_trade", "annualised_turnover", "pct_bars_paying_funding",
    ],
    "position": [
        "num_trades", "avg_position_size", "max_position_size", "avg_long_size",
        "avg_short_size", "time_long", "time_short", "time_flat", "hit_rate"
        "time_in_market", "avg_holding_period", "largest_win", "largest_loss",
        "long_pnl_pct", "short_pnl_pct", "long_sharpe", "short_sharpe",
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


def compute_sharpe(returns: np.ndarray, rf: float, ann_factor: float | None) -> float:
    """
    Compute Sharpe ratio from a return series.
    
    Calculates excess return (returns - risk-free rate) divided by volatility.
    Optionally annualizes by multiplying by sqrt(periods_per_year).
    
    Parameters
    ----------
    returns : np.ndarray
        Per-period returns (1D array). Should be decimal form (e.g., 0.01 = 1%).
    rf : float
        Per-period risk-free rate (log-transformed). Typically close to 0.
    ann_factor : float | None
        Annualization factor (periods per year). If provided, result is scaled
        by sqrt(ann_factor). If None, returns per-period Sharpe.
        
    Returns
    -------
    float
        Sharpe ratio. Returns np.nan if returns have zero volatility.
        
    Raises
    ------
    ValueError
        If returns array is None or empty.
    """
    if returns is None or len(returns) == 0:
        raise ValueError("returns array cannot be None or empty")
    
    std = np.std(returns, ddof=1)

    if std == 0:
        logger.warning("Returns have zero volatility, returning NaN for Sharpe")
        return np.nan
    
    excess = returns - np.log1p(rf)
    sharpe = float(np.mean(excess) / std)

    if ann_factor is not None:
        if ann_factor <= 0:
            raise ValueError(f"ann_factor must be positive, got {ann_factor}")
        sharpe *= float(np.sqrt(ann_factor))

    return sharpe

