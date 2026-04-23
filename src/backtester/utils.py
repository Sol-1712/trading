import pandas as pd
import numpy as np

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
        "cagr", "net_return", "gross_return", "sharpe", "annualised_sharpe",
        "sortino", "volatility", "hit_rate_all", "hit_rate_trade",
        "avg_win_loss", "expectancy", "profit_factor", "skew", "kurtosis",
    ],
    "risk": [
        "max_drawdown", "max_drawdown_duration", "avg_drawdown_duration",
        "time_in_drawdown", "calmar", "var_95", "var_99", "cvar_95",
        "cvar_99", "downside_deviation", "longest_losing_streak",
    ],
    "cost": [
        "total_fee_return", "total_funding_return", "total_cost_return",
        "total_fee_pct_of_net", "funding_pct_of_net", "total_cost_pct_of_net",
        "cost_to_gross_ratio", "fee_drag_sharpe", "funding_drag_sharpe",
        "avg_fee_per_bar", "annualised_turnover", "pct_bars_paying_funding",
    ],
    "position": [
        "avg_position_size", "max_position_size", "avg_long_size",
        "avg_short_size", "time_long", "time_short", "time_flat",
        "time_in_market", "avg_holding_period", "largest_win", "largest_loss",
        "long_pnl_pct", "short_pnl_pct", "long_sharpe", "short_sharpe",
    ],
}

def infer_ann_factor(
    dtindex: pd.DatetimeIndex
    ) -> tuple[str, float]:

    """
    Infers the annualisation factor based of timestamp index

    Returns:
    freq: bar interval frequency (defualt 1H)
    periods: number of intervals per year
    """
    median_diff = pd.Series(dtindex).diff().dt.total_seconds().median()
    freq, periods = SECONDS_TO_PERIODS_24_7.get(int(median_diff), ("1H", 365 * 24))
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


def compute_sharpe(returns: np.ndarray, rf: float, ann_factor: float = None) -> float:
    """
    Compute the per-bar Sharpe ratio for a series of returns.

    Parameters
    ----------
    returns : np.ndarray
        Per-period returns.
    rf : float
        Per period risk-free rate (log-transformed).
    ann_factor: float
        Optional annualisation factor.

    Returns
    -------
    float
        Sharpe ratio.
    """
    std = np.std(returns, ddof=1)

    if std == 0:
        return np.nan
    
    excess = returns - np.log1p(rf)
    sharpe = float(np.mean(excess) / std)

    if ann_factor is not None:
        sharpe *= float(np.sqrt(ann_factor))

    return sharpe