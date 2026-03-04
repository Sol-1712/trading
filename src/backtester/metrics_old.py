import numpy as np
import pandas as pd



# Only works for 24/7 markets
SECONDS_TO_PERIODS_247: dict[int, tuple[str, int]] = {
    60:     ("1min",  365 * 24 * 60),
    300:    ("5min",  365 * 24 * 12),
    900:    ("15min", 365 * 24 * 4),
    3600:   ("1H",    365 * 24),      
    86400:  ("1D",    365),
    604800: ("1W",    52),
}

# Helper Function
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
    freq, periods = SECONDS_TO_PERIODS_247.get(int(median_diff), ("1H", 365 * 24))
    return freq, float(periods)


# --- RETURN METRICS ----------------------------------------------------------------
def _equity_curve(
    returns: pd.Series
    ) -> pd.Series:

    return (1 + returns).cumprod()


def _cagr(
    returns: pd.Series,
    ann_factor: float
    ) -> float:

    if returns.empty:
        return 0.0

    compounded = (1.0 + returns).prod()
    n_years = len(returns) / ann_factor

    if n_years <= 0:
        return 0.0

    return float(compounded ** (1.0 / n_years) - 1.0)


def _sharpe(
    returns: pd.Series,
    ann_factor: float,
    rf: float = 0 
    ) -> float:

    if returns.std() == 0:
        return 0.0
    
    return float(((returns.mean()) / returns.std()) * np.sqrt(ann_factor))


def _sortino(
    returns: pd.Series,
    ann_factor: float,
    rf: float = 0
    ) -> float:
    
    downside = np.minimum(0, returns)
    
    dd_std = np.sqrt((downside ** 2).mean())
    if dd_std == 0:
        return np.nan
    
    return float(((returns.mean() - rf) / dd_std) * np.sqrt(ann_factor))


def _volatility(
    returns: pd.Series,
    ann_factor: float,
    ) -> float:

    return float(returns.std() * np.sqrt(ann_factor))
   

def _hit_rate(
    returns: pd.Series,
    ) -> float:
    return float((returns > 0).mean())


def _avg_win_loss_ratio_expectancy(
    returns : pd.Series,
) -> tuple[float, float, float]:
    
    returns = returns[returns != 0]

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    hit_rate = _hit_rate(returns)

    if losses.empty or wins.empty:
        return (np.nan, np.nan)

    avg_win_loss_ratio = float(wins.mean() / abs(losses.mean()))
    expectancy = (hit_rate * wins.mean()) + ((1 - hit_rate) * losses.mean())

    return (avg_win_loss_ratio, expectancy, hit_rate)

def _profit_factor(
    returns: pd.Series
) -> float:
    
    gross_profit = returns[returns > 0].sum()
    gross_loss   = abs(returns[returns < 0].sum())

    if gross_loss == 0:
        return np.nan
    
    return float(gross_profit / gross_loss)


def return_metrics(
    returns: pd.Series,
    ann_factor: float,
    freq: str,
    ) -> dict:
    
    avg_win_loss, expectancy, hit_rate_trade = _avg_win_loss_ratio_expectancy(returns)
    return {
        "Frequency":              freq,
        "CAGR":                   _cagr(returns, ann_factor),
        "Sharpe":                 _sharpe(returns, ann_factor),
        "Sortino":                _sortino(returns, ann_factor),
        "Annualised Volatility":  _volatility(returns, ann_factor),
        "Hit Rate (All Bars)":    _hit_rate(returns),
        "Hit Rate (Trade Bars)":  hit_rate_trade,
        "Avg Win/Loss Ratio":     avg_win_loss if not np.isnan(avg_win_loss) else None,
        "Expectancy":             expectancy if not np.isnan(expectancy) else None,
        "Profit Factor":          _profit_factor(returns),
    }

# --- RISK METRICS --------------------------------------------------------------------------------


def _max_drawdown(
    equity_curve: pd.Series,
    ) -> float:

    # Do eventually need to guard against 0 equity

    ec = equity_curve.to_numpy()
    running_peak = np.maximum.accumulate(ec)
    drawdown = (ec - running_peak) / running_peak
    return drawdown.min()
 
def _calmar(
    mdd: float ,   
    returns: pd.Series,
    ann_factor: float,
    ) -> float:
    
    if mdd ==0:
        return np.nan
    
    cagr = _cagr(returns, ann_factor)

    return (cagr / np.abs(mdd))
# Ann ret / abs(mdd)








def risk_metrics(
    returns: pd.Series,
    ) -> dict:

    equity_curve = _equity_curve(returns)

    MDD = _max_drawdown(equity_curve)
    return {
        "MDD": MDD,
    }

# --- COST METRICS --------------------------------------------------------------------------------

# def _cost_metrics(
#     pnl_df:   pd.DataFrame,
#     returns:  pd.Series,
#     ann_factor: float,
#     capital:  float,
#     ) -> dict:

#     fees        = pnl_df["fees ($)"]
#     funding     = pnl_df["funding_pnl ($)"]
#     equity      = pnl_df["equity ($)"]
#     pos_pnl     = pnl_df["position_pnl ($)"]

#     total_fees_dollar    = fees.sum()
#     total_funding_dollar = funding.sum()
#     gross_pnl            = pos_pnl.sum()
#     net_pnl              = pnl_df["strategy_pnl ($)"].sum()

#     # % of starting capital
#     total_fees_pct       = total_fees_dollar / capital
#     total_funding_pct    = total_funding_dollar / capital

#     # fee drag on sharpe: recompute sharpe on gross returns (before fees)
#     gross_returns        = returns + (fees / equity.shift(1).fillna(capital))
#     sharpe_gross         = _sharpe(gross_returns, ann_factor)
#     sharpe_net           = _sharpe(returns, ann_factor)
#     fee_drag_sharpe      = sharpe_gross - sharpe_net

#     fee_pct_of_gross     = total_fees_dollar / abs(gross_pnl) if gross_pnl != 0 else np.nan

#     return {
#         "Total Fees ($)":         f"${total_fees_dollar:,.2f}",
#         "Total Fees (% capital)": f"{total_fees_pct:.2%}",
#         "Total Funding ($)":      f"${total_funding_dollar:,.2f}",
#         "Total Funding (% capital)": f"{total_funding_pct:.2%}",
#         "Gross PnL ($)":          f"${gross_pnl:,.2f}",
#         "Net PnL ($)":            f"${net_pnl:,.2f}",
#         "Fees as % of Gross PnL": f"{fee_pct_of_gross:.2%}" if not np.isnan(fee_pct_of_gross) else "N/A",
#         "Fee Drag (Sharpe)":      f"{fee_drag_sharpe:.3f}",
#     }


# --- POSITION METRICS -----------------------------------------------------------------------------
### ADD TIME METRICS (lonmg, short, flat, total)
# def _position_metrics(
#     pnl_df: pd.DataFrame, freq: str
#     ) -> dict:

#     held    = pnl_df["held_pos (% of equity)"]
#     trade   = pnl_df["trade (% of equity)"]
#     td      = pnl_df["trade_dollars ($)"]

#     avg_pos         = held.abs().mean()
#     avg_long        = held[held > 0].mean() if (held > 0).any() else 0.0
#     avg_short       = held[held < 0].mean() if (held < 0).any() else 0.0
#     long_pct        = (held > 0).mean()
#     short_pct       = (held < 0).mean()
#     flat_pct        = (held == 0).mean()
#     turnover        = trade.abs().mean()
#     avg_trade_size  = td.abs()[td != 0].mean()

#     return {
#         "Avg Position Size":      f"{avg_pos:.2%}",
#         "Avg Long Size":          f"{avg_long:.2%}",
#         "Avg Short Size":         f"{avg_short:.2%}",
#         "Time Long":              f"{long_pct:.2%}",
#         "Time Short":             f"{short_pct:.2%}",
#         "Time Flat":              f"{flat_pct:.2%}",
#         "Avg Turnover (per bar)": f"{turnover:.2%}",
#         "Avg Trade Size ($)":     f"${avg_trade_size:,.2f}",
#     }



### MASTER CALL
def calc_metrics(
    pnl_df: pd.DataFrame,
    capital: float
    ) -> pd.DataFrame:
    """
    Compute full suite of metrics from backtest output DataFrame.

    Args:
        pnl_df:  DataFrame returned by pnl module, with DatetimeIndex.
        capital: Initial capital.

    Returns:
        pd.DataFrame with metrics grouped by category, readable in a notebook.
    """

    returns = pnl_df["returns_normalised"]
    freq, ann  = infer_ann_factor(pnl_df.index)
