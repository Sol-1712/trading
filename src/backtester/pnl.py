import numpy as np
import pandas as pd
from numba import njit

TAKER_FEE = 0.000550
LEVERAGE = 1.0
DELAY_BARS = 1 # 1-bar execution delay
STARTING_CAPITAL = 100000.0
# Config File

# assuming execution at the close of bar t (i.e. the same price that triggered the signal), 
# which is slightly optimistic — in reality there's always some slippage between signal generation and fill.
# Fill at open[t+1]?
# price_ret[t] = (close[t] - open[t]) / open[t]  # shifted by your delay

### DOLLAR RETURNS
def pnl(
    data_df: pd.DataFrame,
    signals: pd.Series | np.ndarray,
    fee_rate: float = TAKER_FEE,
    capital: float = STARTING_CAPITAL,
    price_col: str = "mark_close",
    delay_bars: int = DELAY_BARS,
    leverage: float = LEVERAGE,
    funding_col: str = "fundingRate",
) -> pd.DataFrame:
    """
    Assumptions:
    - price returns are close-to-close using price_col
    - pos[t] is the target position chosen at close(t) (fraction of equity notional)
    - execution delay is delay_bars, so held_pos[t] = pos[t-delay_bars]
    - fees are fee_rate * notional_traded, where notional_traded = leverage * capital
    - funding applies on held position: funding_pnl[t] = funding_sign * held_pos[t] * fundingRate[t] * notional

    Returns:
    - asset change (% change)
    - strategy_pnl ($)
    """

    if price_col not in data_df.columns:
        raise ValueError(f"Missing column: {price_col}")
    if funding_col not in data_df.columns:
        raise ValueError(f"Missing column: {funding_col}")
    if not data_df.index.is_monotonic_increasing:
        raise ValueError("data_df index must be sorted increasing (time order)")
    
    prices = data_df[price_col].astype("float64").to_numpy()
    funding  = data_df[funding_col].astype("float64").to_numpy()

    pos = np.asarray(signals, dtype="float64")
    if len(pos) != len(prices):
        raise ValueError(f"pos length {len(pos)} != data length {len(prices)}")
    
    # Held position with explicit delay
    held_pos = np.roll(signals, delay_bars) # Fraction of capital 
    held_pos[:delay_bars] = 0.0

    trade = np.diff(held_pos, prepend=0.0)  # % of equity being traded

    price_ret = np.zeros_like(prices)
    price_ret[1:] = (prices[1:] - prices[:-1]) / prices[:-1]

    strategy_pnl, funding_pnl, fees = pnl_loop(
        held_pos, trade, price_ret, funding
    )

    equity = capital + np.cumsum(strategy_pnl)
    cum_ret = (equity / capital) -1 
    equity_lagged = np.roll(equity, 1)
    equity_lagged[0] = capital

    position_pnl = strategy_pnl - funding_pnl + fees # raw trade pnl
    trade_dollars = trade * (equity_lagged * leverage) # trade size in dollars

    returns_normalized = np.divide(
        strategy_pnl, equity_lagged,
        out=np.zeros_like(strategy_pnl),
        where=equity_lagged != 0
    ) # Puts returns into %

    assert np.isclose(equity[-1], 100_000 + strategy_pnl.sum())

    out = pd.DataFrame(
        {
            "asset_change (%)": price_ret,
            "signal (% of equity)": pos,
            "held_pos (% of equity)": held_pos,
            "trade (% of equity)": trade,
            "trade_dollars ($)": trade_dollars,
            "fees ($)": fees,
            "funding_pnl ($)": funding_pnl,
            "position_pnl ($)": position_pnl,
            "strategy_pnl ($)": strategy_pnl,
            "returns_normalized (%)": returns_normalized,
            "equity ($)": equity,
            "cum_ret": cum_ret
        },
        index=data_df.index
    )

    return out

 
@njit
def pnl_loop(
    held_pos,
    trade,
    price_ret,
    funding,
    capital=STARTING_CAPITAL,
    leverage=LEVERAGE,
    fee_rate=TAKER_FEE,
    delay_bars=DELAY_BARS,
):
    """
    Compute strategy PnL, with dynamic capital sizing.

    Parameters
    ----------
    held_pos : np.ndarray
        Position held at each bar (after signal execution and delay). - bar start

    trade : np.ndarray
        Trade size executed at each bar. 

    price_ret : np.ndarray
        Per-bar asset return series expressed in decimal form
        (e.g. 0.01 = +1%).

    funding : np.ndarray
        Per-bar funding rate applied to open positions
        (positive = paid, negative = received).

    capital : float, default STARTING_CAPITAL
        Initial capital used to compute dynamic position sizing
        and equity evolution.

    leverage : float, default LEVERAGE
        Gross leverage multiplier applied to position exposure.

    fee_rate : float, default TAKER_FEE
        Transaction cost per unit traded, expressed as a decimal.

    delay_bars : int, default DELAY_BARS
        Number of bars between signal generation and execution.

    Returns
    -------
    pnl : np.ndarray
        Per-bar strategy PnL in dollar terms.

    funding_pnl : np.ndarray
        Per-bar funding pnl in dollar terms.

    fees : np.ndarray
        Transaction and/or funding costs applied per bar.

    Notes
    -----
    - Uses a sequential loop (Numba JIT-compiled) because capital
      evolves over time and cannot be vectorised safely.
    - Assumes all input arrays are the same length.
    - Arrays should be float64 for optimal Numba performance.
    """

    n = len(held_pos)
    strategy_pnl  = np.zeros(n)
    equity_lagged = np.empty(n)
    funding_pnl   = np.zeros(n)
    fees          = np.zeros(n)
    equity_lagged[0] = capital
    current_equity   = capital

    start = max(1, delay_bars)
    for t in range(start, n):
        equity_lagged[t] = current_equity
        notional_t       = current_equity * leverage

        funding_pnl[t] = -held_pos[t] * funding[t] * notional_t
        fees[t]        = abs(trade[t]) * notional_t * fee_rate
        position_pnl_t = held_pos[t] * notional_t * price_ret[t]

        pnl_t           = position_pnl_t + funding_pnl[t] - fees[t]
        strategy_pnl[t] = pnl_t
        current_equity += pnl_t

    return strategy_pnl, funding_pnl, fees