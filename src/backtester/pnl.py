import numpy as np
import pandas as pd

TAKER_FEE = 0.000550
LEVERAGE = 1.0
DELAY_BARS = 1 # 1-bar execution delay
# Config File


def pnl(
    data_df: pd.DataFrame,
    pos: pd.Series | np.ndarray,
    fee_rate: float = TAKER_FEE,
    price_col: str = "mark_close",
    delay_bars: int = 1,
    leverage: float = LEVERAGE,
    funding_col: str = "fundingRate",
) -> pd.DataFrame:
    """
    Compute perp strategy returns in 'return space' (equity normalized to 1).

    Assumptions:
    - price returns are close-to-close using price_col
    - pos[t] is the target position chosen at close(t) (fraction of equity notional)
    - execution delay is delay_bars, so held_pos[t] = pos[t-delay_bars]
    - fees are fee_rate * notional_traded, where notional_traded = leverage * abs(trade)
    - funding applies on held position: funding_pnl[t] = funding_sign * held_pos[t] * fundingRate[t]
    """

    if price_col not in data_df.columns:
        raise ValueError(f"Missing column: {price_col}")
    if funding_col not in data_df.columns:
        raise ValueError(f"Missing column: {funding_col}")
    if not data_df.index.is_monotonic_increasing:
        raise ValueError("data_df index must be sorted increasing (time order)")
    
    price = data_df["mark_close"].astype("float64").to_numpy()
    funding  = data_df["fundingRate"].astype("float64").to_numpy()

    ret = np.zeros_like(price)
    ret[1:] = price[1:] / price[:-1] - 1.0

    pos = np.asarray(pos, dtype="float64")
    if len(pos) != len(price):
        raise ValueError(f"pos length {len(pos)} != data length {len(price)}")
    
    # Held position with explicit delay
    held_pos = np.zeros_like(pos)
    if delay_bars < 0:
        raise ValueError("delay_bars must be >= 0")
    if delay_bars == 0:
        held_pos[:] = pos[:]
    else:
        held_pos[delay_bars:] = pos[:-delay_bars]
        held_pos[:delay_bars] = 0.0
    
    trade = pos - held_pos    

    fees = leverage * np.abs(trade) * fee_rate  

    funding_pnl = -held_pos * funding * leverage

    strategy_ret = leverage * held_pos * ret + funding_pnl - fees
    strategy_ret[:max(1, delay_bars)] = 0.0  # no return before you can hold

    equity = np.cumprod(1.0 + strategy_ret)

    out = pd.DataFrame(
        {
            "strategy_ret": strategy_ret,
            "equity": equity,
            "held_pos": held_pos,
            "trade": trade,
            "fees": fees,
            "funding_pnl": funding_pnl,
        },
        index=data_df.index,
    )

    return out

