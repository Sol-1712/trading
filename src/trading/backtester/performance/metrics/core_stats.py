import logging
import pandas as pd
import numpy as np
from functools import cached_property
from typing import cast

from .utils import infer_ann_factor, safe_divide


logger = logging.getLogger(__name__)


class CoreStats:
    """
    Shared computation substrate for all metrics groups.

    Eager: raw column extraction + validation only.
    Lazy (cached_property): everything derived.

    Parameters
    ----------
    portfolio_history : pd.DataFrame
        Portfolio history.
    rf : float
        Annualised simple risk-free rate (e.g. 0.05 = 5%).
    """
    REQUIRED_COLS = (
        "equity", "position_units", "position_fraction",
        "position_pnl", "funding_pnl", "fees", "net_pnl",
        "leverage", "trade_occurred",
    )

    def __init__(self, portfolio_history: pd.DataFrame, rf: float = 0.0):
        self._validate(portfolio_history)

        df = portfolio_history.copy()      
        df.index = pd.to_datetime(df.index)

        self.rf = rf

        # --- raw columns only, eager ---
        self.equity             = df["equity"].to_numpy()
        self.position_units     = df["position_units"].to_numpy()
        self.position_fraction  = df["position_fraction"].to_numpy()
        self.leverage           = df["leverage"].to_numpy()
        self.position_pnl       = df["position_pnl"].to_numpy()
        self.funding_pnl        = df["funding_pnl"].to_numpy()
        self.fees               = df["fees"].to_numpy()
        self.net_pnl            = df["net_pnl"].to_numpy()
        self.trade_occurred     = df["trade_occurred"].to_numpy()

        if np.any(np.isnan(self.equity)):
            raise ValueError("Equity column contains NaN values")
        if np.any(self.equity < 0):
            logger.warning("Portfolio equity went negative (liquidation)")

        self.n_bars = len(self.equity)
        self.n_obs  = self.n_bars  # returns now defined for every bar (t=0 → 0), see below

        self.freq, self.ann_factor = infer_ann_factor(cast(pd.DatetimeIndex, df.index))
        self.ann_sqrt = np.sqrt(self.ann_factor)
        self.rf_bar = (1 + self.rf)**(1/self.ann_factor)-1

        logger.debug("CoreStats initialized: %d bars, freq=%s, ann_factor=%.0f",
                     self.n_bars, self.freq, self.ann_factor)
    

    def _validate(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            raise ValueError("portfolio_history cannot be None or empty")
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for col in self.REQUIRED_COLS:
            if df[col].isna().any():
                raise ValueError(
                    f"{col} contains NaNs"
                )

    # ------------------------------------------------------------------
    # Derived — lazy
    # ------------------------------------------------------------------

    @cached_property
    def equity_lagged(self) -> np.ndarray:
        lag = np.roll(self.equity, 1)
        lag[0] = self.equity[0]
        return lag

    @cached_property
    def returns(self) -> np.ndarray:
        """Single source of truth for per-bar return: derived from net_pnl,
        NOT equity.pct_change(). Guarantees agreement with the
        position/funding/fee return decomposition below by construction."""
        return safe_divide(self.net_pnl, self.equity_lagged)

    @cached_property
    def log_returns(self) -> np.ndarray:
        return np.log1p(self.returns)

    @cached_property
    def excess_returns(self) -> np.ndarray:
        return self.returns - self.rf_bar

    # --- Return decomposition (component-level, agrees with `returns` by construction) ---

    @cached_property
    def position_returns(self) -> np.ndarray:
        return safe_divide(self.position_pnl, self.equity_lagged)

    @cached_property
    def fee_returns(self) -> np.ndarray:
        return safe_divide(self.fees, self.equity_lagged)

    @cached_property
    def funding_returns(self) -> np.ndarray:
        return safe_divide(self.funding_pnl, self.equity_lagged)

    # --- Drawdown ---

    @cached_property
    def running_peak(self) -> np.ndarray:
        return np.maximum.accumulate(self.equity)

    @cached_property
    def drawdown(self) -> np.ndarray:
        gap = self.equity - self.running_peak
        return safe_divide(gap, self.running_peak)
