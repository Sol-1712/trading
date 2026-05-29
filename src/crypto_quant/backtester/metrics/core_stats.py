import logging
import pandas as pd
import numpy as np
from functools import cached_property
from typing import cast
from crypto_quant.backtester.utils import infer_ann_factor, safe_divide

logger = logging.getLogger(__name__)

# Potentially pass an asset config file to set things like rf, mar, etc
# Or use the dataframe appendix thing
class CoreStats:
    """
    Computes core performance statistics from a PnL DataFrame.
    
    Extracts equity, returns, positions, and PnL components. Infers time
    frequency and annualization factor from index. Provides cached properties
    for return components (position, fee, funding returns).
    
    Parameters
    ----------
    pnl_df : pd.DataFrame
        Portfolio history from Portfolio.history() with columns:
        - equity: portfolio equity at bar end
        - position_units: signed position held
        - bar_pnl: MTM PnL from price movement
        - funding_pnl: funding payments
        - fee: fees paid
        - trade_occurred: bool flag if position changed
        Index must be datetime.
        
    Raises
    ------
    ValueError
        If pnl_df is None, empty, or missing required columns.
    """

    def __init__(self, pnl_df: pd.DataFrame):
        if pnl_df is None or pnl_df.empty:
            raise ValueError("pnl_df cannot be None or empty")

        required_cols = ['equity', 'position_units', 'bar_pnl', 'funding_pnl', 'fee']
        missing = [c for c in required_cols if c not in pnl_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.pnl_df = pnl_df
        df = pnl_df

        # --- core series ---
        self.equity = df["equity"].to_numpy()
        
        if np.any(np.isnan(self.equity)):
            raise ValueError("Equity column contains NaN values")
        if np.any(self.equity < 0):
            logger.warning("Portfolio equity went negative (liquidation)")

        # returns (derived safely)
        self.returns = (df["equity"].pct_change().fillna(0)).to_numpy()

        # position / exposure
        self.held_pos = df["position_units"].to_numpy()

        # trades
        self.trade = df.get("trade_occurred", pd.Series(False, index=df.index)).astype(float).to_numpy()

        # pnl components
        self.strategy_pnl = df["bar_pnl"].to_numpy()
        self.position_pnl = df["bar_pnl"].to_numpy()

        self.funding_pnl = df["funding_pnl"].to_numpy()
        self.fee_pnl = df["fee"].to_numpy()

        # Basic stats
        self.log_returns = np.log1p(self.returns)
        
        self.n_bars = len(self.equity)                 # total time steps
        self.n_obs  = len(self.returns)                # number of return observations

        self.rf  = 0.0 # Risk Free per bar (funding_rate / bars_per_year LATER)
        self.mar = 0.0 # Minimum acceptable return per bar (TARGET RETURN)

        pnl_df.index = pd.to_datetime(pnl_df.index)  # Ensure it actually is datetime
        dt_index = cast(pd.DatetimeIndex, pnl_df.index)

        # Infer frequency and annualisation factor
        self.freq, self.ann_factor = infer_ann_factor(dt_index)
        self.ann_sqrt = np.sqrt(self.ann_factor)
        
        logger.debug("CoreStats initialized: %d bars, freq=%s, ann_factor=%.0f",
                    self.n_bars, self.freq, self.ann_factor)


    @cached_property
    def equity_lagged(self):
        eq = self.equity
        lag = np.roll(eq, 1)
        lag[0] = eq[0]
        return lag
    

    # Return Space
    @cached_property
    def position_returns(self):
        return safe_divide(self.position_pnl, self.equity_lagged)


    @cached_property
    def fee_returns(self):
        return safe_divide(self.fee_pnl, self.equity_lagged)


    @cached_property
    def funding_returns(self):
        return safe_divide(self.funding_pnl, self.equity_lagged)
    

    # Drawdowns
    @cached_property
    def running_peak(self):
        return np.maximum.accumulate(self.equity)


    @cached_property
    def drawdown(self):
        gap = self.equity - self.running_peak
        return safe_divide(gap, self.running_peak)