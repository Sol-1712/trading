import pandas as pd
import numpy as np
from functools import cached_property
from backtester.utils import _infer_ann_factor, _safe_divide

# Potentially pass an asset config file to set things like rf, mar, etc
# Or use the dataframe appendix thing
class CoreStats:

    def __init__(self, pnl_df):

        self.pnl_df = pnl_df

        # Convert df to raw numpy arrays
        self.returns     = pnl_df['returns_normalised'].to_numpy()
        self.equity      = pnl_df['equity ($)'].to_numpy()

        self.held_pos     = pnl_df['held_pos (% of equity)'].to_numpy()
        self.trade        = pnl_df['trade (% of equity)'].to_numpy()

        self.strategy_pnl = pnl_df["strategy_pnl ($)"].to_numpy() 
        self.position_pnl = pnl_df["position_pnl ($)"].to_numpy()
        self.funding_pnl  = pnl_df['funding_pnl ($)'].to_numpy()
        self.fee_pnl      = pnl_df["fees ($)"].to_numpy()

        # Basic stats
        self.log_returns = np.log1p(self.returns)
        
        self.n_bars = len(self.equity)                 # total time steps
        self.n_obs  = len(self.returns)                # number of return observations

        self.rf  = 0.0 # Risk Free per bar (funding_rate / bars_per_year LATER)
        self.mar = 0.0 # Minimum acceptable return per bar (TARGET RETURN)

        # Infer frequency and annualisation factor
        self.freq, self.ann_factor = _infer_ann_factor(self.pnl_df.index)
        self.ann_sqrt = np.sqrt(self.ann_factor)


    @cached_property
    def equity_lagged(self):
        eq = self.equity
        lag = np.roll(eq, 1)
        lag[0] = eq[0]
        return lag
    

    # Return Space
    @cached_property
    def position_returns(self):
        return _safe_divide(self.position_pnl, self.equity_lagged)


    @cached_property
    def fee_returns(self):
        return _safe_divide(self.fee_pnl, self.equity_lagged)


    @cached_property
    def funding_returns(self):
        return _safe_divide(self.funding_pnl, self.equity_lagged)
    

    # Drawdowns
    @cached_property
    def running_peak(self):
        return np.maximum.accumulate(self.equity)


    @cached_property
    def drawdown(self):
        gap = self.equity - self.running_peak
        return _safe_divide(gap, self.running_peak)