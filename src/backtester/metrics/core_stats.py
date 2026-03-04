import pandas as pd
import numpy as np
from .utils import infer_ann_factor


class CoreStats:

    def __init__(self, pnl_df):

        self.pnl_df = pnl_df

        # Infer frequency and annualisation factor
        self.freq, self.ann_factor = infer_ann_factor(self.pnl_df.index)
        self.ann_sqrt = np.sqrt(self.ann_factor)

        # Returns
        self.returns = pnl_df['returns_normalised'].to_numpy()
        self.n_obs = len(self.returns)

        # Equity
        self.equity = np.cumprod(1.0 + self.returns) # (1+r), [0] = 1.0

        # Drawdowns
        self.running_peak = np.maximum.accumulate(self.equity)
        self.drawdown = (self.equity - self.running_peak) / self.running_peak
        self.mdd = self.drawdown.min()

        # Basic stats
        self.equity_end = self.equity[-1]
        self.mean = float(np.mean(self.returns)) 
        self.sd = np.std(self.returns, ddof=0) # Population SD (ddof)

