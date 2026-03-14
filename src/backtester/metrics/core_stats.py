import pandas as pd
import numpy as np
from .utils import infer_ann_factor


class CoreStats:

    def __init__(self, pnl_df):

        self.pnl_df = pnl_df

        # Convert df to raw numpy arrays
        self.returns = pnl_df['returns_normalised'].to_numpy()
        self.equity_dollar = pnl_df['equity ($)'].to_numpy()

        self.trade = pnl_df['trade (% of equity)'].to_numpy()
        self.strategy_pnl = pnl_df["strategy_pnl ($)"].to_numpy()
        self.position_pnl = pnl_df["position_pnl ($)"].to_numpy()
        self.funding_pnl = pnl_df['funding_pnl ($)'].to_numpy()
        self.fees = pnl_df["fees ($)"].to_numpy()

        self.equity_dollar_lagged = np.roll(self.equity_dollar, 1)
        self.equity_dollar_lagged[0] = self.equity_dollar[0]

        # Return space
        self.position_returns = np.divide(
            self.position_pnl,
            self.equity_dollar_lagged,
            out=np.zeros_like(self.position_pnl),
            where=self.equity_dollar_lagged != 0
        )

        self.fee_returns = np.divide(
            self.fees,
            self.equity_dollar_lagged,
            out=np.zeros_like(self.fees),
            where=self.equity_dollar_lagged != 0
        )

        self.funding_returns = np.divide(
            self.funding_pnl,
            self.equity_dollar_lagged,
            out=np.zeros_like(self.funding_pnl),
            where=self.equity_dollar_lagged != 0
        )
        
        # Equity curve
        self.equity_curve = np.cumprod(1.0 + self.returns) 

        # Drawdowns
        self.running_peak = np.maximum.accumulate(self.equity)
        self.drawdown = np.divide(
            self.equity - self.running_peak,
            self.running_peak,
            out=np.zeros_like(self.equity),
            where=self.running_peak != 0
        )

        self.mdd = self.drawdown.min()

        # Basic stats
        self.equity_end = self.equity[-1]
        self.mean = float(np.mean(self.returns)) 
        self.sd = np.std(self.returns, ddof=1) # Sample SD (ddof)
        self.n_obs = len(self.returns)
        self.wins = self.returns[self.returns > 0]
        self.losses = self.returns[self.returns < 0]

        # Infer frequency and annualisation factor
        self.freq, self.ann_factor = infer_ann_factor(self.pnl_df.index)
        self.ann_sqrt = np.sqrt(self.ann_factor)
