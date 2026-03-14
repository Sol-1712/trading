import numpy as np
from functools import cached_property


class CostMetrics:
    """
    Computes cost-based metrics from a CoreStats object.

    This class provides key metrics:

    - Total fees
    - Total funding
    - Total costs
    - Total feels pct net
    - Total funding pct net
    - Total costs pct net
    - Cost to gross ratio
    - Pct bars paying funding
    - Avg fee per bar
    - Fee drag on sharpe
    - Funding drag on sharpe

    Attributes:
        core (CoreStats): Precomputed core statistics and returns from PnL.
        return_metrics (ReturnMetrics): Precomputed return metrics.
    """


    def __init__(self, core, return_metrics):
        """
        Initializes CostMetrics with a CoreStats object.

        Args:
            core (CoreStats): Object containing primitive statistics and returns.
            return_metrics (ReturnMetrics): Object containing calculated return metrics.
        """
        self.core           = core
        self.return_metrics = return_metrics
        pnl_df              = self.core.pnl_df

        self.equity_lagged  = pnl_df["equity_lagged ($)"].to_numpy()
        self.fees           = pnl_df["fees ($)"].to_numpy()
        self.funding_pnl    = pnl_df["funding_pnl ($)"].to_numpy()
        self.position_pnl   = pnl_df["position_pnl ($)"].to_numpy()
        self.strategy_pnl   = pnl_df["strategy_pnl ($)"].to_numpy()
        self.trade          = pnl_df["trade (% of equity)"].to_numpy()
  
        # Convert to return space
        self.fee_returns      = self.fees / self.equity_lagged
        self.funding_returns  = self.funding_pnl / self.equity_lagged
        self.position_returns = self.position_pnl / self.equity_lagged
        self.strategy_returns = self.core.returns



    @cached_property
    def total_fee_return(self) -> float:
        """Total fees paid as fraction of equity."""
        return float(np.sum(self.fee_returns))


    @cached_property
    def total_funding_return(self) -> float:
        """Total funding PnL as fraction of equity (signed)."""
        return float(np.sum(self.funding_returns))


    @cached_property
    def total_cost_return(self) -> float:
        """Total trading costs (fees + negative funding) as fraction of equity."""
        funding_cost = np.minimum(self.funding_returns, 0)  # only count funding I paid
        return float(np.sum(self.fee_returns + funding_cost))


    @property
    def total_fee_pct_of_net(self) -> float:
        """Fees as fraction of absolute net return."""
        if self.net_return == 0:
            return np.nan
        return float(self.total_fee_return / abs(self.net_return))


    @property
    def total_funding_pct_of_net(self) -> float:
        """Funding as fraction of absolute net return."""
        if self.net_return == 0:
            return np.nan
        return float(self.total_funding_return / abs(self.net_return))


    @property
    def total_cost_pct_of_net(self) -> float:
        """Total costs as fraction of absolute net return."""
        if self.net_return == 0:
            return np.nan
        return float(self.total_cost_return / abs(self.net_return))


    @property
    def cost_to_gross_ratio(self) -> float:
        """Total costs relative to gross return."""
        if self.gross_return == 0:
            return np.nan
        return float(self.total_cost_return / abs(self.gross_return))


    @property
    def pct_bars_paying_funding(self) -> float:
        """Fraction of bars where funding was a cost (negative)."""
        return float(np.mean(self.funding_returns < 0))


    @property
    def avg_fee_per_bar(self) -> float:
        """Average fee per bar as fraction of equity."""
        return float(np.mean(self.fee_returns))


    @property
    def annualized_turnover(self) -> float:
        """
        Annualized turnover: sum of absolute trade fractions scaled to year."""
        return float(np.mean(np.abs(self.trade)) * self.core.ann_factor)
    

    @property
    def fee_drag_on_sharpe(self) -> float:
        """Reduction in Sharpe ratio caused by trading fees."""
        returns_no_fee = self.strategy_returns + self.fee_returns

        sharpe_fee = self.return_metrics.sharpe
        sharpe_no_fee = self.return_metrics.compute_sharpe(returns_no_fee)
        return float(sharpe_no_fee - sharpe_fee)
    
    
    @property
    def funding_drag_on_sharpe(self) -> float:
        """Impact of funding payments on Sharpe ratio."""
        returns_no_funding = self.strategy_returns - self.funding_returns

        sharpe_funding = self.return_metrics.sharpe
        sharpe_no_funding = self.return_metrics.compute_sharpe(returns_no_funding)
        return float(sharpe_no_funding - sharpe_funding)