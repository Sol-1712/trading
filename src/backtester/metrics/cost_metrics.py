import numpy as np
from functools import cached_property
from backtester.utils import _compute_sharpe

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
    - Annualised turnover
    - Fee drag on sharpe
    - Funding drag on sharpe

    Attributes:
        core (CoreStats): Precomputed core statistics and returns from PnL.
    """


    def __init__(self, core):
        """
        Initializes CostMetrics with a CoreStats object.

        Args:
            core (CoreStats): Object containing primitive statistics and returns.
        """
        self.core             = core
        

    @cached_property
    def net_return(self) -> float:
        """
        Arithmetic sum of per-bar net returns.
        Used internally for cost ratio calculations only.
        NOT the true compounded return — see ReturnMetrics.net_return for that.
        """
        return float(np.sum(self.core.returns))
    

    @cached_property
    def gross_return(self) -> float:
        return float(np.sum(self.core.position_returns))


    @cached_property
    def sharpe(self) -> float:
        """Sharpe ratio of the strategy returns."""
        return _compute_sharpe(self.core.returns, self.core.rf)
    

    @cached_property
    def total_fee_drag(self) -> float:
        """Cumulative fee drag as sum of per-bar fee costs normalised by equity."""
        return float(np.sum(self.core.fee_returns))


    @cached_property
    def total_funding_drag(self) -> float:
        """
        Cumulative funding drag as sum of per-bar funding normalised by equity. 
        Signed — positive means funding helped.
        """
        return float(np.sum(self.core.funding_returns))


    @cached_property
    def total_cost_drag(self) -> float:
        """Total cost drag — fees plus negative funding."""
        return float(np.sum(self.core.fee_returns - self.core.funding_returns))


    @property
    def total_fee_pct_of_net(self) -> float:
        """Fees as fraction of absolute net return."""
        if self.net_return == 0:
            return np.nan
        return float(self.total_fee_drag / abs(self.net_return))


    @property
    def total_funding_pct_of_net(self) -> float:
        """Funding as fraction of absolute net return."""
        if self.net_return == 0:
            return np.nan
        return float(self.total_funding_drag / abs(self.net_return))


    @property
    def total_cost_pct_of_net(self) -> float:
        """Total costs as fraction of absolute net return."""
        if self.net_return == 0:
            return np.nan
        return float(self.total_cost_drag / abs(self.net_return))


    @property
    def cost_to_gross_ratio(self) -> float:
        """Total costs relative to gross return."""
        if self.gross_return == 0:
            return np.nan
        return float(self.total_cost_drag / abs(self.gross_return))


    @property
    def pct_bars_paying_funding(self) -> float:
        """Fraction of bars where funding was a cost (negative)."""
        return float(np.mean(self.core.funding_returns < 0))


    @property
    def avg_fee_per_bar(self) -> float:
        """Average fee per bar as fraction of equity."""
        return float(np.mean(self.core.fee_returns))


    @property
    def annualized_turnover(self) -> float:
        """
        Annualized turnover: sum of absolute trade fractions scaled to year."""
        return float(np.mean(np.abs(self.core.trade)) * self.core.ann_factor)
    

    @property
    def fee_drag_on_sharpe(self) -> float:
        """Reduction in Sharpe ratio caused by trading fees."""
        returns_no_fee = self.core.returns + self.core.fee_returns

        sharpe_no_fee = _compute_sharpe(returns_no_fee, self.core.rf)
        return float(sharpe_no_fee - self.sharpe)
    

    @property
    def funding_drag_on_sharpe(self) -> float:
        """Impact of funding payments on Sharpe ratio."""
        returns_no_funding = self.core.returns - self.core.funding_returns

        sharpe_no_funding = _compute_sharpe(returns_no_funding, self.core.rf)
        return float(sharpe_no_funding - self.sharpe)