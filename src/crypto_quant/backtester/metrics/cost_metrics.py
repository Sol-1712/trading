import logging
import numpy as np
from functools import cached_property
from crypto_quant.backtester.utils import compute_sharpe

logger = logging.getLogger(__name__)

class CostMetrics:
    """
    Computes cost-based metrics from a CoreStats object.
    
    Quantifies impact of trading costs (fees and funding) on returns and Sharpe ratio.
    Provides per-bar costs, cumulative drag, and cost ratios for performance analysis.
    
    Parameters
    ----------
    core : CoreStats
        Precomputed core statistics with fee_pnl, funding_pnl, and returns.
        
    Raises
    ------
    ValueError
        If core is None or missing required attributes.
    """

    def __init__(self, core):
        if core is None:
            raise ValueError("core cannot be None")
        if not hasattr(core, 'fee_pnl') or core.fee_pnl is None:
            raise ValueError("core must have 'fee_pnl' attribute")
        if not hasattr(core, 'funding_pnl') or core.funding_pnl is None:
            raise ValueError("core must have 'funding_pnl' attribute")
            
        self.core = core
        logger.debug("CostMetrics initialized with %d observations", core.n_obs)
        

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
        return compute_sharpe(self.core.returns, self.core.rf, self.core.ann_factor)
    

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

        sharpe_no_fee = compute_sharpe(returns_no_fee, self.core.rf, self.core.ann_factor)
        return float(sharpe_no_fee - self.sharpe)
    

    @property
    def funding_drag_on_sharpe(self) -> float:
        """Impact of funding payments on Sharpe ratio."""
        returns_no_funding = self.core.returns - self.core.funding_returns

        sharpe_no_funding = compute_sharpe(returns_no_funding, self.core.rf, self.core.ann_factor)
        return float(sharpe_no_funding - self.sharpe)