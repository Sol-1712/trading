from __future__ import annotations

import dataclasses
from datetime import datetime


@dataclasses.dataclass(slots=True)
class PortfolioSnapshot:
    """
    Immutable record of portfolio state at the end of a single bar.

    Produced by ``Portfolio.account_fills()`` and accumulated into the history
    DataFrame. ``slots=True`` reduces per-instance memory for long runs.

    Attributes
    ----------
    timestamp : datetime
        Bar timestamp (index key in the history DataFrame).
    price : float
        Mark-to-market price used for this bar (fraction basis).
    position_units : float
        Signed position held after fills on this bar.
    position_fraction : float
        Post-fill exposure ``(units * price) / equity``; 0.0 if equity ≤ 0.
        +1.0 ≈ fully long, -1.0 ≈ fully short.
    equity : float
        Portfolio equity at end of bar after MTM, funding, and fees.
    position_pnl : float
        Mark-to-market PnL on the position held into this bar
        (close-to-close price move), before fills.
    funding_pnl : float
        Funding cashflow this bar. Negative = paid, positive = received.
    fees : float
        Total fees paid this bar across fills.
    net_pnl : float
        ``position_pnl + funding_pnl - fees``.
    fill_occurred : bool
        True if at least one fill was applied this bar.
    """

    timestamp:         datetime
    price:             float     # bar close — used for fraction calculation
    position_units:    float     # post-fill
    position_fraction: float     # post-fill: (units × price) / equity
    equity:            float     # post-fill
    position_pnl:      float     # Position PnL from price movement
    funding_pnl:       float     # funding payment (negative = paid, positive = received)
    fees:              float     # total fees paid this bar
    net_pnl:           float     # position_pnl + funding_pnl - fee
    fill_occurred:     bool
