from __future__ import annotations

import dataclasses
from datetime import datetime


@dataclasses.dataclass(slots=True)
class PortfolioSnapshot:
    """
    Immutable record of portfolio state at the end of a single bar.

    Produced by Portfolio.step() and accumulated into the history DataFrame.
    Using __slots__ reduces per-instance memory overhead, which matters
    when storing tens of thousands of snapshots across long backtests.

    Fields
    ------
    timestamp : datetime
        Bar timestamp (index key in the history DataFrame).
    price : float
        Execution price used for this bar (already delay-adjusted by the engine).
    target_fraction : float
        Target position as a signed fraction of equity passed in this step.
        +1.0 = fully long, -1.0 = fully short, 0.0 = flat.
    position_units : float
        Signed position held *after* any rebalancing on this bar.
    equity : float
        Portfolio equity at end of bar, after MTM PnL and fees.
    bar_pnl : float
        Mark-to-market PnL earned by the *previous* position over this bar's
        price move. Realised before any rebalancing occurs.
    fee : float
        Fee paid this bar, charged on the notional of any position change.
    trade_occurred : bool
        True if the position changed (|delta_units| > numerical tolerance).
    """

    timestamp:       datetime
    price:           float
    target_fraction: float
    position_units:  float
    equity:          float
    bar_pnl:         float
    fee:             float
    funding_pnl:     float
    trade_occurred:  bool