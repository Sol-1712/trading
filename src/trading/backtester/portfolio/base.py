from __future__ import annotations

import dataclasses
import logging
from typing import ClassVar, cast
import numpy as np
import pandas as pd
import math

from trading.backtester.portfolio.snapshot import PortfolioSnapshot
from trading.backtester.fill import Fill


logger = logging.getLogger(__name__)


class Portfolio:
    """
    Stateful bar-by-bar portfolio simulator.

    Receives pre-sized target fractions from the execution engine
    and handles all accounting: MTM, funding, fees, equity tracking.

    Deliberately decoupled from ExecutionConfig — takes only the 
    scalar values it needs so it can be used or tested independently.
    Parameters
    ----------
    initial_capital : float
        Starting equity in quote currency (e.g. USDT).
    fee_rate : float
        Proportional fee applied to trade notional (e.g. 0.0005 = 0.05%).

    Example
    -------
    >>> portfolio = Portfolio(initial_capital=10_000.0, fee_rate=0.0005)
    >>> snap = portfolio.step(timestamp=ts, price=50_000.0, target_fraction=0.5)
    >>> df = portfolio.history()
    """

    _UNITS_TOLERANCE: ClassVar[float] = 1e-9  # below which delta is treated as zero

    def __init__(self, initial_capital: float, fee_rate: float) -> None:
        """
        Initialize portfolio simulator.
        
        Parameters
        ----------
        initial_capital : float
            Starting equity in quote currency (e.g., USDT).
            Must be positive.
        fee_rate : float
            Proportional fee as decimal (e.g., 0.0005 = 0.05%).
            Checked to be in range [0.0, 0.01].
            
        Raises
        ------
        ValueError
            If initial_capital <= 0 or fee_rate outside [0, 0.01].
        """
        
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")
        if not (0.0 <= fee_rate <= 0.01):
            raise ValueError(f"fee_rate {fee_rate} outside expected range [0, 0.01]")
        
        self._initial_capital: float                   = initial_capital
        self._equity:          float                   = initial_capital
        self._position_units:  float                   = 0.0
        self._last_price:      float | None            = None
        self._fee_rate:        float                   = fee_rate
        self._snapshots:       list[PortfolioSnapshot] = []
        
        logger = logging.getLogger(__name__)
        logger.debug("Portfolio initialized: capital=%.2f, fee_rate=%.5f", 
                    initial_capital, fee_rate)

    # ------------------------------------------------------------------ #
    # Primary interface                                                     #
    # ------------------------------------------------------------------ #

    def step(
        self,
        timestamp:       pd.Timestamp,
        fills:           list[Fill],
        mtm_price:       float,
        funding_rate:    float
    ) -> PortfolioSnapshot:
        """
        Advance portfolio by one bar.
        
        ORDER OF OPERATIONS:
        1. MTM existing position using mark_close (bar's close price)
        2. Apply funding based on position held at start of bar
        3. Execute fills at execution prices (from fill model)
        4. Record snapshot at mark_close (for position fraction calc)
        
        PRICE BASIS:
        - bar_pnl: uses mark_close (MTM)
        - position_fraction: uses mark_close (MTM for leverage calc)
        - fills: use their own fill_price (from execution model)
        
        This separation is intentional:
        - MTM uses mark (index price, more accurate)
        - Execution uses last (market price, realistic)

        Parameters
        ----------
        timestamp : pd.Timestamp
            Timestamp of the current bar.
        fills : list[Fill]
            Fills from execute_pending. Empty list if no fills this bar.
        mtm_price : float
            Mark-to-market price for the current bar.
        funding_rate : float
            Funding rate for the current bar. Positive means long pays short.

        Returns
        -------
        PortfolioSnapshot
            Immutable record of state after this bar.
        """

            
        # ── Validate inputs ──────────────────────────────────────────────
        if not (0 < mtm_price < 1e10):
            raise ValueError(f"Invalid mtm_price {mtm_price} at {timestamp}")
        if not math.isfinite(funding_rate):
            raise ValueError(f"Non-finite funding_rate {funding_rate} at {timestamp}")
    
        prev_price = self._last_price if self._last_price is not None else mtm_price

        # ── 1. MTM existing position ─────────────────────────────────────
        # Position held during this bar earns close-to-close price change.
        bar_pnl      = self._position_units * (mtm_price - prev_price)
        self._equity += bar_pnl

        # ── 2. Funding settlement ────────────────────────────────────────
        # Applied on position held at start of bar, at prev bar's close price.
        # Negative funding_pnl = equity decreases -> I paid.

        funding_pnl   = -(self._position_units * prev_price * funding_rate)
        self._equity += funding_pnl

        # ── 3. Execute fills ─────────────────────────────────────────────
        remaining_equity = self._equity

        total_fee = 0.0
        for fill in fills:
            fee = abs(fill.units_filled) * fill.fill_price * self._fee_rate
            if not (0 < fill.fill_price < 1e10):
                raise ValueError(f"Invalid fill_price {fill.fill_price} at {timestamp}")
            
            if fee > remaining_equity:
                logger.warning(
                    "Fee %.2f exceeds remaining equity %.2f at %s — ruin.",
                    fee, remaining_equity, timestamp,
        )
            
            self._equity -= fee
            remaining_equity -= fee
            total_fee += fee
            self._position_units += fill.units_filled

        # ── 4.Verify state ────────────────────────────────────
        if math.isnan(self._position_units) or math.isnan(self._equity):
            raise RuntimeError(
                f"Portfolio state became NaN at {timestamp}. "
                f"position_units={self._position_units}, equity={self._equity}"
    )

        # ── 5. Update price reference ────────────────────────────────────
        self._last_price = mtm_price

        # ── 6. Derived values ────────────────────────────────────────────
        position_fraction = (
            (self._position_units * mtm_price) / self._equity
            if self._equity > 0 else 0.0
        )
        snapshot = PortfolioSnapshot(
            timestamp         = timestamp,
            price             = mtm_price,
            position_units    = self._position_units,
            position_fraction = position_fraction,
            equity            = self._equity,
            bar_pnl           = bar_pnl,
            funding_pnl       = funding_pnl,
            fees               = total_fee,
            net_pnl           = bar_pnl + funding_pnl - total_fee,
            leverage          = abs(position_fraction),
            trade_occurred    = len(fills) > 0,
        )

        self._snapshots.append(snapshot)
        return snapshot


    # ------------------------------------------------------------------ #
    # State accessors                                                      
    # ------------------------------------------------------------------ #

    @property
    def equity(self) -> float:
        """Current portfolio equity (quote currency)."""
        return self._equity

    @property
    def position_units(self) -> float:
        """Current signed position in base asset units."""
        return self._position_units
   

    @property
    def initial_capital(self) -> float:
        return self._initial_capital

    @property
    def n_bars(self) -> int:
        """Number of bars processed."""
        return len(self._snapshots)

    def is_flat(self) -> bool:
        """True if the portfolio currently holds no position."""
        return abs(self._position_units) <= self._UNITS_TOLERANCE

    # ------------------------------------------------------------------ #
    # Output                                                                #
    # ------------------------------------------------------------------ #

    def history(self) -> pd.DataFrame:
        """
        Convert accumulated snapshots to a DataFrame indexed by timestamp.
        
        Called once after all bars are processed. Suitable for metrics computation
        and performance analysis.
        
        Returns
        -------
        pd.DataFrame
            Portfolio state history with columns from PortfolioSnapshot fields
            and timestamp as index. Empty DataFrame if no snapshots recorded.
        """
        if not self._snapshots:
            logger.warning("Portfolio has no snapshot history")
            return pd.DataFrame()

        result = pd.DataFrame(
            [dataclasses.asdict(s) for s in self._snapshots]
        ).set_index("timestamp")
        
        logger.debug("Exported portfolio history: %d rows", len(result))
        return result

    # ------------------------------------------------------------------ #
    # Lifecycle                                                             #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """
        Reset portfolio to its initial state.
        
        Clears all snapshots and position state. Useful for parameter sweeps or
        walk-forward loops where the same Portfolio object is reused.
        Maintains original capital and fee configuration.
        """
        self._equity         = self.initial_capital
        self._position_units = 0.0
        self._last_price     = None
        self._snapshots      = []
        logger.debug("Portfolio reset to initial state: capital=%.2f", 
                    self._initial_capital)

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _fraction_to_units(self, fraction: float, price: float) -> float:
        """
        Convert a signed position fraction to signed asset units.

        Units = sign(fraction) x |equity x fraction| / price

        If equity is zero or negative (portfolio is ruined), returns 0
        rather than attempting further positioning.

        Parameters
        ----------
        fraction : float
            Signed target fraction of equity. ±1.0 = ±100% notional.
        price : float
            Current asset price (quote currency per unit).
        """
        if self._equity <= 0.0:
            return 0.0  # Ruined
        notional = self._equity * abs(fraction)
        return float(np.sign(fraction)) * notional / price


    def __repr__(self) -> str:
        return (
            f"Portfolio("
            f"equity={self._equity:,.2f}, "
            f"position_units={self._position_units:.6f}, "
            f"n_bars={self.n_bars}"
            f")"
        )