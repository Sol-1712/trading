from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import ClassVar, cast

import numpy as np
import pandas as pd

from backtester.portfolio.snapshot import PortfolioSnapshot
from backtester.engine.execution.fill import Fill


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
        
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")
        if not (0.0 <= fee_rate <= 0.01):
            raise ValueError(f"fee_rate {fee_rate} outside expected range [0, 0.01]")
        

        self._equity:          float                   = initial_capital
        self._position_units:  float                   = 0.0
        self._last_price:      float | None            = None
        self._fee_rate:        float                   = fee_rate
        self._snapshots:       list[PortfolioSnapshot] = []

    # ------------------------------------------------------------------ #
    # Primary interface                                                     #
    # ------------------------------------------------------------------ #

    def step(
        self,
        fills:           list[Fill],
        bar:             pd.Series
    ) -> PortfolioSnapshot:
        """
        Advance portfolio by one bar.

        Order of operations
        -------------------
        1. MTM existing position close-to-close
        2. Apply funding on existing position
        3. Execute fills — update position, charge fees
        4. Record snapshot at bar close

        Parameters
        ----------
        fills : list[Fill]
            Fills from execute_pending. Empty list if no fills this bar.
        bar: pd.Series
            Current bar.

        Returns
        -------
        PortfolioSnapshot
            Immutable record of state after this bar.
        """

        mtm_price = bar['mark_close']
        funding_rate = bar['funding_rate']
        timestamp = cast(pd.Timestamp, bar.name).to_pydatetime()

        prev_price = self._last_price if self._last_price is not None else mtm_price

        # ── 1. MTM existing position ─────────────────────────────────────
        # Position held during this bar earns close-to-close price change.
        bar_pnl      = self._position_units * (mtm_price - prev_price)
        self._equity += bar_pnl

        # ── 2. Funding settlement ────────────────────────────────────────
        # Applied on position held at start of bar, at prev bar's close price.
        # Negative funding_pnl = equity decreases (you paid).
        funding_pnl   = -(self._position_units * prev_price * funding_rate)
        self._equity += funding_pnl

        # ── 3. Execute fills ─────────────────────────────────────────────
        total_fee = 0.0
        for fill in fills:
            fee            = abs(fill.units_filled) * fill.fill_price * self._fee_rate
            self._equity  -= fee
            total_fee      += fee
            self._position_units += fill.units_filled

        # ── 4. Update price reference ────────────────────────────────────
        self._last_price = mtm_price

        # ── 5. Derived values ────────────────────────────────────────────
        if self._equity > 0:
            position_fraction = (self._position_units * mtm_price) / self._equity
        else:
            position_fraction = 0.0

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
    def total_funding(self) -> float:
        """
        Cumulative funding PnL over all bars.
        Negative means net payer; positive means net receiver.
        """
        return self._total_funding

    @property
    def initial_capital(self) -> float:
        return self.initial_capital

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
        Convert snapshots to a DataFrame indexed by timestamp.
        Called once after all bars are processed.
        """
        if not self._snapshots:
            return pd.DataFrame()

        return pd.DataFrame(
            [dataclasses.asdict(s) for s in self._snapshots]
        ).set_index("timestamp")

    # ------------------------------------------------------------------ #
    # Lifecycle                                                             #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """
        Reset portfolio to its initial state.

        Useful for parameter sweeps or walk-forward loops where you want
        to reuse the same Portfolio object across multiple runs without
        re-instantiating it.
        """
        self._equity         = self.initial_capital
        self._position_units = 0.0
        self._last_price     = None
        self._total_funding  = 0.0
        self._snapshots      = []

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