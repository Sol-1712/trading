from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import ClassVar

import numpy as np
import pandas as pd

from backtester.portfolio.snapshot import PortfolioSnapshot


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

    def __init__(self, 
                 initial_capital: float, 
                 fee_rate: float
                 ) -> None:
        
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")
        if not (0.0 <= fee_rate <= 0.01):
            raise ValueError(f"fee_rate {fee_rate} outside expected range [0, 0.01]")
        
        # Mutable state — modified only by step()
        self._equity:          float
        self._position_units:  float
        self._last_price:      float | None
        self._total_fees:      float
        self._total_funding:   float
        self._snapshots:       list[PortfolioSnapshot]

    # ------------------------------------------------------------------ #
    # Primary interface                                                     #
    # ------------------------------------------------------------------ #

    def step(
        self,
        timestamp:       datetime,
        fill_price:      float,
        target_fraction: float,
        funding_rate:    float = 0.0,
    ) -> PortfolioSnapshot:
        """
        Advance the portfolio by one bar.

        Order of operations (matters for correctness):
            1. MTM existing position at new price
            2. Apply funding on existing position
            4. Convert fraction → units using post-MTM equity
            5. Compute trade delta, apply fees
            6. Update state, record and return snapshot

        Parameters
        ----------
        timestamp : datetime
            Bar timestamp. Used as the index key in history().
        price : float
            Execution price for this bar. The engine is responsible for
            applying any delay_bars offset before calling this method.
        target_fraction : float
            Desired position as a signed fraction of current equity.
            +1.0 → fully long, -1.0 → fully short, 0.0 → flat.
            Values beyond ±1.0 imply leverage.
        funding_rate : float, optional
            Funding rate for this bar. Non-zero only on funding settlement bars.
            Positive rate → longs pay shorts (equity decreases if long).
            Negative rate → shorts pay longs (equity decreases if short).
            Default 0.0 (no funding event this bar).

        Returns
        -------
        PortfolioSnapshot
            Immutable record of state after this bar.
        """

        if fill_price <= 0.0:
            raise ValueError(f"price must be positive, got {fill_price} at {timestamp}")

        pass


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
    def total_fees(self) -> float:
        """Cumulative fees paid over all bars processed so far."""
        return self._total_fees
    
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
        Return the full per-bar history as a DataFrame.

        Index   : timestamp (datetime)
        Columns : price, target_fraction, position_units, equity,
                  bar_pnl, fee, trade_occurred

        Returns an empty DataFrame if no bars have been processed.
        """
        if not self._snapshots:
            return pd.DataFrame()

        rows = [dataclasses.asdict(s) for s in self._snapshots]
        df   = pd.DataFrame(rows).set_index("timestamp")
        df.index.name = "timestamp"
        return df

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
        self._total_fees     = 0.0
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
            f"total_fees={self._total_fees:,.2f}, "
            f"n_bars={self.n_bars}"
            f")"
        )