from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass, asdict
import logging
from typing import ClassVar
import numpy as np
import pandas as pd
import math

from .snapshot import PortfolioSnapshot
from .trade import TradeLog
from trading.backtester.fill import Fill


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _BarDraft:
    timestamp:     datetime
    position_pnl:  float = 0.0
    funding_pnl:   float = 0.0
    fees:          float = 0.0
    fill_occurred: bool = False


class Portfolio:

    """
    Stateful bar-by-bar portfolio simulator.

    Applies MTM, funding, and fill accounting each bar. Decoupled from
    ExecutionConfig — takes only the scalar inputs it needs so it can be
    used or tested independently.
    """

    _UNITS_TOLERANCE:  ClassVar[float] = 1e-9  # below which delta is treated as zero
    _EQUITY_TOLERANCE: ClassVar[float] = 1e-9  # below which equity is treated as zero

    def __init__(self, initial_capital: float) -> None:
        """
        Initialize the portfolio simulator.

        Parameters
        ----------
        initial_capital : float
            Starting equity in quote currency (e.g. USDT). Must be positive.

        Raises
        ------
        ValueError
            If ``initial_capital`` <= 0.
        """
        
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")

        self.trade_log:        TradeLog                = TradeLog()
        self._initial_capital: float                   = initial_capital
        self._equity:          float                   = initial_capital
        self._position_units:  float                   = 0.0
        self._last_price:      float | None            = None
        self._bar:             _BarDraft | None        = None
        self._snapshots:       list[PortfolioSnapshot] = []
        

        logger.debug("Portfolio initialized: capital=%.2f", initial_capital)

    # ------------------------------------------------------------------ #
    # Primary interface                                                   
    # ------------------------------------------------------------------ #

    def accrue_bar(
        self, 
        timestamp: datetime, 
        mtm_price: float, 
        funding_rate: float
        ) -> None:
        """
        Accrue the portfolio state by one bar.
        This function should be called once per bar.

        Parameters
        ----------
        timestamp : datetime
            Timestamp of the current bar.
        mtm_price : float
            Mark-to-market price for this bar.
        funding_rate : float    
            Funding rate for this bar.

        Raises
        ------
        ValueError
            If ``mtm_price`` is out of range or ``funding_rate`` is non-finite.
        RuntimeError
            If accrue_bar called before commit_snapshot.
        """

        if self._bar is not None:
            raise RuntimeError("accrue_bar called before commit_snapshot")

        # ── Validate inputs ──────────────────────────────────────────────
        if not (0 < mtm_price < 1e10):
            raise ValueError(f"Invalid mtm_price {mtm_price} at {timestamp}")
        if not math.isfinite(funding_rate):
            raise ValueError(f"Non-finite funding_rate {funding_rate} at {timestamp}")
    
        prev_price = self._last_price if self._last_price is not None else mtm_price

        # ── 1. MTM existing position ─────────────────────────────────────
        # Position held during this bar earns close-to-close price change.
        position_pnl  = self._position_units * (mtm_price - prev_price)
        self._equity += position_pnl        

        # ── 2. Funding settlement ────────────────────────────────────────
        # Applied on position held at start of bar, at mtm price.
        # Negative funding_pnl = equity decreases -> I paid.

        funding_pnl   = -(self._position_units * mtm_price * funding_rate)
        self._equity += funding_pnl
        self.trade_log.accrue_bar(funding_pnl) 

        # ── 3. Update price reference ────────────────────────────────────
        self._last_price = mtm_price

        self._bar = _BarDraft(
            timestamp=timestamp,
            position_pnl=position_pnl,
            funding_pnl=funding_pnl,
        )

        
    def apply_fills(self, fills: list[Fill]) -> None:
        """
        Apply fills (fees + unit changes) from the execution engine.
        Fills use each fill's own ``fill_price``. May be called multiple
        times per bar; empty list is a no-op (does not clear fill_occurred).

        Parameters
        ----------
        fills : list[Fill]
            Fills from ``execute_pending``; empty if none this call.

        Raises
        ------
        RuntimeError
            If called before accrue_bar / after commit_snapshot, units
            disagree with TradeLog, or state is non-finite.
        """
        if self._bar is None:
            raise RuntimeError("apply_fills before accrue_bar")

        if not fills:
            return

        for fill in fills:
            fees = fill.fees
            self._equity -= fees
            self._bar.fees += fees
            self._position_units += fill.units_filled
            self.trade_log.on_fill(fill, self._bar.timestamp)
        self._bar.fill_occurred = True
        self._reconcile()

    def commit_snapshot(self) -> PortfolioSnapshot:
        if self._bar is None:
            raise RuntimeError("commit_snapshot with no open bar")
        if self._last_price is None:
            raise RuntimeError("commit_snapshot with no mtm price")

        self._reconcile()

        bar = self._bar
        price = self._last_price
        position_fraction = (
            (self._position_units * price) / self._equity
            if self._equity > 0 else 0.0
        )
        snapshot = PortfolioSnapshot(
            timestamp         = bar.timestamp,
            price             = price,
            position_units    = self._position_units,
            position_fraction = position_fraction,
            equity            = self._equity,
            position_pnl      = bar.position_pnl,
            funding_pnl       = bar.funding_pnl,
            fees              = bar.fees,
            net_pnl           = bar.position_pnl + bar.funding_pnl - bar.fees,
            fill_occurred     = bar.fill_occurred,
        )

        self._snapshots.append(snapshot)
        self._bar = None
        return snapshot


    # ------------------------------------------------------------------ #
    # State accessors                                                    # 
    # ------------------------------------------------------------------ #

    @property
    def equity(self) -> float:
        """Current portfolio equity in quote currency."""
        return self._equity


    @property
    def position_units(self) -> float:
        """Current signed position in base-asset units."""
        return self._position_units
   

    @property
    def initial_capital(self) -> float:
        """Starting equity set at construction."""
        return self._initial_capital


    @property
    def n_bars(self) -> int:
        """Number of bars processed (snapshots recorded)."""
        return len(self._snapshots)


    def is_flat(self) -> bool:
        """
        Return True if the portfolio holds no meaningful position (within tolerance).
        """
        return abs(self._position_units) <= self._UNITS_TOLERANCE


    def is_ruined(self) -> bool:
        """
        Return True if the portfolio equity is below the tolerance (ruined).
        """
        return self._equity <= self._EQUITY_TOLERANCE

    # ------------------------------------------------------------------ #
    # Output                                                             #
    # ------------------------------------------------------------------ #

    def history(self) -> pd.DataFrame:
        """
        Convert accumulated snapshots to a DataFrame indexed by timestamp.

        Intended to be called once after the run for metrics and reporting.

        Returns
        -------
        pd.DataFrame
            One row per bar with PortfolioSnapshot fields; empty if no
            snapshots have been recorded.
        """
        if not self._snapshots:
            logger.warning("Portfolio has no snapshot history")
            return pd.DataFrame()

        result = pd.DataFrame(
            [asdict(s) for s in self._snapshots]
        ).set_index("timestamp")
        
        logger.debug("Exported portfolio history: %d rows", len(result))
        return result

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """
        Reset portfolio to its initial cash / flat state.

        Clears snapshots, position, and last price. Reuses the original
        ``initial_capital``. Useful for parameter sweeps or walk-forward loops
        sweeps or walk-forward loops that reuse the same instance.
        """
        self._equity         = self.initial_capital
        self._position_units = 0.0
        self._last_price     = None
        self._snapshots      = []
        self.trade_log       = TradeLog()
        self._bar            = None
        logger.debug("Portfolio reset to initial state: capital=%.2f", 
                    self._initial_capital)

    # ------------------------------------------------------------------ #
    # Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _reconcile(self) -> None:
        """Assert book units match TradeLog and equity/units are finite."""
        expected_units = (
            self.trade_log.open_trade.units if self.trade_log.open_trade else 0.0
        )
        if not math.isclose(
            self._position_units, expected_units, abs_tol=self._UNITS_TOLERANCE
        ):
            raise RuntimeError(
                f"Position/TradeLog mismatch: "
                f"portfolio={self._position_units}, trade_log={expected_units}"
            )
        if not math.isfinite(self._position_units) or not math.isfinite(self._equity):
            raise RuntimeError(
                f"Portfolio state non-finite: "
                f"position_units={self._position_units}, equity={self._equity}"
            )

    def _fraction_to_units(self, fraction: float, price: float) -> float:
        """
        Convert a signed position fraction of equity into asset units.

        ``units = sign(fraction) * |equity * fraction| / price``.
        Returns 0 if equity is zero or negative (ruined).

        Parameters
        ----------
        fraction : float
            Signed target fraction of equity (±1.0 = ±100% notional).
        price : float
            Asset price in quote currency per unit.

        Returns
        -------
        float
            Signed base-asset units, or 0.0 if ruined.
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