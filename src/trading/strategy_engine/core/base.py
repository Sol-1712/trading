from __future__ import annotations

from abc import ABC
from typing import Generic, TypeVar
from dataclasses import dataclass
from trading.data_utils.core.enums import PriceType


@dataclass(frozen=True)
class StrategyConfig:
    strategy_id:    str
    price_type:     PriceType = PriceType.MARK


@dataclass(frozen=True)
class DataRequirements:
    """
    Declares what market data a strategy needs.
    Constructed by data_requirements() and consumed by the runner.
    """
    price_type:  PriceType
    columns:     tuple[str, ...] | None = None
    

ConfigT = TypeVar("ConfigT", bound = StrategyConfig)


class StrategyBase(ABC, Generic[ConfigT]):

    def __init__(self, config: ConfigT) -> None:
        self.config = config
        
    # ------------------------------------------------------------------
    # Data interface — called by runner before loading
    # ------------------------------------------------------------------

    def data_requirements(self) -> DataRequirements:
        """
        Derives data requirements from strategy config.
        Override if a strategy needs additional columns or 
        more complex requirements.
        """
        return DataRequirements(price_types=self.config.signal_price_types)

    def _resolve_column(self, base: str, price_type: PriceType | None = None) -> str:
        """
        Resolve a prefixed column name.
        Always prefixed — no ambiguity regardless of how many price types are loaded.

        Parameters
        ----------
        base : str
            Base column name e.g. 'close', 'volume'
        price_type : PriceType, optional
            Which price type prefix to use.
            Defaults to first declared signal price type.
        """

        pt = price_type or self.config.signal_price_types[0]
        return f"{pt.value}_{base}"

    # ------------------------------------------------------------------
    # Lifecycle hooks — optional
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Called once before the backtest/live session begins."""
        pass

    def on_stop(self) -> None:
        """Called once after the backtest/live session ends."""
        pass



