from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from dataclasses import dataclass, field

from trading.data_utils.core.enums import PriceType


@dataclass(frozen=True)
class StrategyConfig(ABC):
    price_type: PriceType = field(
        default=PriceType.MARK,
        kw_only=True
    )
    @property
    @abstractmethod
    def config_id(self) -> str:
        """Return a unique identifier for this configuration."""
        pass
    
    @property
    def name(self):
        return self.config_id

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
        return DataRequirements(price_type=self.config.price_type)

    def _resolve_column(self, base: str) -> str:
        """
        Resolve a prefixed column name.
        Always prefixed — no ambiguity regardless of how many price types are loaded.

        Parameters
        ----------
        base : str
            Base column name e.g. 'close', 'volume'
        """

        return f"{self.config.price_type.value}_{base}"

    # ------------------------------------------------------------------
    # Lifecycle hooks — optional
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Called once before the backtest/live session begins."""
        pass

    def on_stop(self) -> None:
        """Called once after the backtest/live session ends."""
        pass



