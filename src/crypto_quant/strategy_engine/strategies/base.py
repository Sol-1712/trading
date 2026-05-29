from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from dataclasses import dataclass
from crypto_quant.data_utils.enums import PriceType



@dataclass(frozen=True)
class StrategyConfig:
    strategy_id:        str
    signal_price_types: tuple[PriceType, ...] = (PriceType.LAST,)


@dataclass(frozen=True)
class DataRequirements:
    price_types: tuple[PriceType, ...] = (PriceType.LAST,)
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
        price_types = self.config.signal_price_types
        pt = price_type or price_types[0]
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



