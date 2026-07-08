from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from trading.data_utils.config import DataConfig
from trading.data_utils.enums import PriceType
from trading.backtester.risk.config import RiskConfig



if TYPE_CHECKING:
    from trading.backtester.fill.base import FillModel


@dataclass(frozen=True)
class RunConfig:
    """Experiment identity and metadata."""
    name:        str
    description: str              = ""
    tags:        tuple[str, ...]  = field(default_factory=tuple)


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution simulation parameters only. No risk constraints."""
    fee_rate:             float
    delay_bars:           int              = 1
    execution_price_type: PriceType        = PriceType.LAST
    fill_model:           FillModel | None = field(default=None, hash=False, compare=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.fee_rate <= 0.01:
            raise ValueError(f"fee_rate {self.fee_rate} out of range [0, 0.01]")
        if self.delay_bars < 1:
            raise ValueError(f"delay_bars must be >= 1, got {self.delay_bars}")
        
        object.__setattr__(
            self, "execution_price_type", PriceType(self.execution_price_type)
        )


@dataclass(frozen=True)
class BacktestConfig:
    """
    Composes all sub-configs for a backtest run.

    Downstream components receive only the sub-config they need:
        prepare_data()      → config.data
        ExecutionEngine     → config.execution
        RiskEngine          → config.risk
        Portfolio           → config.initial_capital
    """
    run:             RunConfig
    data:            DataConfig
    execution:       ExecutionConfig
    risk:            RiskConfig
    initial_capital: float

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {self.initial_capital}")