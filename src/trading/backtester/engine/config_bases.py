from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from trading.data_utils.core.config import DataConfig
from trading.data_utils.core.enums import PriceType
from trading.backtester.risk.config import RiskConfig
from trading.backtester.fill import FILL_MODELS


if TYPE_CHECKING:
    from trading.backtester.fill.base import FillModel


@dataclass(frozen=True)
class RunConfig:
    """
    Experiment identity and metadata for a backtest run.

    Attributes
    ----------
    name : str
        Short human-readable run name.
    description : str, default ""
        Free-text description of the experiment.
    tags : tuple[str, ...], default ()
        Optional labels for filtering or grouping runs.
    """
    name:        str
    description: str              = ""
    tags:        tuple[str, ...]  = field(default_factory=tuple)


@dataclass(frozen=True)
class ExecutionConfig:
    """
    Execution simulation parameters (fees, delay, fill model, MTM).

    Does not include risk constraints — those live on RiskConfig.

    Attributes
    ----------
    fee_rate : float
        Proportional trading fee in ``[0, 1)``.
    delay_bars : int, default 1
        Bars between order submission and fill eligibility.
    fill_model : str, default "market"
        Key into FILL_MODELS selecting the fill simulation model.
    mtm_price_type : PriceType, default PriceType.MARK
        Price series used for mark-to-market valuation.
    """
    fee_rate:       float
    delay_bars:     int       = 1
    fill_model:     str       = "market"
    mtm_price_type: PriceType = PriceType.MARK

    def __post_init__(self) -> None:
        if not 0.0 <= self.fee_rate < 1.0:
            raise ValueError(f"fee_rate {self.fee_rate} out of range [0, 0.01]")
        if self.delay_bars < 1:
            raise ValueError(f"delay_bars must be >= 1, got {self.delay_bars}")
        
    @property
    def fill_model_cls(self) -> type[FillModel]:
        """
        Resolve the fill model class for ``fill_model``.

        Returns
        -------
        type[FillModel]
            Concrete FillModel subclass registered under ``fill_model``.

        Raises
        ------
        KeyError
            If ``fill_model`` is not present in FILL_MODELS.
        """
        return FILL_MODELS[self.fill_model]
        
    @property
    def price_type(self) -> PriceType:
        """
        Price series required by the selected fill model.

        The fill model is the authority on which price column is used
        for execution fills.

        Returns
        -------
        PriceType
            Price type declared by ``fill_model_cls``.
        """
        return self.fill_model_cls.price_type


@dataclass(frozen=True)
class BacktestConfig:
    """
    Composite configuration for a full backtest run.

    Downstream components receive only the sub-config they need:
        prepare_data()      → config.data
        ExecutionEngine     → config.execution
        RiskEngine          → config.risk
        Portfolio           → config.initial_capital

    Attributes
    ----------
    run : RunConfig
        Experiment identity and metadata.
    data : DataConfig
        Market data selection (symbol, interval, date range).
    execution : ExecutionConfig
        Execution simulation parameters.
    risk : RiskConfig
        Risk / position constraint parameters.
    initial_capital : float
        Starting portfolio cash; must be positive.
    """
    run:             RunConfig
    data:            DataConfig
    execution:       ExecutionConfig
    risk:            RiskConfig
    initial_capital: float

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {self.initial_capital}")