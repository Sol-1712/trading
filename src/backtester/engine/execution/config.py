from dataclasses import dataclass
from data_utils.enums import PriceType
from backtester.engine.execution.fill import FillModel


@dataclass(frozen=True)
class ExecutionConfig:
    """
    Attributes
    ----------
    fee_rate : float
        Proportional fee rate per trade (e.g. 0.001 for 0.1% fee).
    delay_bars : int
        Number of bars to delay fill orders.
    execution_price_type : PriceType
        Type of price to use for execution.
    leverage_max : float
        Maximum leverage allowed.
    fill_model : FillModel | None
        Optional custom fill model. If None, defaults to MarketFillModel.
    """
    fee_rate:             float            = 0.000550
    delay_bars:           int              = 1
    execution_price_type: PriceType        = PriceType.LAST
    leverage_max:         float            = 100.0
    fill_model:           FillModel | None = None # Defaults to MarketFillModel


    def __post_init__(self) -> None:
        if not 0.0 <= self.fee_rate <= 0.01:
            raise ValueError(f"fee_rate {self.fee_rate} out of range [0, 0.01]")
        if self.delay_bars < 1:
            raise ValueError(f"delay_bars must be >= 1, got {self.delay_bars}")
        if self.leverage_max < 1.0:
            raise ValueError(f"leverage_max must be >= 1.0, got {self.leverage_max}")


        # Coerce raw string from YAML → PriceType enum
        object.__setattr__(
            self, "execution_price_type", PriceType(self.execution_price_type)
        )