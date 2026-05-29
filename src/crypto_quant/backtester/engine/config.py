from dataclasses import dataclass
from crypto_quant.data_utils.enums import PriceType
from crypto_quant.data_utils.config import DataConfig
from crypto_quant.backtester.engine.execution import ExecutionConfig


@dataclass(frozen=True)
class BacktestConfig:
    """
    Attributes
    ----------
        data : DataConfig
            Symbol, interval, and date range for the backtest.
        execution : ExecutionConfig
            Execution assumptions like fees and slippage.
        initial_capital : float
            Starting capital for the backtest.
    """
    data:            DataConfig
    execution:       ExecutionConfig
    initial_capital: float

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError(
                f"initial_capital must be positive, got {self.initial_capital}"
            )