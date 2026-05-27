from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
from enum import Enum

from strategy_engine.features import Feature


class MovingAverage(Feature, ABC):

    def __init__(self, period: int, column: str) -> None:
        self._period = period
        self._column = column
    

    @property
    def window(self) -> int:
        return self._period
    

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if self._column not in data.columns:
            raise ValueError(
                f"{type(self).__name__} requires column '{self._column}'. "
                f"Available columns: {list(data.columns)}"
            )

        return self._compute(data[self._column])


    @abstractmethod
    def _compute(self, series: pd.Series) -> pd.Series:
        pass


class MAType(str, Enum):
    SMA = "sma"
    EMA = "ema"
    WMA = "wma"

    def build(self, period: int, column: str) -> MovingAverage:
        match self:
            case MAType.SMA: return SMA(period, column)
            case MAType.EMA: return EMA(period, column)
            case MAType.WMA: return WMA(period, column)
            case _: raise ValueError(f"Unhandled MAType: {self}")


class SMA(MovingAverage):

    @property
    def name(self) -> str:
        return f'sma_{self._period}_{self._column}'
    

    def _compute(self, series: pd.Series) -> pd.Series:
        return series.rolling(self._period).mean()
    
    



class EMA(MovingAverage):

    @property
    def name(self) -> str:
        return f'ema_{self._period}_{self._column}'
    

    def _compute(self, series: pd.Series) -> pd.Series:
        return series.ewm(span=self._period, adjust=False).mean()


class WMA(MovingAverage):

    @property
    def name(self) -> str:
        return f'wma_{self._period}_{self._column}'
    

    def _compute(self, series: pd.Series) -> pd.Series:
        weights = pd.Series(range(1, self._period + 1))
        return series.rolling(self._period).apply(
            lambda x: (x * weights).sum() / weights.sum(), raw=True
        )