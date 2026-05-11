from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
from strategy_engine.features import Feature
from enum import Enum

class MAType(str, Enum):
    SMA = "sma"
    EMA = "ema"
    WMA = "wma"


class MovingAverage(Feature, ABC):

    def __init__(self, period: int, column: str = 'mark_close'):
        self.period = period
        self.column = column
    

    @property
    def window(self) -> int:
        return self.period
    

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if self.column not in data:
            raise ValueError(f"{self.column} not found in data")

        series = data[self.column]
        return self._compute(series)


    @abstractmethod
    def _compute(self, series: pd.Series) -> pd.Series:
        pass


class SMA(MovingAverage):

    @property
    def name(self) -> str:
        return f'sma_{self.period}'
    

    def _compute(self, series: pd.Series) -> pd.Series:
        return series.rolling(self.period).mean()


class EMA(MovingAverage):

    @property
    def name(self) -> str:
        return f'ema_{self.period}'
    

    def _compute(self, series: pd.Series) -> pd.Series:
        return series.ewm(span=self.period, adjust=False).mean()
