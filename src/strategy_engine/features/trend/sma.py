from pandas.core.api import Series as Series
from strategy_engine.features.trend import MovingAverage




class SMA(MovingAverage):

    @property
    def name(self) -> str:
        return f'sma_{self.period}'
    

    def _compute(self, series: Series) -> Series:
        return series.rolling(self.period).mean()
