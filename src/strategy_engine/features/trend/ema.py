from pandas.core.api import Series as Series

from strategy_engine.features.trend import MovingAverage


class EMA(MovingAverage):


    @property
    def name(self) -> str:
        return f'ema_{self.period}'
    

    def _compute(self, series: Series) -> Series:
        return series.ewm(span=self.period, adjust=False).mean()
