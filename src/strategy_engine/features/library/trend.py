from ..base import Feature
import pandas as pd

class MA(Feature):

    def __init__(self, period: int):
        self.period = period


    @property
    def name(self) -> str:
        return f'ma_{self.period}'
    

    @property
    def window(self) -> int:
        return self.period
    

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df['mark_close'].rolling(self.period).mean()
    ### Needs to be generalised