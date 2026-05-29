from abc import ABC, abstractmethod
import pandas as pd



class Feature(ABC):
    

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique string identifier for this feature instance.
        Used as the column name in the DataFrame and the key in Features dict.
        e.g. 'ma_30', 'rsi_14', 'atr_14'
        Must be deterministic — same config always produces same name.
        """
        pass


    @property
    @abstractmethod
    def window(self) -> int:
        """
        Minimum number of bars required to produce a valid value.
        """
        pass


    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute this feature over the full DataFrame.
        Must only look backwards — no future data.
        Returns a Series aligned to df's index.
        """
        pass