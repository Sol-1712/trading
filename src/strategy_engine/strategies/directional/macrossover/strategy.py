from typing import Any
import pandas as pd
from strategy_engine.core_classes import Signal
from strategy_engine.strategies import DirectionalStrategy
from strategy_engine.features import FeatureRegistry
from strategy_engine.features.trend import SMA
from .config import MACrossoverConfig



class MACrossover(DirectionalStrategy[MACrossoverConfig]):

    def __init__(self, config: MACrossoverConfig) -> None:
        super().__init__(config)


    def on_start(self) -> None:
        raise NotImplementedError


    def on_stop(self) -> None:
        raise NotImplementedError


    def required_features(self) -> list[str]:
        # List of all the features the strategy needs, as strings
        #return [f'ma_{self.fast}', f'ma_{self.slow}']
        req  = [f'ma_{self.config.fast_window}',
                f'ma_{self.config.slow_window}'
               ]
        return req


    def register_features(self, registry: FeatureRegistry) -> None:
        # .register gets the name, just need to initialise objects.
        registry.register(SMA(self.config.fast_window))
        registry.register(SMA(self.config.slow_window))


    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        fast = df[f'ma_{self.config.fast_window}']
        slow = df[f'ma_{self.config.slow_window}']

        signals = []

        for timestamp, f, s in zip(df.index, fast, slow):
            if pd.isna(f) or pd.isna(s):
                continue      


        return signals


