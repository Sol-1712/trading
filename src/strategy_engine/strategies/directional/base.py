from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime
import pandas as pd

from strategy_engine.core.enums import SignalDirection
from strategy_engine.strategies import StrategyBase, ConfigT
from strategy_engine.features import Feature, FeatureRegistry
from strategy_engine.core import Signal



class DirectionalStrategy(StrategyBase[ConfigT], ABC):


    def __init__(self, config: ConfigT) -> None:
        """
        Abstract class for directional strategies

        Args:      
            config: Generic[ConfigT]
            
        """
        
        super().__init__(config)
        self.config = config


    @abstractmethod
    def required_features(self) -> list[str]:
        """
        Names of all features this strategy needs.
        Called by the engine before the backtest starts.
        e.g. ['ma_30', 'ma_90', 'atr_14']
        """
        pass


    @abstractmethod
    def register_features(self, registry: FeatureRegistry) -> None:
        """
        Constructs and adds each feature object to the registry
        e.g registry.register(MA(30))   
        """


    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """
        Takes the feature-rich dataset and generates an array of Signal objects.
        Args:
            df: pd.DataFrame
        """
        pass


    def _long(self, strength: float, timestamp: datetime, **meta) -> Signal:
        return Signal(SignalDirection.LONG, strength, timestamp,
                      source=self.__class__.__name__, metadata=meta)

    def _short(self, strength: float, timestamp: datetime, **meta) -> Signal:
        return Signal(SignalDirection.SHORT, strength, timestamp,
                      source=self.__class__.__name__, metadata=meta)

    def _flat(self, timestamp: datetime, **meta) -> Signal:
        return Signal(SignalDirection.FLAT, 0.0, timestamp,
                      source=self.__class__.__name__, metadata=meta)


