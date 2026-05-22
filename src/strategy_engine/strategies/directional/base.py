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
        self._features: list[Feature] | None = None

        self._signal_state: SignalDirection  = SignalDirection.FLAT


    @abstractmethod
    def _build_features(self) -> list[Feature]:
        """
        Construct all Feature instances this strategy needs.
        Built once and cached — required_features() and 
        register_features() both derive from this.
        """


    def _get_features(self) -> list[Feature]:
        if self._features is None:
            self._features = self._build_features()
        return self._features


    def register_features(self, registry: FeatureRegistry) -> None:
        for feature in self._get_features():
            registry.register(feature)


    def required_features(self) -> list[str]:
        return [f.name for f in self._get_features()]


    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """
        Takes the feature-rich dataset and generates an array of Signal objects.
        This is the actual strategy logic
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


