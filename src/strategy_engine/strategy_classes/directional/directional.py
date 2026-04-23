from abc import abstractmethod
from typing import Optional
from datetime import datetime
from strategy_engine.core_classes.enums import SignalDirection
from strategy_engine.strategy_classes import StrategyBase
from strategy_engine.features import Feature
from strategy_engine.core_classes import StrategyContext
from strategy_engine.core_classes import Signal


class DirectionalStrategy(StrategyBase):

    @abstractmethod
    def generate_signals(
        self,
        features: Feature,
        context:  StrategyContext,
    ) -> Optional[Signal]: ...

    def _long(self, strength: float, timestamp: datetime, **meta) -> Signal:
        return Signal(SignalDirection.LONG, strength, timestamp,
                      source=self.__class__.__name__, metadata=meta)

    def _short(self, strength: float, timestamp: datetime, **meta) -> Signal:
        return Signal(SignalDirection.SHORT, strength, timestamp,
                      source=self.__class__.__name__, metadata=meta)

    def _flat(self, timestamp: datetime, **meta) -> Signal:
        return Signal(SignalDirection.FLAT, 0.0, timestamp,
                      source=self.__class__.__name__, metadata=meta)
    pass