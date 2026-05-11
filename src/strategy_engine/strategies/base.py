from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from dataclasses import dataclass

ConfigT = TypeVar("ConfigT")


class StrategyBase(ABC, Generic[ConfigT]):

    def __init__(self, config: ConfigT) -> None:
        """
        Abstract base class for strategies
        
        """
        self.config = config


    @abstractmethod
    def on_start(self) -> None:
        pass


    @abstractmethod
    def on_stop(self) -> None:
        pass



@dataclass(frozen=True)
class StrategyConfig:
    strategy_id: str