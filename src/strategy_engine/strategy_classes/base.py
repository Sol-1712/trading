from abc import ABC, abstractmethod
from strategy_engine.features.registry import FeatureRegistry

class StrategyBase(ABC):
    
    def __init__(self, strategy_id: str, config: dict) -> None:
        self.strategy_id = strategy_id
        self.config      = config

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



    # @abstractmethod
    # def on_start(self) -> None:
    #     pass


    # @abstractmethod
    # def on_stop(self) -> None:
    #     pass