from ..base import StrategyBase
from ...features.registry import FeatureRegistry
from ...features.library.trend import MA

class MACrossover(StrategyBase):

    def __init__(self, strategy_id: str, config: dict) -> None:
        """
        config:
        'fast_period'
        'slow_period'
        """
        super().__init__(strategy_id, config)
        self.fast = config['fast_period']   # e.g. 30
        self.slow = config['slow_period']   # e.g. 90

    def required_features(self) -> list[str]:
        # just the names — the strings the engine will look up
        ### List of all the features the strat needs, as strings
        return [f'ma_{self.fast}', f'ma_{self.slow}']

    def register_features(self, registry: FeatureRegistry) -> None:
        ### .register gets the name, just need to initialise objects.
        registry.register(MA(self.fast))
        registry.register(MA(self.slow))