from trading.strategy_engine.features import Feature
import pandas as pd

class FeatureRegistry:
    """
    Holds registered Feature instances, keyed by name.
    Strategies register what they need.
    Registry calls each stored Feature's compute function.
    """

    def __init__(self) -> None:
        self._features: dict[str, Feature] = {}

    def register(self, feature: Feature) -> None:
        if feature.name in self._features:
            raise ValueError(
                f"Feature '{feature.name}' is already registered. "
                f"Ensure feature names are unique across the strategy."
            )
        self._features[feature.name] = feature

    def compute_batch(
        self, 
        data:  pd.DataFrame, 
        names: list[str],
    ) -> pd.DataFrame:
        """
        Compute requested features and return enriched DataFrame.
        Features are added as new columns — existing columns unchanged.
        
        Parameters
        ----------
        data : pd.DataFrame
            Raw market data.
        names : list[str]
            Feature names to compute. Must all be registered.
        """
        missing = set(names) - self._features.keys()
        if missing:
            raise KeyError(
                f"Requested features not registered: {missing}. "
                f"Registered: {set(self._features.keys())}"
            )

        result = data.copy()
        for name in names:
            result[name] = self._features[name].compute(result)
        return result