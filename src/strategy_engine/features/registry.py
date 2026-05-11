from strategy_engine.features import Feature
import pandas as pd

class FeatureRegistry:
    """
    Holds registered Feature instances, keyed by name.
    Strategies register what they need.
    Registry calls each stored Feature's compute function.
    """

    def __init__(self):
        self._registry: dict[str, Feature] = {}


    def register(self, feature: Feature) -> None:
        if feature.name in self._registry:
            raise ValueError(
                f"Feature '{feature.name}' already registered."
            )
        self._registry[feature.name] = feature


    def get(self, name: str) -> Feature:
        if name not in self._registry:
            raise ValueError(f"Feature '{name}' not registered.")
        return self._registry[name]


    def validate(self, requested: list[str]) -> None:
        missing = [f for f in requested if f not in self._registry]
        if missing:
            raise ValueError(f"Unregistered features: {missing}")


    def compute_batch(self, df: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
        self.validate(requested)
        out = df.copy()
        for name in requested:
            out[name] = self._registry[name].compute(df)
        return out