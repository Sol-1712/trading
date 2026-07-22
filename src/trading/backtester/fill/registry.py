"""Registry mapping fill-model config names to FillModel classes."""

from .market import MarketFillModel

FILL_MODELS = {
    "market": MarketFillModel,
}
