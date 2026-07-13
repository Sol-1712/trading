
from .strategies.directional import MACrossover, MACrossoverConfig



STRATEGY_REGISTRY = {
    "ma_crossover": (MACrossoverConfig, MACrossover),
}