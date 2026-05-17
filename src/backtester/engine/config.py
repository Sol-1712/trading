import yaml
from typing import Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from backtester.engine.enums import PriceType

DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "research" / "configs" 


@dataclass
class Config:
    symbol: str
    interval: int
    start: datetime
    end: datetime
    initial_capital: float
    fee_rate: float
    delay_bars: int = 1
    signal_price_type:    PriceType = PriceType.LAST
    execution_price_type: PriceType = PriceType.LAST
    leverage_max: float = 100
    

    def __post_init__(self):
        if self.initial_capital <= 0:
            raise ValueError("capital must be positive")

        if not 0 <= self.fee_rate <= 0.01:
            raise ValueError("fee_rate out of range")

        if self.delay_bars < 1:
            raise ValueError("delay_bars must be >= 1")

        if self.start == self.end:
            raise ValueError("start and end are identical")

        if self.leverage_max < 1.0:
            raise ValueError("leverage must be >= 1.0")


def load_config(overrides: dict[str, Any] | None, file: str = "default_config.yaml") -> Config:
    """
    Load backtest config from YAML with optional overrides.

    Parameters
    ----------
    overrides : dict, optional
        Keys to override from default config e.g. {"symbol": "ETHUSDT"}
    file : str, optional
        Name of config file. Defaults to default_config.yaml.

    Returns
    -------
    Config
        Validated Config object.
        Required attributes:
            "symbol", "interval", "start", "end",
            "initial_capital", "fee_rate", "delay_bars", 
            "signal_price_type", "execution_price_type", "leverage_max",

    Default
    -------
    >>> symbol:               BTCUSDT
    >>> interval:             60
    >>> start:                01/01/2025
    >>> end:                  01/01/2026
    >>> initial_capital:      100000
    >>> fee_rate:             0.000550
    >>> delay_bars:           1
    >>> signal_price_type:    last
    >>> execution_price_type: last
    >>> leverage_max:         100
    """

    config_path = DEFAULT_PATH / file

    required = [
    "symbol", "interval", "start", "end",
    "initial_capital", "fee_rate", "delay_bars", 
    "signal_price_type", "execution_price_type", "leverage_max"
    ]

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
        
    if overrides: 
        unknown = set(overrides) - set(raw)
        if unknown:
            raise ValueError(f"Unknown override keys: {unknown}")
        raw.update(overrides)

    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Missing keys in config: {missing}")
    
    # Type normalization
    raw["interval"] = int(raw["interval"])
    raw["initial_capital"] = float(raw["initial_capital"])
    raw["fee_rate"] = float(raw["fee_rate"])
    raw["delay_bars"] = int(raw["delay_bars"])
    raw["symbol"] = str(raw["symbol"])
    try:
        raw["signal_price_type"]  = PriceType(raw["signal_price_type"])
        raw["execution_price_type"] = PriceType(raw["execution_price_type"])
    except ValueError as e:
        raise ValueError(f"Invalid price config: {e}") from e
    raw["leverage_max"] = float(raw["leverage_max"])
    
    # Date parsing
    raw["start"] = datetime.strptime(raw["start"], "%d/%m/%Y")
    raw["end"] = datetime.strptime(raw["end"], "%d/%m/%Y")

    return Config(**raw)


