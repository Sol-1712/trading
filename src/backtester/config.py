import yaml
from typing import Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "research" / "configs" 


@dataclass
class Config:
    symbol: str
    interval: int
    start: datetime
    end: datetime
    capital: float
    fee_rate: float
    delay_bars: int = 1
    price_column: str = 'mark_close' 
    leverage_max: float = 100


    def __post_init__(self):
        if self.capital <= 0:
            raise ValueError("capital must be positive")

        if not 0 <= self.fee_rate <= 0.01:
            raise ValueError("fee_rate out of range")

        if self.delay_bars < 1:
            raise ValueError("delay_bars must be >= 1")

        if self.start == self.end:
            raise ValueError("start and end are identical")

        if self.leverage_max < 1.0:
            raise ValueError("leverage must be >= 1.0")


def load_config(overrides: dict[str, Any] | None, file: str = "default_config.yaml")-> Config:
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
    """

    config_path = DEFAULT_PATH / file

    required = [
    "symbol", "interval", "start", "end",
    "capital", "fee_rate", "delay_bars", "price_column", 
    "leverage_max",
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
    raw["capital"] = float(raw["capital"])
    raw["fee_rate"] = float(raw["fee_rate"])
    raw["delay_bars"] = int(raw["delay_bars"])
    raw["symbol"] = str(raw["symbol"])
    raw["price_column"] = str(raw["price_column"])
    raw["leverage_max"] = float(raw["leverage_max"])

    # Date parsing
    raw["start"] = datetime.strptime(raw["start"], "%d/%m/%Y")
    raw["end"] = datetime.strptime(raw["end"], "%d/%m/%Y")

    return Config(**raw)


