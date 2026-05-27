import logging
from datetime import datetime
from pathlib import Path
import yaml
from typing import Any

from .config import BacktestConfig
from backtester.engine.execution import ExecutionConfig
from data_utils.config import DataConfig
from data_utils.enums import PriceType





_DATE_FORMATS = ["%d/%m/%Y", "%Y-%m-%d"]
DEFAULT_PATH  = Path(__file__).resolve().parent.parent.parent.parent / "research" / "configs"


def _parse_date(value: Any) -> datetime:
    """
    Parse date from multiple formats.
    
    Parameters
    ----------
    value : Any
        Date value to parse. If already datetime, returned as-is.
        
    Returns
    -------
    datetime
        Parsed datetime object.
        
    Raises
    ------
    ValueError
        If value cannot be parsed in any supported format.
    """
    if isinstance(value, datetime):
        return value
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date '{value}'. Supported formats: {_DATE_FORMATS}")


def _apply_overrides(raw: dict, overrides: dict[str, Any]) -> dict:
    """
    Apply dotted-key overrides to nested config dict.
    
    Allows dot notation to navigate nested dicts:
    "data.symbol" → raw["data"]["symbol"]
    
    Parameters
    ----------
    raw : dict
        Base configuration dictionary.
    overrides : dict[str, Any]
        Override key-value pairs with optional dot notation.
        
    Returns
    -------
    dict
        Updated configuration with overrides applied.
        
    Raises
    ------
    ValueError
        If override key references unknown section or top-level key.
    """
    result = {k: dict(v) if isinstance(v, dict) else v for k, v in raw.items()}

    for key, value in overrides.items():
        parts = key.split(".", maxsplit=1)

        if len(parts) == 2:
            section, field = parts
            if section not in result or not isinstance(result[section], dict):
                raise ValueError(f"Unknown config section in override: '{section}'")
            result[section][field] = value

        else:
            if key not in result:
                raise ValueError(
                    f"Unknown top-level override key: '{key}'. "
                    f"Use dotted notation for nested fields e.g. 'data.symbol'."
                )
            result[key] = value

    return result


def _build_data_config(raw: dict) -> DataConfig:
    """
    Build DataConfig from raw config dict.
    
    Parameters
    ----------
    raw : dict
        Raw configuration with keys: symbol, interval, start, end.
        
    Returns
    -------
    DataConfig
        Validated data configuration.
        
    Raises
    ------
    KeyError
        If required fields are missing.
    ValueError
        If date parsing fails or interval is invalid.
    """
    if not all(k in raw for k in ['symbol', 'interval', 'start', 'end']):
        missing = [k for k in ['symbol', 'interval', 'start', 'end'] if k not in raw]
        raise ValueError(f"Missing required data config fields: {missing}")
    
    return DataConfig(
        symbol   = str(raw["symbol"]),
        interval = int(raw["interval"]),
        start    = _parse_date(raw["start"]),
        end      = _parse_date(raw["end"]),
    )


def _build_execution_config(raw: dict) -> ExecutionConfig:
    """
    Build ExecutionConfig from raw config dict.
    
    Parameters
    ----------
    raw : dict
        Raw configuration with required fee_rate and optional other fields.
        
    Returns
    -------
    ExecutionConfig
        Validated execution configuration with defaults applied.
        
    Raises
    ------
    KeyError
        If fee_rate is missing.
    ValueError
        If fee_rate is invalid.
    """
    if 'fee_rate' not in raw:
        raise ValueError("Missing required execution config field: fee_rate")
        
    return ExecutionConfig(
        fee_rate             = float(raw["fee_rate"]),
        delay_bars           = int(raw.get("delay_bars", 1)),
        execution_price_type = raw.get("execution_price_type", PriceType.LAST),
        leverage_max         = float(raw.get("leverage_max", 100.0)),
    )


def load_config(
    file:      str         = "default.yaml",
    overrides: dict[str, Any] | None = None,
) -> BacktestConfig:
    """
    Load and parse backtest configuration from YAML file.
    
    Merges file-based config with optional dotted-key overrides.
    Validates all required fields are present and valid.
    
    Parameters
    ----------
    file : str, default "default.yaml"
        Configuration filename (relative to DEFAULT_PATH).
    overrides : dict[str, Any] | None
        Optional key-value overrides using dot notation for nested fields.
        
    Returns
    -------
    BacktestConfig
        Complete validated backtest configuration.
        
    Raises
    ------
    FileNotFoundError
        If config file not found at DEFAULT_PATH / file.
    ValueError
        If YAML is malformed or required fields are missing.
    """
    config_path = DEFAULT_PATH / file
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw: dict = yaml.safe_load(f) or {}

    if not raw:
        raise ValueError(f"Config file {config_path} is empty or invalid YAML")

    if overrides:
        raw = _apply_overrides(raw, overrides)

    try:
        return BacktestConfig(
            data            = _build_data_config(raw["data"]),
            execution       = _build_execution_config(raw["execution"]),
            initial_capital = float(raw["initial_capital"]),
        )
    except KeyError as e:
        raise ValueError(f"Missing required config field: {e}") from e
    except (ValueError, TypeError) as e:
        logger.error("Config parsing failed: %s", e)
        raise