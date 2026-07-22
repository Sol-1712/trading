from datetime import datetime
from pathlib import Path
import yaml
from typing import Any

from .config_bases import BacktestConfig, ExecutionConfig, RunConfig
from trading.data_utils.core.config import DataConfig
from trading.backtester.risk import RiskConfig
from trading.data_utils.core.enums import PriceType
from trading.data_utils.core.paths import CONFIGS_ROOT
from trading.backtester.fill import FillModel, FILL_MODELS


_DATE_FORMATS = ["%d/%m/%Y", "%Y-%m-%d"]


def load_config(
    file:      str | Path     = "default_backtest.yaml",
    overrides: dict[str, Any] | None = None,
) -> BacktestConfig:
    """
    Load and parse backtest configuration from a YAML file.

    Merges file-based config with optional dotted-key overrides, then
    builds a validated BacktestConfig via build_config.

    Parameters
    ----------
    file : str or Path, default "default_backtest.yaml"
        Configuration filename relative to CONFIGS_ROOT / 'backtests',
        or an absolute Path to the YAML file.
    overrides : dict[str, Any] or None
        Optional key-value overrides using dot notation for nested
        fields (e.g. ``"data.symbol"``).

    Returns
    -------
    BacktestConfig
        Complete validated backtest configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If YAML is empty/invalid or required fields are missing.
    """

    path = CONFIGS_ROOT / 'backtests' / file if isinstance(file, str) else file

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw: dict = yaml.safe_load(f) or {}

    if not raw:
        raise ValueError(f"Config file {path} is empty or invalid YAML")

    if overrides:
        raw = _apply_overrides(raw, overrides)

    return build_config(raw)


def build_config(raw: dict) -> BacktestConfig:
    """
    Build BacktestConfig from a raw configuration dictionary.

    Assembles run, data, execution, and risk sub-configs plus
    initial_capital from the corresponding top-level keys.

    Parameters
    ----------
    raw : dict
        Raw configuration with keys: run, data, execution, risk,
        initial_capital.

    Returns
    -------
    BacktestConfig
        Validated composite backtest configuration.

    Raises
    ------
    ValueError
        If a required top-level field is missing or a sub-config
        value is invalid.
    """
    try:
        return BacktestConfig(
            run             = _build_run_config(raw["run"]),
            data            = _build_data_config(raw["data"]),
            execution       = _build_execution_config(raw["execution"]),
            risk            = _build_risk_config(raw["risk"]),
            initial_capital = float(raw["initial_capital"]),
        )
    except KeyError as e:
        raise ValueError(f"Missing required config field: {e}") from e
    except (ValueError, TypeError):
        raise

# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------
def _build_run_config(raw: dict) -> RunConfig:
    return RunConfig(
        name        = str(raw.get("name", "unnamed")),
        description = str(raw.get("description", "")),
        tags        = tuple(raw.get("tags", []))
    )

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
        Raw configuration with required ``fee_rate`` and optional
        delay_bars, fill_model, and mtm_price_type.

    Returns
    -------
    ExecutionConfig
        Validated execution configuration with defaults applied.

    Raises
    ------
    ValueError
        If fee_rate is missing.
    """
    if "fee_rate" not in raw:
        raise ValueError("Missing required field in [execution] config: fee_rate")

    return ExecutionConfig(
        fee_rate             = float(raw["fee_rate"]),
        delay_bars           = int(raw.get("delay_bars", 1)),
        fill_model           = raw.get("fill_model", "market"), 
        mtm_price_type       = PriceType(raw.get("mtm_price_type", 'mark'))
    )
    


def _build_risk_config(raw: dict)-> RiskConfig:
    return RiskConfig(
        leverage_max = float(raw.get("leverage_max", 100.0)),
        max_drawdown = float(raw.get("max_drawdown", 0.20)),
        max_position = float(raw.get("max_position", 1.0)),
    )
  
  
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
