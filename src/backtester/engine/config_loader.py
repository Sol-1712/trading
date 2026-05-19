from datetime import datetime
from pathlib import Path
import yaml
from typing import Any

from .config import ExecutionConfig, BacktestConfig
from data_utils.config import DataConfig
from data_utils.enums import PriceType





_DATE_FORMATS = ["%d/%m/%Y", "%Y-%m-%d"]
DEFAULT_PATH  = Path(__file__).resolve().parent.parent.parent.parent / "research" / "configs"


def _parse_date(value: Any) -> datetime:
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
    "data.symbol" -> raw["data"]["symbol"]
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
    return DataConfig(
        symbol   = str(raw["symbol"]),
        interval = int(raw["interval"]),
        start    = _parse_date(raw["start"]),
        end      = _parse_date(raw["end"]),
    )


def _build_execution_config(raw: dict) -> ExecutionConfig:
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
    config_path = DEFAULT_PATH / file
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw: dict = yaml.safe_load(f) or {}

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