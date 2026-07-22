"""
Canonical path resolution for the raw data store.

Directory layout::

    data/
    └── raw/
        └── {exchange}/
            └── {SYMBOL}/
                ├── klines/
                │   └── {interval}m/
                │       ├── last/   <- last price OHLCV
                │       ├── mark/   <- mark price OHLC
                │       └── index/  <- index price OHLC
                |          |___ year=/month=/
                └── funding/        <- funding rate history

"""

from __future__ import annotations

from pathlib import Path

import trading
from trading.data_utils.core.enums import PriceType, DataType


PROJECT_ROOT = Path(trading.__file__).resolve().parents[2]

DATA_ROOT    = PROJECT_ROOT / "data" / "raw"
CONFIGS_ROOT = PROJECT_ROOT / "configs"
LOGS_ROOT    = PROJECT_ROOT / "logs"

DEFAULT_EXCHANGE = 'bybit'


def make_data_path(
    symbol: str,
    data_type: DataType,
    exchange: str = DEFAULT_EXCHANGE,
    interval: int | None = None,
    price_type: PriceType | None = None,
    validate_exists: bool = False,
) -> Path:
    """
    Construct a canonical, deterministic path to a dataset directory.

    Parameters
    ----------
    symbol : str
        Instrument symbol, e.g. ``"BTCUSDT"``. Uppercased automatically.
    data_type : DataType
        ``"klines"`` or ``"funding"``.
    exchange : str
        Exchange name. Lowercased automatically. Default: ``"bybit"``.
    interval : int, optional
        Bar interval in minutes. Required for ``data_type="klines"``.
    price_type : PriceType, optional
        One of ``"last"``, ``"mark"``, ``"index"``.
        Required for ``data_type="klines"``.
    validate_exists : bool
        If ``True``, raises ``FileNotFoundError`` when the path does not
        exist on disk. Useful for read paths; leave ``False`` for write paths.

    Returns
    -------
    Path
        Absolute path to the dataset directory (not a file).

    Raises
    ------
    ValueError
        If required arguments are missing or invalid.
    FileNotFoundError
        If ``validate_exists=True`` and the path does not exist.

    Examples
    --------
    >>> make_data_path("BTCUSDT", "klines", interval=1, price_type="last")
    PosixPath('.../data/raw/bybit/BTCUSDT/klines/1m/last')

    >>> make_data_path("BTCUSDT", "funding")
    PosixPath('.../data/raw/bybit/BTCUSDT/funding')
    """
    symbol = symbol.upper()
    exchange = exchange.lower()

    base = DATA_ROOT / exchange / symbol

    if data_type == DataType.KLINES:
        _validate_klines_args(interval, price_type)
        assert price_type is not None  # for type checker
        path = base / "klines" / f"{interval}m" / price_type 

    elif data_type == DataType.FUNDING:
        path = base / "funding"

    else:
        # Unreachable after validation, but keeps type checker happy.
        raise ValueError(f"Unhandled data_type: {data_type!r}")

    if validate_exists and not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    return path


def list_available_symbols(
    exchange: str = DEFAULT_EXCHANGE,
) -> list[str]:
    """
    Return all symbols that have data under the given exchange.

    Parameters
    ----------
    exchange : str
        Exchange name. Default: ``"bybit"``.

    Returns
    -------
    list[str]
        Sorted list of symbol names (e.g. ``["BTCUSDT", "ETHUSDT"]``).
    """
    exchange_root = DATA_ROOT / exchange.lower()
    if not exchange_root.exists():
        return []
    return sorted(p.name for p in exchange_root.iterdir() if p.is_dir())


def list_available_intervals(
    symbol: str,
    price_type: PriceType,
    exchange: str = DEFAULT_EXCHANGE,
) -> list[int]:
    """
    Return all intervals (in minutes) available for a symbol/price_type.

    Parameters
    ----------
    symbol : str
        Instrument symbol, e.g. ``"BTCUSDT"``.
    price_type : PriceType
        Price series directory to inspect (``last`` / ``mark`` / ``index``).
    exchange : str
        Exchange name. Default: ``"bybit"``.

    Returns
    -------
    list[int]
        Sorted list of available intervals in minutes.
    """
    klines_root = DATA_ROOT / exchange.lower() / symbol.upper() / "klines"
    if not klines_root.exists():
        return []

    intervals: list[int] = []
    for interval_dir in klines_root.iterdir():
        if not interval_dir.is_dir():
            continue
        price_type_dir = interval_dir / price_type
        if price_type_dir.exists():
            try:
                minutes = int(interval_dir.name.removesuffix("m"))
                intervals.append(minutes)
            except ValueError:
                continue

    return sorted(intervals)


# ---------------------------------------------------------------------------
# Internal validators
# ---------------------------------------------------------------------------

def _validate_klines_args(interval: int | None, price_type: PriceType | None) -> None:
    if interval is None:
        raise ValueError("'interval' is required for data_type=KLINES.")
    if price_type is None:
        raise ValueError("'price_type' is required for data_type=KLINES.")