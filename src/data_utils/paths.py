from pathlib import Path

# utils -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data" / "raw"

DEFAULT_EXCHANGE = "bybit"

VALID_DATA_TYPES = {"ohlcv", "funding"}


def make_data_path(
    data_type: str,
    symbol: str,
    interval: int | None = None,
    exchange: str = DEFAULT_EXCHANGE,
    validate_exists: bool = True,
) -> Path:
    """
    Construct an absolute, deterministic path to a dataset.

    Parameters
    ----------
    data_type : str
        'ohlcv' or 'funding'
    symbol : str
        Instrument symbol (e.g., BTCUSDT)
    interval : int, optional
        Required for ohlcv data
    exchange : str
        Exchange name (default: bybit)
    validate_exists : bool
        If True, raises if path does not exist

    Returns
    -------
    Path
        Absolute path to dataset directory
    """

    data_type = data_type.lower()
    symbol = symbol.upper()
    exchange = exchange.lower()

    if data_type not in VALID_DATA_TYPES:
        raise ValueError(
            f"Invalid data_type '{data_type}'. "
            f"Valid options: {VALID_DATA_TYPES}"
        )

    base = DATA_ROOT / data_type / exchange / symbol

    if data_type == "ohlcv":
        if interval is None:
            raise ValueError("interval must be provided for OHLCV data")
        base = base / f"{interval}m"

    if validate_exists and not base.exists():
        raise FileNotFoundError(
            f"Data path does not exist: {base}"
        )

    return base


