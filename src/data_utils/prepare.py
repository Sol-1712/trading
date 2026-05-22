import logging
import pandas as pd

from data_utils.enums  import PriceType, DataType
from data_utils.config import DataConfig
from data_utils.paths  import make_data_path
from data_utils.io     import load_partitioned_parquet


logger = logging.getLogger(__name__)

_IDX = "timestamp"  # canonical index name used internally during merges

def prepare_data(
    config:      DataConfig,
    price_types: tuple[PriceType, ...],
    columns:     tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Load and merge klines and funding data into a single DataFrame
    ready for the strategy engine.

    Parameters
    ----------
    config : DataConfig
        Symbol, interval, and date range.
    price_types : tuple[PriceType, ...]
        Which klines series(s) to load (last / mark / index).
        Comes from strategy.data_requirements().
    columns : tuple[str, ...], optional
        If provided, only these columns are returned.
        Validated against available columns before filtering.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (UTC), sorted ascending.
        Kline columns + funding columns where available.
    """
    kline_frames = [_load_klines(config, pt) for pt in price_types]
    
    kline_frames = [
        _prefix_columns(df, pt) 
        for df, pt in zip(kline_frames, price_types)
    ]
    
    klines  = pd.concat(kline_frames, axis=1) if len(kline_frames) > 1 else kline_frames[0]
    funding = _load_funding(config)
    data    = _merge_funding(klines, funding) if funding is not None else klines

    _validate(data, config)

    if columns is not None:
        missing = set(columns) - set(data.columns)
        if missing:
            raise ValueError(
                f"Strategy requested columns not present in data: {missing}. "
                f"Available: {sorted(data.columns)}"
            )
        return data[list(columns)]

    return data

def _prefix_columns(df: pd.DataFrame, price_type: PriceType) -> pd.DataFrame:
    return df.rename(columns={col: f"{price_type.value}_{col}" for col in df.columns})


# ---------------------------------------------------------------------------
# Private loaders
# ---------------------------------------------------------------------------

def _load_klines(config: DataConfig, price_type: PriceType) -> pd.DataFrame:
    path = make_data_path(
        symbol     = config.symbol,
        data_type  = DataType.KLINES,
        interval   = config.interval,
        price_type = price_type,
    )
    df = load_partitioned_parquet(path, start=config.start, end=config.end)

    if df.empty:
        raise ValueError(
            f"No kline data found for {config.symbol} | "
            f"{config.interval}m | {price_type.value} | "
            f"{config.start:%Y-%m-%d} → {config.end:%Y-%m-%d}"
        )
    return df


def _load_funding(config: DataConfig) -> pd.DataFrame | None:
    path = make_data_path(
        symbol    = config.symbol,
        data_type = DataType.FUNDING,
    )

    if not path.exists():
        logger.warning(
            "No funding data found for %s — proceeding without funding rates.",
            config.symbol,
        )
        return None

    df = load_partitioned_parquet(path, start=config.start, end=config.end)

    if df.empty:
        logger.warning(
            "Funding path exists but no data in range for %s.", config.symbol
        )
        return None

    return df


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _merge_funding(klines: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """
    Join funding onto kline bars by exact timestamp.
    Bars without a funding event get 0 — funding is a discrete cash flow,
    not a continuous rate. Strategies that need a continuous signal
    should forward-fill in the feature layer.
    """
    merged = klines.join(funding, how="left")
    merged[funding.columns] = merged[funding.columns].fillna(0.0)
    return merged


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(data: pd.DataFrame, config: DataConfig) -> None:
    if data.empty:
        raise ValueError("Merged dataset is empty after combining klines and funding.")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(
            "Data does not have a DatetimeIndex — check save format."
        )
    
    bar_duration    = pd.Timedelta(minutes=config.interval)
    requested_start = pd.Timestamp(config.start, tz="UTC")
    requested_end   = pd.Timestamp(config.end,   tz="UTC")
    actual_start    = data.index[0]
    actual_end      = data.index[-1]

    if actual_start > requested_start:
        logger.warning(
            "Data starts at %s, later than requested %s. "
            "Results will cover a shorter window.",
            actual_start.date(), requested_start.date(),
        )
    if actual_end < requested_end - bar_duration:
        logger.warning(
            "Data ends at %s, earlier than requested %s. "
            "Results will cover a shorter window.",
            actual_end.date(), requested_end.date(),
        )