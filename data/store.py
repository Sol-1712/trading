import pandas as pd
from pathlib import Path
import os

DATA_ROOT = Path("data") / "raw" 
EXCHANGE = 'bybit'

### Should add a columns argument to loading



def make_file_path(
    data_type: str,
    symbol: str, 
    interval: int | None = None
    ) -> Path:

    """
    Creates file path for given paramters.

    Parameters
    ----------
    data_type : str
        Requested data type e.g 'ohlcv'
    symbol : str
        Instrument symbol e.g 'BTCUSDT'.
    interval : int (optional)
        Bar interval in minutes.
        Required for OHLCV data.

    Returns
    -------
    Path
        Path object.
    """

    data_type = data_type.lower()
    symbol = symbol.upper()

    base = DATA_ROOT / data_type / EXCHANGE / symbol

    if data_type == "ohlcv":
        if interval is None:
            raise ValueError("interval must be provided for ohlcv")
        base = base / f"{interval}m"
    elif data_type == "funding":
        pass
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    return base


def save_partitioned_parquet(
    df: pd.DataFrame,
    base_path: str,
    data_type: str,
    symbol: str,
    interval: int | None = None,
    mode: str = 'merge'
    ) -> None:
    """
    Save a DataFrame as Parquet files partitioned by year/month/day.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. Must contain a unix timestamp column
    base_dir : str or Path
        Root directory where all data will be saved.
    data_type : str
        Dataset type, e.g. "ohlcv" or "funding".
    symbol : str
        Instrument symbol, e.g. "BTCUSDT".
    interval : int, optional
        Bar interval in minutes (required for OHLCV data).
    mode : str, optional
        Write mode: "merge" or "overwrite".
    """

    if df.empty:
        print("DataFrame is empty, nothing to save.")
        return

    data_type = data_type.lower()
    symbol = symbol.upper()

    # Work on a copy so we don't mutate the original
    data = df.copy()

    # Ensure index is a DatetimeIndex in UTC
    if not isinstance(data.index, pd.DatetimeIndex):
        # assume Unix ms
        data.index = pd.to_datetime(data.index, unit="ms", utc=True)
    else:
        # Make sure it's timezone-aware in UTC if not already
        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC")
        else:
            data.index = data.index.tz_convert("UTC")

    # Sort by time index just to be safe
    data = data.sort_index()

    # Add year/month helper columns for grouping
    data["year"] = data.index.year
    data["month"] = data.index.month

    # Ensure base directory exists
    os.makedirs(base_path, exist_ok=True)

    # Filename pattern per data_type
    if interval is not None:
        def make_filename(year: int, month: int) -> str:
            return f"{symbol}_{interval}m_{year}-{month:02d}.parquet"
    else:
        def make_filename(year: int, month: int) -> str:
            return f"{symbol}_{data_type}_{year}-{month:02d}.parquet"

    # Group by year + month and save/update each group
    for (year, month), group in data.groupby(["year", "month"]):
        # Drop helper cols before saving
        group = group.drop(columns=["year", "month"])

        # Folder: .../year=YYYY/month=MM/
        folder = os.path.join(base_path, f"year={year}", f"month={month:02d}")
        os.makedirs(folder, exist_ok=True)

        filename = make_filename(year, month)
        filepath = os.path.join(folder, filename)

        if mode == "merge":
            if os.path.exists(filepath):
                # Load existing parquet
                existing = pd.read_parquet(filepath)

                # Ensure DatetimeIndex
                if not isinstance(existing.index, pd.DatetimeIndex):
                    existing.index = pd.to_datetime(existing.index, utc=True)

                # Merge and deduplicate
                combined = pd.concat([existing, group])
                combined = (
                    combined[~combined.index.duplicated(keep="last")]
                    .sort_index()
                )

                combined.to_parquet(filepath)

                print(
                    f"Updated {filepath} — "
                    f"{len(existing)} existing, "
                    f"{len(combined) - len(existing)} new, "
                    f"{len(combined)} total"
                )

            else:
                print(f"[merge] File not found — creating new file instead: {filepath}")
                group.to_parquet(filepath)
                print(f"Created {filepath} ({len(group)} rows)")

        elif mode == "overwrite":
            # Overwrite always writes only the new group
            group.to_parquet(filepath)
            print(f"Overwritten {filepath} with {len(group)} rows")

        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Expected 'merge' or 'overwrite'."
    )


def _month_starts(
    start_dt: pd.Timestamp, 
    end_dt: pd.Timestamp
    ) -> list[pd.Timestamp]:

    start_dt = start_dt.normalize().replace(day=1)
    end_dt = end_dt.normalize().replace(day=1)
    out = []
    cur = start_dt
    while cur <= end_dt:
        out.append(cur)
        cur = cur + pd.offsets.MonthBegin(1)
    return out


def load_parquet_partitioned(
    base_path: Path,
    start: str | None = None, # MM/DD/YYYY
    end: str | None = None, # Stops at the last entry before end point
    index_as: str = "datetime",     # "datetime" or "unix_ms"
    columns: list[str] | None = None,
) -> pd.DataFrame:
    
    """
    Load parquet partitions under:
      base_path/year=YYYY/month=MM/*.parquet

    base_path example:
      .../ohlcv/bybit/BTCUSDT/15m
    """

    # Parse bounds
    start_dt = pd.Timestamp(start, tz="UTC") if start is not None else None
    end_dt = pd.Timestamp(end, tz="UTC") if end is not None else None

    # Build file list
    if start_dt is None and end_dt is None:
        files = sorted(base_path.rglob("*.parquet"))
    else:
        if start_dt is None:
            start_dt = pd.Timestamp("1970-01-01", tz="UTC")
        if end_dt is None:
            end_dt = pd.Timestamp.utcnow().tz_convert("UTC")

        files = []
        for m in _month_starts(start_dt, end_dt):
            y = m.year
            mm = f"{m.month:02d}"
            month_dir = base_path / f"year={y}" / f"month={mm}"
            if month_dir.exists():
                files.extend(sorted(month_dir.glob("*.parquet")))

    if not files:
        return pd.DataFrame()

    # Read + concat
    df = pd.concat((pd.read_parquet(f, columns=columns) for f in files), axis=0, copy=False)

    # Ensure datetime index (UTC)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

    df = df.sort_index()

    # Exact trimming
    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df.index < pd.Timestamp(end, tz="UTC")]

    # Optional de-dupe (useful if your updater overlaps)
    df = df[~df.index.duplicated(keep="last")]

    if index_as == "unix_ms":
        df.index = (df.index.view("int64") // 1_000_000).astype("int64")
    elif index_as != "datetime":
        raise ValueError("index_as must be 'datetime' or 'unix_ms'")

    return df
    
    
if __name__ == "__main__":

    symbol = 'BTCUSDT'
    interval = 60
    data_type = 'ohlcv'
    path = make_file_path(data_type, symbol, interval)

    # data_type = 'funding'
    # symbol = 'BTCUSDT'
    # path = make_file_path(data_type, symbol)

    # df = load_parquet(path)

    df = load_parquet_partitioned(path, start="01/01/2026", end="01/02/2026")
    print(df.head(10))
    print(df.tail(10))