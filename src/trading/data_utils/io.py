
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

WriteMode = Literal["merge", "overwrite"]
Bound     = Literal["start", "end"]

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_partitioned_parquet(
    df: pd.DataFrame,
    dataset_path: str | Path,
    mode: WriteMode = "merge",
) -> None:
    """
    Persist a DataFrame into a year/month-partitioned parquet dataset.

    Directory layout::

        dataset_path/
            year=2024/
                month=01/
                    data.parquet
                month=02/
                    data.parquet
            year=2025/
                ...

    The DataFrame must be indexable as a time series. If the index is not
    already a ``DatetimeIndex``, it is assumed to be Unix milliseconds and
    converted to UTC.

    In ``"merge"`` mode, new rows are merged with any existing partition
    file. Duplicate timestamps are resolved by keeping the latest write
    (``keep="last"``). Use ``"overwrite"`` to replace partitions entirely.

    Parameters
    ----------
    df : pd.DataFrame
        Data to write. Must have a time-based index.
    dataset_path : str | Path
        Root directory of the partitioned dataset.
    mode : WriteMode
        ``"merge"`` (default) or ``"overwrite"``.
    """
    dataset_path = Path(dataset_path)

    if df.empty:
        logger.warning("save_partitioned_parquet called with empty DataFrame — nothing written.")
        return

    data = _ensure_utc_datetime_index(df.copy())
    data = data.sort_index()

    # Attach partition keys as temporary columns for groupby.
    assert isinstance(data.index, pd.DatetimeIndex)
    data["_year"] = data.index.year
    data["_month"] = data.index.month

    for (year, month), group in data.groupby(["_year", "_month"]):
        assert isinstance(year, int) and isinstance(month, int)
        group = group.drop(columns=["_year", "_month"])
        _write_partition(group, dataset_path, int(year), int(month), mode)


def _write_partition(
    group: pd.DataFrame,
    dataset_path: Path,
    year: int,
    month: int,
    mode: WriteMode,
) -> None:
    """Write a single month partition."""
    partition_dir = dataset_path / f"year={year}" / f"month={month:02d}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    file_path = partition_dir / "data.parquet"

    if mode == "merge" and file_path.exists():
        existing = pd.read_parquet(file_path)
        combined = (
            pd.concat([existing, group])
            .pipe(_ensure_utc_datetime_index)
            .sort_index()
        )
        # Resolve duplicates: last write wins.
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.to_parquet(file_path)
        logger.debug("Merged %s (%d rows).", file_path, len(combined))

    else:
        group.to_parquet(file_path)
        logger.debug("Wrote %s (%d rows).", file_path, len(group))


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------
def get_stored_range(path: str | Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """
    Returns the timestamp range stored in a partitioned dataset,
    or None if no data exists yet.
    
    Parameters
    ----------
    path : str | Path
        Root directory of the partitioned dataset.
    
    Returns
    -------
    tuple[pd.Timestamp, pd.Timestamp] | None
        (min_timestamp, max_timestamp) in the dataset, or None if no data exists.
    """

    path = Path(path)

    try:
        year_dirs  = sorted(path.glob("year=*"))
        dirs: dict[Bound, Path] = {}
        dirs["start"] = year_dirs[0]  # Earliest year directory
        dirs["end"]   = year_dirs[-1] # Latest year directory

    except (IndexError, ValueError):
        logger.warning("No valid year directories found in path: %s", path)
        return None
    
    timestamps: dict[Bound, pd.Timestamp] = {}
    for bound, year_dir in dirs.items():
        try:
            month_dirs = sorted(year_dir.glob("month=*")) 
            if bound == "start":
                month_dir = month_dirs[0]  # Earliest month directory
            else:
                month_dir = month_dirs[-1] # Latest month directory

        except (IndexError, ValueError):
            logger.warning("No valid month directories found in year directory: %s", year_dir)
            return None

        file_path = month_dir / "data.parquet"

        if file_path.exists():
            df = pd.read_parquet(file_path,).iloc[:, :0]  # Load only the index (no columns)
        else:
            logger.warning("No parquet files found in path: %s", path)
            return None
    
        if len(df.index) == 0:
            logger.warning("get_stored_range retrieved an empty DataFrame")
            return None

        timestamps[bound] = df.index.min() if bound == "start" else df.index.max()

    return (timestamps["start"], timestamps["end"])


def load_partitioned_parquet(
    path: str | Path,
    cols: list[str] | None = None,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    max_workers: int = 8,
) -> pd.DataFrame:
    """
    Load a partitioned parquet dataset with optional column selection
    and date-range filtering.

    Partition files are loaded in parallel to reduce wall-clock time on
    large date ranges.

    Parameters
    ----------
    path : str | Path
        Root directory of the partitioned dataset.
    cols : list[str], optional
        Columns to load. If None, all columns are loaded.
    start : str | datetime, optional
        Inclusive start of the date range.
    end : str | datetime, optional
        Exclusive end of the date range.
    max_workers : int
        Number of threads for parallel partition loading.

    Returns
    -------
    pd.DataFrame
        Sorted by UTC DatetimeIndex, filtered to [start, end).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If any loaded partition lacks a valid DatetimeIndex.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    start_ts = pd.to_datetime(start, utc=True) if start is not None else None
    end_ts = pd.to_datetime(end, utc=True) if end is not None else None

    partition_files = _collect_partition_files(path, start_ts, end_ts)

    if not partition_files:
        logger.warning("No partitions found in range for path: %s", path)
        return pd.DataFrame()

    dfs = _load_partitions_parallel(partition_files, cols, max_workers)

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs).sort_index()

    if not isinstance(result.index, pd.DatetimeIndex):
        raise ValueError(
            f"Loaded data from {path} does not have a DatetimeIndex. "
            "Ensure data was saved via save_partitioned_parquet."
        )

    # Precise row-level filtering after combining partitions.
    if start_ts is not None:
        result = result[result.index >= start_ts]
    if end_ts is not None:
        result = result[result.index < end_ts]

    return result


def _collect_partition_files(
    path: Path,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> list[Path]:
    """
    Walk the partition directory tree and return parquet file paths
    whose year/month range overlaps [start, end].
    """
    files: list[Path] = []

    for year_dir in sorted(path.glob("year=*")):
        try:
            year = int(year_dir.name.split("=")[1])
        except (IndexError, ValueError):
            logger.warning("Skipping malformed directory: %s", year_dir)
            continue

        for month_dir in sorted(year_dir.glob("month=*")):
            try:
                month = int(month_dir.name.split("=")[1])
            except (IndexError, ValueError):
                logger.warning("Skipping malformed directory: %s", month_dir)
                continue

            if start is not None or end is not None:
                # Partition covers [partition_start, partition_end] inclusive.
                # partition_end is the last nanosecond of the month to avoid
                # incorrectly skipping partitions on same-day boundaries.
                partition_start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
                partition_end = (
                    partition_start + pd.offsets.MonthBegin(1) - pd.Timedelta(nanoseconds=1)
                )

                if start is not None and partition_end < start:
                    continue
                if end is not None and partition_start >= end:
                    continue

            file_path = month_dir / "data.parquet"
            if file_path.exists():
                files.append(file_path)

    return files


def _load_partitions_parallel(
    files: list[Path],
    cols: list[str] | None,
    max_workers: int,
) -> list[pd.DataFrame]:
    """Load a list of parquet files in parallel."""
    dfs: list[pd.DataFrame] = []

    def _load(file_path: Path) -> pd.DataFrame:
        return pd.read_parquet(file_path, columns=cols)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(files))) as pool:
        futures = {pool.submit(_load, fp): fp for fp in files}
        for future in as_completed(futures):
            fp = futures[future]
            try:
                dfs.append(future.result())
            except Exception as exc:
                logger.error("Failed to load partition %s: %s", fp, exc)

    return dfs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_utc_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee the DataFrame has a UTC-aware DatetimeIndex.

    - If the index is already a DatetimeIndex, it is localized or converted to UTC.
    - If the index is numeric (Unix milliseconds), it is converted.
    - Raises if the index type is unrecognised.
    """
    idx = df.index

    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            df.index = idx.tz_localize("UTC")
        else:
            df.index = idx.tz_convert("UTC")
        return df

    if pd.api.types.is_numeric_dtype(idx):
        df.index = pd.to_datetime(idx.astype("int64"), unit="ms", utc=True)
        return df

    raise ValueError(
        f"Cannot convert index of type {type(idx).__name__} to a UTC DatetimeIndex. "
        "Expected a DatetimeIndex or a numeric (Unix millisecond) index."
    )

