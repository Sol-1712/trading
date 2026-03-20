from pathlib import Path
import pandas as pd
from datetime import datetime


def save_partitioned_parquet(
    df: pd.DataFrame,
    dataset_path: str | Path,
    mode: str = "merge",
) -> None:
    """
    Save DataFrame into an existing dataset root path
    partitioned by year/month.
    """

    dataset_path = Path(dataset_path)

    if df.empty:
        print("DataFrame is empty, nothing to save.")
        return

    data = df.copy()

    # Ensure UTC datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, unit="ms", utc=True)
    else:
        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC")
        else:
            data.index = data.index.tz_convert("UTC")

    data = data.sort_index()

    data["year"] = data.index.year
    data["month"] = data.index.month

    for (year, month), group in data.groupby(["year", "month"]):

        group = group.drop(columns=["year", "month"])

        partition_dir = dataset_path / f"year={year}" / f"month={month:02d}"
        partition_dir.mkdir(parents=True, exist_ok=True)

        file_path = partition_dir / "data.parquet"

        if mode == "merge" and file_path.exists():
            existing = pd.read_parquet(file_path)

            combined = pd.concat([existing, group])
            combined = (
                combined[~combined.index.duplicated(keep="last")]
                .sort_index()
            )

            combined.to_parquet(file_path)
            print(f"Updated {file_path} ({len(combined)} rows)")

        else:
            group.to_parquet(file_path)
            print(f"Wrote {file_path} ({len(group)} rows)")


def load_partitioned_parquet(
    path: str | Path,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load partitioned parquet dataset with optional date filtering
    and column selection.

    Parameters
    ----------
    base_path : root data directory
    start : start datetime (inclusive)
    end : end datetime (exclusive)
    columns : subset of columns to load

    Returns
    -------
    pd.DataFrame
    """


    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Normalize start/end
    if start is not None:
        start = pd.to_datetime(start, utc=True)
    if end is not None:
        end = pd.to_datetime(end, utc=True)

    dfs = []

    # Iterate year/month partitions
    for year_dir in sorted(path.glob("year=*")):
        year = int(year_dir.name.split("=")[1])

        for month_dir in sorted(year_dir.glob("month=*")):
            month = int(month_dir.name.split("=")[1])

            # Skip partitions outside date range
            if start or end:
                partition_start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
                partition_end = (
                    partition_start + pd.offsets.MonthEnd(0)
                )

                if start and partition_end < start:
                    continue
                if end and partition_start > end:
                    continue

            file_path = month_dir / "data.parquet"
            if not file_path.exists():
                continue

            df = pd.read_parquet(file_path, columns=columns)
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs).sort_index()
    
    if not isinstance(result.index, pd.DatetimeIndex):
        if "timestamp" in result.columns:
            result.index = pd.to_datetime(result["timestamp"], utc=True)
        else:
            raise ValueError("DataFrame has no DatetimeIndex or 'timestamp' column")
        
        # Final precise filtering
    if start:
        result = result[result.index >= start]
    if end:
        result = result[result.index < end]

    return result