import pandas as pd

from data_utils.paths import make_data_path
from data_utils.io import load_partitioned_parquet
from backtester.config import Config

### Crrently only works for raw ohlcv data, fix when i fix the rest


def _load_backtest_data(config: Config) -> pd.DataFrame:
    """
    Load and merge OHLCV and funding rate data for a given symbol and interval.

    Loads OHLCV data with the specified columns, merges with funding rate data
    on timestamp, and fills missing funding rates with zero.

    Parameters
    ----------
    config : Config
        Backtest configuration containing symbol, interval, start, end.
    cols : list[str]
        OHLCV columns to load e.g. ['mark_close', 'volume'].

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with OHLCV and funding rate columns, sorted by index.

    Raises
    ------
    ValueError
        If either OHLCV or funding data loads empty for the given parameters.
    """
    path_main = make_data_path("ohlcv", config.symbol, config.interval)

    df_main = load_partitioned_parquet(
        path_main,
        start=config.start,
        end=config.end
    )
    if df_main.empty:
        raise ValueError("df_ohlcv loaded empty. Check path or date filters.")

    path_funding = make_data_path('funding', config.symbol)
    df_funding = load_partitioned_parquet(path_funding, start=config.start, end=config.end)

    if df_funding.empty:
        raise ValueError("df_funding loaded empty. Check path or date filters.")

    df_merge = df_main.merge(df_funding, how='left', on='timestamp')
    df_merge['fundingRate'] = df_merge['fundingRate'].fillna(0)
    df_merge = df_merge.sort_index()

    return df_merge


def prepare_data(config: Config) -> pd.DataFrame:
    """
    Prepare market data for backtesting, including all columns required by the signal generator.

    Combines any signal-specific columns with the price column defined in config,
    deduplicates, and delegates to the data loader.

    Parameters
    ----------
    config : Config
        Backtest configuration containing symbol, interval, start, end, price_column.
    Currently loads all columns. 

    Returns
    -------
    pd.DataFrame
        Merged OHLCV and funding DataFrame with all required columns.

    """

    return _load_backtest_data(config)







