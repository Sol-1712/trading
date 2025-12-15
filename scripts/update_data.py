from datetime import datetime, timezone
from pathlib import Path
from pybit.unified_trading import HTTP
import key

from data.fetch import fetch_funding_rate, fetch_ohlcv
from data.store import make_file_path, load_parquet, save_partitioned_parquet


SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVALS = [1, 5, 15]
START_FALLBACK = 1609459200000 # 01/01/20021 00:00:00


def get_last_timestamp(
        path: Path
        ) -> int | None:
    
    """
    Returns last timestamp (unix ms) in existing parquet files,
    or None if no data exists.
    """
    files = sorted(path.rglob("*.parquet"))
    if not files:
        return None

    df = load_parquet(path, index_as="unix_ms")
    return int(df.index.max())


def update_ohlcv(
        session,
        symbol: str,
        interval: int,
        start_fallback: int = START_FALLBACK
    ) -> None:

    """
    Updates ohlcv data

    Parameters
    ----------
    session : HTTP
        Bybit API session
    symbol : str
        Instrument symbol, e.g. "BTCUSDT".
    interval : int
        Bar interval in minutes 
    start_fallback: int (optional)
        Start time fallback if data cannot be found
    """

    path = make_file_path('ohlcv', symbol, interval)

    last_ts = get_last_timestamp(path)
    start = last_ts + 1 if last_ts is not None else start_fallback # No files found

    print(f"Updating OHLCV {symbol} {interval}m from {start}")
    df = fetch_ohlcv(
        session=session,
        symbol=symbol,
        interval=interval,
        start=start,
    )

    if df.empty:
        print("No new OHLCV data.")
        return
    

    save_partitioned_parquet(
        df=df, 
        base_path=path,
        data_type='ohlcv',
        symbol=symbol,
        interval=interval
        )
    

def update_funding(
    session,
    symbol: str,
    start_fallback: int = START_FALLBACK,
    ):
    
    """
    Updates funding rate data

    Parameters
    ----------
    session : HTTP
        Bybit API session
    symbol : str
        Instrument symbol, e.g. "BTCUSDT".
    start_fallback: int (optional)
        Start time fallback if data cannot be found
    """

    path = make_file_path("funding", symbol)

    last_ts = get_last_timestamp(path)
    start = last_ts + 1 if last_ts is not None else start_fallback

    print(f"Updating funding {symbol} from {start}")

    df = fetch_funding_rate(
        session=session,
        symbol=symbol,
        start=start,
    )

    if df.empty:
        print("No new funding data.")
        return

    save_partitioned_parquet(
        df=df,
        base_path=path,
        data_type="funding",
        symbol=symbol,
        mode="merge",
    )



def main():
    session = HTTP(api_key=key.BYBIT_API_KEY, api_secret=key.BYBIT_API_SECRET)

    for symbol in SYMBOLS:
        for interval in INTERVALS:
            update_ohlcv(session, symbol, interval)

        update_funding(session, symbol)


if __name__ == "__main__":

    main()