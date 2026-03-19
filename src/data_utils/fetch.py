import pandas as pd


def fetch_time_series(
        call_fn, 
        time_col: str, 
        start: int, 
        end: int 
    ):

    """
    Fetch a time-series dataset by repeatedly calling an API wrapper function.

    Parameters
    ----------
    call_fn : callable
        A function with signature ``call_fn(start, end)`` that returns a
        response from the API (such as Bybit). Example:

            def call_fn(start, end):
                return session.get_funding_rate_history(
                    category=category,
                    symbol=symbol,
                    startTime=start,
                    endTime=end,
                    limit=200,
                )

    time_col : str
        Name of the timestamp field in the returned data (e.g. ``"fundingRateTimestamp"``).
    start : int
        Starting Unix timestamp (milliseconds since epoch).
    end : int
        Ending Unix timestamp (milliseconds since epoch).

    Returns
    -------
    pandas.DataFrame
        Concatenated time-series data from all API calls.
    """


    data_frames = []

    while end > start:
        df_page = call_fn(start, end)

        if df_page is None or df_page.empty:
            break

        df_page[time_col] = df_page[time_col].astype("int64")
        data_frames.append(df_page)

        # Move end backwards
        earliest = df_page[time_col].min()
        end = earliest - 1

    if not data_frames:
        return pd.DataFrame()

    out = pd.concat(data_frames, ignore_index=True)
    out = out.sort_values(time_col)
    return out

# Function that fetchs last price ohlcv data for given instrument
def fetch_last_ohlcv(
        session, 
        symbol: str, 
        interval: int, 
        start: int, 
        end_init: int | None = None, 
        category: str = 'linear', 
        limit: int = 1000
    ):

    """
    Fetch ohlcv data for given paramters.

    Parameters
    ----------
    session : HTTP
        Bybit api session object
    symbol : str
        Instrument symbol e.g 'BTCUSDT'.
    interval : int
        Bar interval in minutes.
    start : int
        Starting Unix timestamp (milliseconds since epoch).
    end_init : int (optional)
        Ending Unix timestamp (milliseconds since epoch).
        Default is current servertime.
    category : str (optional)
        Instrument category. Default is linear.
    limit : int (optional)
        Number of entries per page. Default is 1000.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing all requested ohlcv data.
    """


    # Set end timestamp to current time if none
    if end_init == None:
        end = session.get_server_time()
        end = end['time']
    else:
        end = end_init        
    cols = ["start_time", 'last_open', 'last_high', 'last_low',
             'last_close', 'volume', 'turnover']

    # Call function
    def call_fn(start, end):
        response = session.get_kline(
            category=category,
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            limit=limit
        )
        data = response['result']['list']
        if not data:
            return pd.DataFrame(columns=cols)

        return pd.DataFrame(data, columns=cols)

    df = fetch_time_series(
        call_fn=call_fn,
        time_col='start_time',
        start=start,
        end=end,   
    )

    if df.empty:
        return df

    df.rename(columns={'start_time': 'timestamp'}, inplace=True)
    df.set_index('timestamp', inplace=True)
    num_cols = ['last_open', 'last_high', 'last_low',
             'last_close', 'volume', 'turnover']
    df[num_cols] = df[num_cols].astype("float64")

    return df

# Function that fetchs price ohlc data for given instrument
def fetch_mark_ohlc(
        session, 
        symbol: str, 
        interval: int, 
        start: int, 
        end_init: int | None = None, 
        category: str = 'linear', 
        limit: int = 1000
    ):
    """
    Fetch mark price ohlc data for given paramters.

    Parameters
    ----------
    session : HTTP
        Bybit api session object
    symbol : str
        Instrument symbol e.g 'BTCUSDT'.
    interval : int
        Bar interval in minutes.
    start : int
        Starting Unix timestamp (milliseconds since epoch).
    end_init : int (optional)
        Ending Unix timestamp (milliseconds since epoch).
        Default is current servertime.
    category : str (optional)
        Instrument category. Default is linear.
    limit : int (optional)
        Number of entries per page. Default is 1000.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing all requested ohlc data.
    """


    # Set end timestamp to current time if none
    if end_init == None:
        end = session.get_server_time()
        end = end['time']
    else:
        end = end_init        
    cols = ["start_time", 'mark_open', 'mark_high', 'mark_low', 'mark_close']

    # Call function
    def call_fn(start, end):
        response = session.get_mark_price_kline(
            category=category,
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            limit=limit
        )
        data = response['result']['list']
        if not data:
            return pd.DataFrame(columns=cols)

        return pd.DataFrame(data, columns=cols)

    df = fetch_time_series(
        call_fn=call_fn,
        time_col='start_time',
        start=start,
        end=end,   
    )

    if df.empty:
        return df

    df.rename(columns={'start_time': 'timestamp'}, inplace=True)
    df.set_index('timestamp', inplace=True)
    num_cols = ['mark_open', 'mark_high', 'mark_low', 'mark_close']
    df[num_cols] = df[num_cols].ffill()
    df[num_cols] = df[num_cols].astype("float64")

    return df



# Function to fetch funding rate data from bybit api 
def fetch_funding_rate(
        session, 
        symbol: str, 
        start: int, 
        end_init: int | None = None, 
        limit: int = 200, 
        category: str = 'linear'
    ):

    """
    Fetches funding rate data from bybit api.

    Parameters
    ----------
    session : HTTP
        Bybit api session object
    symbol : str
        Instrument symbol e.g 'BTCUSDT'.
    start : int
        Starting Unix timestamp (milliseconds since epoch).
    end_init : int (optional)
        Ending Unix timestamp (milliseconds since epoch).
        Default is current servertime.
    category : str (optional)
        Instrument category. Default is linear.
    limit : int (optional)
        Number of entries per page. Default is 200.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing all requested funding rate data.
    """

    
    cols = ['fundingRate', 'fundingRateTimestamp']

    # Set end timestamp to current time if none
    if end_init == None:
        end = session.get_server_time()
        end = end['time']
    else:
        end = end_init

    def call_fn(start, end):
        response = (session.get_funding_rate_history(
            category=category, 
            symbol=symbol,
            startTime = start,
            endTime = end,
            limit = limit)
            )     
         
        data = response['result']['list']
        if not data:
            return pd.DataFrame(columns=cols)

        return pd.DataFrame(data, columns=cols)     

    df = fetch_time_series(call_fn,
                           'fundingRateTimestamp',
                           start,
                           end
                           ) 
    
    if df.empty:
        return df
    
    df.rename(columns={'fundingRateTimestamp': 'timestamp'}, inplace=True)
    df.set_index('timestamp', inplace=True)
    df["fundingRate"] = df["fundingRate"].astype("float64")

    return df
