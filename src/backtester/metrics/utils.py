import pandas as pd

SECONDS_TO_PERIODS_24_7: dict[int, tuple[str, int]] = {
    60:     ("1min",  365 * 24 * 60),
    300:    ("5min",  365 * 24 * 12),
    900:    ("15min", 365 * 24 * 4),
    3600:   ("1H",    365 * 24),      
    86400:  ("1D",    365),
    604800: ("1W",    52),
}

# Helper Function
def infer_ann_factor(
    dtindex: pd.DatetimeIndex
    ) -> tuple[str, float]:

    """
    Infers the annualisation factor based of timestamp index

    Returns:
    freq: bar interval frequency (defualt 1H)
    periods: number of intervals per year
    """
    median_diff = pd.Series(dtindex).diff().dt.total_seconds().median()
    freq, periods = SECONDS_TO_PERIODS_24_7.get(int(median_diff), ("1H", 365 * 24))
    return freq, float(periods)
