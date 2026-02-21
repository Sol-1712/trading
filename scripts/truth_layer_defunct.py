import numpy as np
import pandas as pd
from utils.store_defunct import make_file_path, load_parquet_partitioned


### Return at t is (Ret[t] / Ret[t-1] - 1 )* held_pos
### Pos is the position at bar close, i.e taken exactly after 00:00:00
### Position is expressed as fraction of capital allocated to the asset, not # contracts
### held_pos[t]=pos[t−1]
## prices → signal → desired position → execution delay → held position → returns → PnL



# Fees (%)
# Can pull from api when it matters
MAKER_FEE = 0.000200
TAKER_FEE = 0.000550

FUNDING_INTERVAL = 480 # 8 hours in minutes

# PARAMS
SYMBOL = 'BTCUSDT'
INTERVAL = 60
START = '01/01/2026'
END = None





def main():

    # Loads ohlcv and funding data
    data_type = 'ohlcv'
    path_ohlcv = make_file_path(data_type, SYMBOL, INTERVAL)
    df_ohlcv = load_parquet_partitioned(path_ohlcv, start=START, end=END)

    data_type = 'funding'
    path_funding = make_file_path(data_type, SYMBOL)
    df_funding = load_parquet_partitioned(path_funding,start=START, end=END)


    df_merge = df_ohlcv.merge(df_funding, how='left', on='timestamp')
    df_merge['fundingRate'] = df_merge['fundingRate'].fillna(0)
    df_merge = df_merge.sort_index()


    mark_close = df_merge["mark_close"].astype("float64").to_numpy()
    funding  = df_merge["fundingRate"].astype("float64").to_numpy()

    # I want to pass it these two columns only.
    # ------------------------------------------------------------------------

    ret = np.zeros_like(mark_close)
    ret[1:] = mark_close[1:] / mark_close[:-1] - 1.0

    ### BASIC STRATEGY
    pos = np.zeros_like(mark_close)
    pos[:] = 1 # Always long (enter at t=0)

    held_pos = np.roll(pos, 1)
    held_pos[0] = 0

    trade = pos - held_pos        
    fee_rate = TAKER_FEE           
    fees = np.abs(trade) * fee_rate

    funding_pnl = - held_pos * funding

    strategy_ret = held_pos * ret + funding_pnl - fees
    strategy_ret[0] = 0

    equity = np.cumprod(1.0 + strategy_ret)

    out = pd.DataFrame(
        {
            "strategy_ret": strategy_ret,
            "equity": equity,
            "held_pos": held_pos,
            "trade": trade,
            "fees": fees,
            "funding_pnl": funding_pnl,
        },
        index=df_merge.index,
    )
    print(out.head(10))
    print(equity)
    return out

if __name__ == "__main__":
    df = main()

