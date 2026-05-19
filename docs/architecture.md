# Backtesting Architecture

## Data Layer
Responsible for:
- historical market data
- parquet storage
- incremental updates

## Signal Layer
Responsible for:
- generating directional signals
- no portfolio logic
- no execution assumptions

## Portfolio Layer
Responsible for:
- position sizing
- leverage
- portfolio state
- exposure management

## Execution Layer
Responsible for:
- fills
- slippage
- fees
- trade timing assumptions

## Metrics Layer
Responsible for:
- PnL
- returns
- drawdowns
- Sharpe
- analytics