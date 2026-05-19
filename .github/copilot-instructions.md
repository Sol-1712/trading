This project is a quantitative trading research and backtesting system focused on directional trading strategies and crypto perpetual futures.

The system is designed for:
- strategy research
- backtesting
- portfolio/risk modelling
- execution simulation
- data integrity and reproducibility

The architecture should remain modular and explicit.

Core layers:
1. data ingestion
2. signal generation
3. portfolio/risk management
4. execution simulation
5. performance analytics

Primary priorities:
1. correctness
2. deterministic behaviour
3. prevention of lookahead bias
4. timestamp integrity
5. modular architecture
6. realistic execution assumptions
7. maintainability

Important constraints:
- Never introduce lookahead bias
- Never use future information implicitly
- Preserve timestamp ordering strictly
- Avoid hidden state or implicit mutation
- Prefer explicit data flow
- Keep strategy logic separate from execution logic
- Keep execution logic separate from portfolio/risk logic
- Preserve reproducibility of backtests
- Prefer small safe edits over large refactors
- Explain assumptions before modifying architecture

Backtesting assumptions:
- Signals are generated only from information available at that timestamp
- Position sizing and leverage handling must be explicit
- Fees and slippage should be handled consistently
- Metrics should not rely on future bars
- Execution timing assumptions must be clearly stated

When modifying code:
- trace the full data flow first
- identify risks before editing
- explain architectural implications
- prefer minimal changes
- add validation/logging where useful
- maintain modular boundaries

When debugging:
- inspect timestamps carefully
- inspect indexing alignment carefully
- check for hidden leakage
- validate portfolio state transitions
- validate position sizing calculations