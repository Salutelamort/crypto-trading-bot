# Research acceleration and frozen trial identity

The paper runner remains the reference execution engine. Real exchange orders are
not enabled by this release.

## Computation

- `evolution.compiled_backtest` enables Numba only for research runs without trade
  or order details. Detailed backtests and reconciliation retain the Python path.
  Invalid/nonfinite market arrays use the reference path. `fastmath` is disabled.
- Numba 0.63.1 and llvmlite 0.46.0 are pinned alongside the existing NumPy 2.3.5.
  First-use compilation adds startup work; later calls reuse the compiled kernel.
- The correlation filter reuses complete UTC daily returns from the same evaluated
  genome, configuration and candle contents. Its separate cache is capped at
  2,048 results / 8 MiB and persists only within a research process.
- A local Windows benchmark of 75 walk-forward evaluations on 6,000 synthetic
  candles took 1.949 seconds with the reference path and 1.044 seconds compiled
  after warm-up (1.87x). This is not a forecast for total cloud throughput.

## Search proposals

The bounded `guided_training_archive_v1` runtime-state record stores up to 512
valid parameter sets and their training Sharpe/trade counts. Families are balanced
by symbol, timeframe and strategy type. It survives research-process restarts.
At most half of proposal slots are survivor mutations; at least a quarter remain
fresh random exploration. Remaining slots use archived mutations when available.
Archive rankings are proposal hints, never current performance evidence: every
proposal runs current walk-forward validation, audits and promotion gates.
Validation results are not supplied to the archive. Reports include proposal
origin counters and correlation-cache memory/hits.

## Shared observations

One runner cycle shares successful public-data requests across accounts. Provider
identity and all request arguments are part of the key, including catch-up cursors.
Each account receives its own deep copy. Quote timestamps are preserved and each
account still enforces wall-clock freshness. Account balances, positions, execution
liquidity consumption and transactions remain independent. Scope ends even on an
exception; unsuccessful requests are not cached.

## Version boundaries and deployment

Research exchange files keep the full source fingerprint. Frozen trials and account
experiments use `versioning.trading_hash`, which fingerprints execution modules and
dependencies while excluding named research/report-only definitions. New helpers
in mixed modules are included by default. Missing manifest files fail closed.
Training settings do not change experiment identity; risk and cost settings do.

The first release changes market observation collection and introduces this version
boundary. Existing frozen trials are marked `version_changed`, with their serialized
ledgers retained. New trials start independently; old and new observations must not
be combined into one continuous trial. The main account is not reset.

There is no SQL schema migration. The new archive uses existing runtime state.
Because compiler dependencies change, the generic exact-dependency rollback guard
deliberately rejects this first rollout. It requires a reviewed dependency rollout:
upload only staged source, keep the mounted ledger, verify the new dependency and
source hashes plus fresh ticks, and inspect learning logs. Do not override the
contract to pretend the dependency sets match. Old trial identities should not be
silently reactivated on rollback.
