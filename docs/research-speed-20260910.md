# Research and monitoring changes, 2026-09-10

Baseline: ten complete Railway research runs on September 7–9 performed 50,465
evaluations; result-cache hit rate was 7.62%. Mean elapsed time was 258 seconds:
124 evaluating, 32 downloading, 16 correlating, 85 unclassified. Paper stayed at
9,923.318502577935 with no open positions or entries in the inspected logs.

## Changes

- Insert candidates and metrics in transactions of at most 32 candidates. Record
  selection and correlation rejections in batches. Paper account write paths retain
  their original transaction behavior. Research timing now separates candidate and
  selection persistence; report includes write-transaction counts.
- Reject candidates below the existing minimum training-trade count before running
  validation. The screen receives only training bars. Survivors still undergo full
  validation and audits. Rejected attempts are counted separately in reports and
  included in the multiple-testing reference count without fabricated validation
  results. Historical stored metrics remain intact.
- Reuse RSI, ADX and Supertrend results within a research process using a bounded
  32 MiB content-keyed cache. Input changes invalidate reuse; returned values are
  copied. Do not cache cheap moving averages: a benchmark showed hashing overhead
  outweighed savings. Paper execution does not activate the cache.
- Forward diagnostics explain all admission failures per candidate and show the
  latest recorded signals. Missing signal observations are explicitly unknown;
  they are not classified as zero signals. Last-nonzero lookup is limited to 1,000
  recorded bars. Protected endpoint: `/api/forward-diagnostics`; cloud logs:
  `FORWARD_DIAGNOSTICS`. No admission thresholds were lowered.
- Cloudflare records HTTP status and failure category. It avoids stale cached
  responses and follows at most three same-origin HTTPS redirects; foreign origins
  and HTTPS downgrades are rejected. Diagnostic redirect targets are stored only in
  private KV. GitHub dispatch still requires its narrowly scoped Worker secret.

## Validation

Local component benchmark (not a production speed guarantee): 512 candidate/metric
inserts took 3.859s with per-call commits and 0.108s in batches of 32. A fixed set of
100 synthetic candidates took 1.442s without training screening and 1.337s with it;
18 were rejected. Indicator-cache benefit was small on the synthetic mixed set
(approximately 1–4%); actual usefulness must be checked in server reports.

Tests cover exact survivor-metric parity, no validation access during rejection,
cache mutation isolation and invalidation, transaction rollback including trial
statistics, and forward admission reasons. Existing golden backtests remain the
acceptance check for execution math.

Reference approaches: [Freqtrade indicator precomputation](https://docs.freqtrade.io/en/stable/hyperopt/),
[SQLite batched transactions](https://www.sqlite.org/faq.html),
[Optuna early stopping](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.SuccessiveHalvingPruner.html).
The implemented screen uses the existing deterministic eligibility condition,
not Optuna's relative-performance pruning.

Offsite backup work is paused at the user's request. No new storage is provisioned.
