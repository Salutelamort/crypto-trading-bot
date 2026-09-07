# Reliability upgrades, 2026-09-07

## Available in this release

- Research shares a content-addressed evaluation cache across cycles and promoted-agent reevaluation in one process. Limits: 512 results and approximately 32 MiB of retained result data (not total process RAM). Changed candles, genome or configuration invalidate reuse. Cached objects are copied on read/write.
- Reconciliation reports calculate a nonnegative 95th-percentile additional cost-stress suggestion from **all** matched orders, before truncating the visible detail list. Requires 30 orders across 7 distinct days, no data gaps and no signal mismatches. This measures candle-to-paper differences, not real exchange execution. It does not change trading costs or admission gates automatically.
- `/monitor` exposes deployment identity and a compatibility contract (schema, state protocol, dependency and source fingerprints), without balances or credentials.
- `scripts/safe_deploy.py` guards reviewed uploads: checks compatibility with the current healthy release, observes three healthy checks on the actual new deployment, and requests Railway rollback after failure. An incompatible migration requires separate review. Rollback does not replace the volume or restore an old ledger. Railway also restores the previous release's custom variables.

## Deployment procedure

Generate `release-contract.json` in a curated upload directory with `src.release_contract.describe(upload_dir, reviewed_database_copy)`; never include that database in the upload. The database copy must represent the intended schema after migrations. Pass the directory to `scripts/safe_deploy.py` with project/environment/service, monitor URL, Railway CLI path and the existing operator token in its environment. The token is not uploaded.

The first release exposing a contract needs a manually reviewed deployment. Automated rollback is tested with mocked API failure/recovery, not by crashing production. API unavailability, expired image retention or an intervening deployment can still require operator intervention. This script only supervises deployments started through it; it is not a permanent rollback daemon.

## Still pending

- Cloudflare authentication succeeded. The independent Worker scheduler has been deployed; configuration and notification setup are in `infra/cloudflare/README.md`.
- R2 is not activated (API error 10042). Offsite backup upload and independent restore validation remain unconnected. Current backups remain on the Railway volume; GitHub scheduled monitoring remains active with its existing scheduling limitations.
- Cloudflare-triggered GitHub notifications need a narrowly scoped credential or GitHub App integration. Do not copy a broad personal GitHub token into a Worker.
- Cost-stress evidence currently supplies a report only. Applying it to future research requires a separate versioned calibration policy; it must never silently lower costs or weaken validation.

References: [Railway rollback API](https://docs.railway.com/integrations/api/manage-deployments), [rollback behavior](https://docs.railway.com/guides/roll-back-bad-deploy).
