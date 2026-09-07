# Independent paper-bot observer

Cloudflare Cron executes `monitor.mjs` every five minutes. It checks the Railway
operational endpoint and stores only health, observation time and alert status in
KV. `/status` reads stored evidence; it never triggers checks. Missing or more than
12-minute-old evidence returns HTTP 503. No trading keys or LLM are used.

Deployed status: https://trading-paper-observer.cirillogallo167.workers.dev/status

## GitHub notifications

Existing GitHub scheduled checks continue separately. To let Cloudflare trigger
GitHub failure notifications even when GitHub's scheduled workflow is disabled,
configure the Worker secret `GITHUB_DISPATCH_TOKEN`:

1. Create a GitHub fine-grained personal access token restricted to
   `Salutelamort/crypto-trading-bot` with **Actions: read and write**. Choose an
   expiry and arrange renewal; do not use the existing broad CLI token.
2. Run `wrangler secret put GITHUB_DISPATCH_TOKEN --config infra/cloudflare/wrangler.jsonc`
   and paste into its prompt. Do not paste the token into chat or repository files.
3. Confirm an intentional test workflow dispatch and notification delivery.

After two failed observations, the Worker dispatches the existing workflow with
`external_incident=true`. The job fails even if the bot has recovered by then so
the incident is not hidden. Successful dispatches are limited to one per hour.
Delivery still depends on GitHub availability, token validity and user notification
settings. `notification=not_configured` explicitly identifies a missing token.

## Validation and deployment

`node --test infra/cloudflare/monitor.test.mjs`

`wrangler deploy --config infra/cloudflare/wrangler.jsonc`

Cron changes can take up to 15 minutes to propagate. Confirm that `/status` shows
an advancing `checked_at`; a successful upload alone does not prove cron execution.

## Backups

R2 was not enabled in the account when checked on 2026-09-07 (API error 10042).
No offsite backups exist yet. Enable R2 in the Cloudflare dashboard before creating
the private bucket and implementing upload/restore validation. Railway's local
volume backups remain the only automated backups currently connected.
