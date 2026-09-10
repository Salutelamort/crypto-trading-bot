import test from 'node:test';
import assert from 'node:assert/strict';
import worker, { verify, observe } from './monitor.mjs';

const now = Date.now();
const healthy = { schema_version: 1, ok: true, checked_at: new Date(now).toISOString(), heartbeat_age_seconds: 1,
  checks: Object.fromEntries(['process', 'paper_mode', 'heartbeat', 'market_data', 'ledger', 'research', 'backup'].map(k => [k, true])) };
function environment() {
  const values = new Map();
  return { BOT_MONITOR_URL: 'https://example.test/monitor', STATE: {
    async get(key = 'latest') { return values.get(key) ?? null; },
    async put(key, text) { values.set(key, JSON.parse(text)); },
  } };
}
test('rejects stale and incomplete operational evidence', () => {
  assert.equal(verify(healthy, now), true);
  assert.equal(verify(healthy, now + 121000), false);
  assert.equal(verify({ ...healthy, heartbeat_age_seconds: true }, now), false);
  assert.equal(verify({ ...healthy, checks: {} }, now), false);
});
test('persists health without storing financial response fields', async () => {
  const env = environment();
  await observe(env, async () => Response.json({ ...healthy, balance: 100 }), now);
  const value = await env.STATE.get();
  assert.equal(value.ok, true); assert.equal(value.notification, 'not_configured');
  assert.equal('balance' in value, false);
});
test('alerts only after repeated failure and throttles successful dispatch', async () => {
  const env = { ...environment(), GITHUB_DISPATCH_TOKEN: 'test-only' }; let dispatches = 0;
  const fetcher = async (url) => {
    if (url.includes('api.github.com')) { dispatches++; return new Response(null, { status: 204 }); }
    throw new Error('offline');
  };
  await observe(env, fetcher, now);
  assert.equal(dispatches, 0);
  await observe(env, fetcher, now + 300000);
  assert.equal(dispatches, 1);
  await observe(env, fetcher, now + 600000);
  assert.equal(dispatches, 1);
});
test('missing cron observation and public mutation requests fail closed', async () => {
  const env = environment();
  assert.equal((await worker.fetch(new Request('https://example.test/status'), env)).status, 503);
  assert.equal((await worker.fetch(new Request('https://example.test/status', { method: 'POST' }), env)).status, 404);
});
test('failed notification is recorded and retried without losing incident', async () => {
  const env = { ...environment(), GITHUB_DISPATCH_TOKEN: 'test-only' };
  const fetcher = async () => new Response(null, { status: 503 });
  await observe(env, fetcher, now);
  const failed = await observe(env, fetcher, now + 300000);
  assert.equal(failed.notification, 'failed');
  assert.equal(failed.alerted_at, 0);
  assert.equal(failed.failures, 2);
});
test('oversized response is rejected and stale stored success returns 503', async () => {
  const env = environment();
  const result = await observe(env, async () => new Response(' '.repeat(65537)), now);
  assert.equal(result.ok, false);
  await env.STATE.put('latest', JSON.stringify({ ok: true, checked_at: new Date(now - 900000).toISOString() }));
  assert.equal((await worker.fetch(new Request('https://example.test/status'), env)).status, 503);
});
test('follows same-origin HTTPS redirects but rejects foreign destinations', async () => {
  const env = environment(); let calls = 0;
  const fetcher = async () => ++calls === 1
    ? new Response(null, { status: 302, headers: { Location: '/fresh-monitor' } })
    : Response.json(healthy);
  assert.equal((await observe(env, fetcher, now)).ok, true);
  assert.equal(calls, 2);
  const rejected = await observe(env, async () => new Response(null,
    { status: 302, headers: { Location: 'https://unrelated.test' } }), now);
  assert.equal(rejected.ok, false);
  assert.equal(rejected.reason, 'redirect_rejected');
});
