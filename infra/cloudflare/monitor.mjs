const required = ['process', 'paper_mode', 'heartbeat', 'market_data', 'ledger', 'research', 'backup'];

export function verify(value, now = Date.now()) {
  const age = now - Date.parse(value?.checked_at);
  return value?.schema_version === 1 && value.ok === true
    && required.every(key => value.checks?.[key] === true)
    && Number.isFinite(age) && age >= -30000 && age <= 120000
    && typeof value.heartbeat_age_seconds === 'number'
    && value.heartbeat_age_seconds >= 0 && value.heartbeat_age_seconds <= 180;
}

export async function observe(env, fetcher = fetch, now = Date.now()) {
  let ok = false;
  try {
    const response = await fetcher(env.BOT_MONITOR_URL, {
      headers: { 'Cache-Control': 'no-cache' }, redirect: 'error',
      signal: AbortSignal.timeout(15000),
    });
    if (response.status === 200) {
      const reader = response.body.getReader();
      const chunks = []; let size = 0;
      while (true) {
        const part = await reader.read();
        if (part.done) break;
        size += part.value.length;
        if (size > 65536) { await reader.cancel(); throw new Error('oversize'); }
        chunks.push(part.value);
      }
      const bytes = new Uint8Array(size); let offset = 0;
      for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
      ok = verify(JSON.parse(new TextDecoder().decode(bytes)), now);
    }
  } catch { /* Store failure without response bodies, URLs or credentials. */ }
  const previous = await env.STATE.get('latest', 'json');
  const failures = ok ? 0 : (previous?.failures ?? 0) + 1;
  let alertedAt = previous?.alerted_at ?? 0;
  let notification = env.GITHUB_DISPATCH_TOKEN ? 'idle' : 'not_configured';
  // Two failed scheduled observations reduce alerts during short redeploys.
  if (env.GITHUB_DISPATCH_TOKEN && failures >= 2 && now - alertedAt >= 3600000) {
    try {
      const response = await fetcher(
        'https://api.github.com/repos/Salutelamort/crypto-trading-bot/actions/workflows/monitor.yml/dispatches', {
          method: 'POST', redirect: 'error', signal: AbortSignal.timeout(15000),
          headers: { Authorization: `Bearer ${env.GITHUB_DISPATCH_TOKEN}`, Accept: 'application/vnd.github+json',
            'User-Agent': 'paper-bot-observer', 'Content-Type': 'application/json' },
          body: JSON.stringify({ ref: 'main', inputs: { external_incident: 'true' } }),
        });
      if (response.status !== 204) throw new Error('dispatch_failed');
      alertedAt = now; notification = 'dispatched';
    } catch { notification = 'failed'; }
  }
  const value = { schema_version: 1, ok, checked_at: new Date(now).toISOString(), failures,
    alerted_at: alertedAt, notification };
  await env.STATE.put('latest', JSON.stringify(value));
  return value;
}

export default {
  async scheduled(_controller, env) { await observe(env); },
  async fetch(request, env) {
    if (request.method !== 'GET' || new URL(request.url).pathname !== '/status') {
      return new Response('Not found', { status: 404 });
    }
    const latest = await env.STATE.get('latest', 'json');
    const age = Date.now() - Date.parse(latest?.checked_at);
    const fresh = Number.isFinite(age) && age >= -30000 && age <= 720000;
    return Response.json({ ...latest, observer_fresh: fresh }, {
      status: latest?.ok && fresh ? 200 : 503, headers: { 'Cache-Control': 'no-store' },
    });
  },
};
