const required = ['process', 'paper_mode', 'heartbeat', 'market_data', 'ledger', 'research', 'backup'];

export function verify(value, now = Date.now()) {
  const age = now - Date.parse(value?.checked_at);
  return value?.schema_version === 1 && value.ok === true
    && required.every(key => value.checks?.[key] === true)
    && Number.isFinite(age) && age >= -30000 && age <= 120000
    && typeof value.heartbeat_age_seconds === 'number'
    && value.heartbeat_age_seconds >= 0 && value.heartbeat_age_seconds <= 180;
}

export async function observe(env, fetcher = (...args) => globalThis.fetch(...args), now = Date.now()) {
  let ok = false;
  let reason = 'http_error'; let httpStatus = null;
  try {
    const url = new URL(env.BOT_MONITOR_URL);
    url.searchParams.set('observation', String(now));
    let target = url;
    let response;
    const signal = AbortSignal.timeout(15000);
    for (let hop = 0; hop <= 3; hop++) {
      response = await fetcher(target.toString(), {
        headers: { 'Cache-Control': 'no-cache' }, redirect: 'manual', signal,
        cf: { cacheTtl: 0, cacheEverything: false },
      });
      if (![301, 302, 303, 307, 308].includes(response.status)) break;
      const location = response.headers.get('Location');
      const next = location ? new URL(location, target) : null;
      await env.STATE.put('redirect_diagnostic', JSON.stringify({ status: response.status,
        hostname: next?.hostname, pathname: next?.pathname }));
      if (!next || next.protocol !== 'https:' || next.origin !== url.origin || hop === 3) {
        throw new Error('redirect rejected');
      }
      await response.body?.cancel();
      target = next;
    }
    httpStatus = response.status;
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
      reason = ok ? 'healthy' : 'invalid_or_unhealthy_evidence';
    }
  } catch (error) {
    reason = ['TimeoutError', 'AbortError', 'SyntaxError', 'TypeError'].includes(error?.name)
      ? error.name : 'request_failed';
    if (/illegal invocation|incorrect this/i.test(error?.message ?? '')) reason = 'invalid_fetch_binding';
    if (/redirect/i.test(error?.message ?? '')) reason = 'redirect_rejected';
    await env.STATE.put('request_diagnostic', JSON.stringify({ at: new Date(now).toISOString(),
      reason, detail: String(error?.message ?? '').replace(/https?:\/\/\S+/g, '[url]').slice(0, 200) }));
  }
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
    alerted_at: alertedAt, notification, reason, http_status: httpStatus };
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
