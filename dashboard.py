"""
Веб-дашборд торгового бота. Принцип из треda: трейдеры строят локальный
дашборд, чтобы визуально проверять сделки и решения (veritas7411, Flying8ball).

Запуск:
    python dashboard.py
Затем открой в браузере:  http://127.0.0.1:5000

Возможности:
- Кнопки запуска: Fetch / Evolve / Supervise / Paper / Run All — прямо из браузера.
- Живой лог выполнения.
- Таблицы агентов (продвинутые / кандидаты / убитые), решения супервизора, сделки.
- Макро-статус (потоки в ETF, risk-on/off) с кэшем.
Всё локально, никаких ключей. Торговля остаётся детерминированной.
"""
import io
import json
import time
import threading
import contextlib

import yaml
from flask import Flask, jsonify, render_template_string, abort

from src import db
from src import data_feed as feed
from src import evolution, supervisor, paper_trade, macro_feed, live_trade

app = Flask(__name__)
CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))

# --- состояние фоновой задачи ---
TASK = {"running": False, "name": None, "log": [], "finished_at": None}
_LOCK = threading.Lock()

# --- кэш макро (farside дёргать раз в 5 мин, не на каждый рефреш) ---
_MACRO = {"ts": 0, "data": None}

# --- состояние живой торговли (отдельный фоновый цикл) ---
LIVE = {"running": False, "stop": False, "log": []}


class _LiveLog(io.TextIOBase):
    def write(self, s):
        if s.strip():
            LIVE["log"].append(s.rstrip("\n"))
            LIVE["log"][:] = LIVE["log"][-100:]
        return len(s)


def _live_loop():
    conn = db.connect(CFG["db_path"])
    interval = CFG.get("live", {}).get("interval_seconds", 300)
    writer = _LiveLog()
    while not LIVE["stop"]:
        try:
            with contextlib.redirect_stdout(writer):
                live_trade.tick(conn, CFG)
        except Exception as e:  # noqa
            LIVE["log"].append(f"[ошибка] {type(e).__name__}: {e}")
        slept = 0
        while slept < interval and not LIVE["stop"]:
            time.sleep(2)
            slept += 2
    conn.close()
    LIVE["running"] = False
    LIVE["log"].append("⏹ Живая торговля остановлена.")


class _LogWriter(io.TextIOBase):
    """Перехватывает print из модулей в лог задачи."""
    def write(self, s):
        if s.strip():
            TASK["log"].append(s.rstrip("\n"))
            TASK["log"][:] = TASK["log"][-300:]
        return len(s)


def _load_data(conn):
    data = {}
    for sym in CFG["symbols"]:
        df = feed.load_ohlcv(conn, sym, CFG["timeframe"])
        if not df.empty:
            data[sym] = df
    return data


def _run_task(name):
    """Выполняет команду в фоне со свежим соединением SQLite."""
    conn = db.connect(CFG["db_path"])
    writer = _LogWriter()
    try:
        with contextlib.redirect_stdout(writer):
            if name == "fetch":
                for sym in CFG["symbols"]:
                    feed.fetch_ohlcv(conn, sym, CFG["timeframe"], CFG["history_days"])
            elif name == "evolve":
                evolution.evolve(conn, CFG, _load_data(conn))
            elif name == "supervise":
                supervisor.supervise(conn, CFG)
            elif name == "paper":
                paper_trade.run_paper(conn, CFG, _load_data(conn))
            elif name == "run":
                for sym in CFG["symbols"]:
                    feed.fetch_ohlcv(conn, sym, CFG["timeframe"], CFG["history_days"])
                evolution.evolve(conn, CFG, _load_data(conn))
                supervisor.supervise(conn, CFG)
                paper_trade.run_paper(conn, CFG, _load_data(conn))
            print(f"[OK] Задача '{name}' завершена.")
    except Exception as e:  # noqa
        print(f"[ОШИБКА] {type(e).__name__}: {e}")
    finally:
        conn.close()
        with _LOCK:
            TASK["running"] = False
            TASK["finished_at"] = time.strftime("%H:%M:%S")


@app.route("/api/run/<name>", methods=["POST"])
def api_run(name):
    if name not in ("fetch", "evolve", "supervise", "paper", "run"):
        abort(404)
    with _LOCK:
        if TASK["running"]:
            return jsonify({"ok": False, "error": "Задача уже выполняется"}), 409
        TASK.update(running=True, name=name, log=[f"▶ Запуск: {name} ..."],
                    finished_at=None)
    threading.Thread(target=_run_task, args=(name,), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/task")
def api_task():
    return jsonify(TASK)


@app.route("/api/live/<action>", methods=["POST"])
def api_live(action):
    if action == "start":
        if not LIVE["running"]:
            LIVE.update(running=True, stop=False, log=["▶ Живая торговля запущена..."])
            threading.Thread(target=_live_loop, daemon=True).start()
        return jsonify({"ok": True})
    if action == "stop":
        LIVE["stop"] = True
        return jsonify({"ok": True})
    abort(404)


def _macro():
    if time.time() - _MACRO["ts"] > 300 or _MACRO["data"] is None:
        mc = CFG.get("macro", {})
        try:
            _MACRO["data"] = macro_feed.etf_flow_bias(
                mc.get("asset", "BTC"), mc.get("lookback_days", 5),
                mc.get("block_threshold_musd", 0))
        except Exception as e:  # noqa
            _MACRO["data"] = {"bias": "neutral", "note": f"ошибка: {e}"}
        _MACRO["ts"] = time.time()
    return _MACRO["data"]


@app.route("/api/status")
def api_status():
    conn = db.connect(CFG["db_path"])
    out = {"counts": {}, "agents": [], "decisions": [], "trades": [], "paper": {}}
    for st in ("candidate", "promoted", "killed"):
        out["counts"][st] = len(db.get_agents(conn, st))

    rows = conn.execute(
        "SELECT * FROM agents WHERE status IN ('candidate','promoted') "
        "ORDER BY test_sharpe DESC LIMIT 25").fetchall()
    for a in rows:
        g = json.loads(a["genome"])
        out["agents"].append({
            "id": a["id"], "type": g["type"], "symbol": a["symbol"],
            "status": a["status"],
            "test_sharpe": a["test_sharpe"], "train_sharpe": a["train_sharpe"],
            "consistency": a["consistency"], "test_trades": a["test_trades"],
            "test_return": a["test_return"], "test_maxdd": a["test_maxdd"],
        })

    for d in conn.execute(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT 20").fetchall():
        out["decisions"].append({
            "ts": d["ts"][11:19], "agent_id": d["agent_id"],
            "action": d["action"], "backend": d["backend"],
            "rationale": d["rationale"]})

    trades = conn.execute(
        "SELECT * FROM paper_trades ORDER BY id DESC LIMIT 30").fetchall()
    for t in trades:
        out["trades"].append({
            "ts": t["ts"][11:19], "agent_id": t["agent_id"], "symbol": t["symbol"],
            "side": t["side"], "price": round(t["price"], 2),
            "pnl": t["pnl"], "reason": t["reason"]})

    pnls = [t["pnl"] for t in trades if t["pnl"] is not None]
    out["paper"] = {
        "closed_trades": len(pnls),
        "realized_pnl": round(sum(pnls), 2) if pnls else 0,
        "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls), 3) if pnls else 0,
        "start_capital": CFG["paper"]["starting_capital"]}

    out["macro"] = _macro()

    # живой счёт
    acc = conn.execute("SELECT * FROM live_account WHERE id=1").fetchone()
    npos = conn.execute("SELECT COUNT(*) c FROM live_positions").fetchone()["c"]
    out["live"] = {
        "running": LIVE["running"],
        "log": LIVE["log"][-12:],
        "capital": round(acc["capital"], 2) if acc else None,
        "open_positions": npos,
        "interval": CFG.get("live", {}).get("interval_seconds", 300),
    }
    conn.close()
    return jsonify(out)


@app.route("/")
def index():
    return render_template_string(HTML)


HTML = r"""
<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Крипто-бот · панель</title>
<style>
:root{--bg:#0b0e14;--panel:#141a24;--panel2:#1b2230;--line:#222c3c;
--txt:#d7dee8;--mut:#8493a8;--grn:#22c55e;--red:#ef4444;--blu:#3b82f6;--yel:#eab308;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);
font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:20px}
h1{font-size:20px;margin:0}h2{font-size:14px;color:var(--mut);text-transform:uppercase;
letter-spacing:.06em;margin:0 0 10px}
.top{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:18px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:18px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.card .v{font-size:22px;font-weight:600;margin-top:4px}
.card .l{color:var(--mut);font-size:12px}
.badge{padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600}
.risk_on{background:rgba(34,197,94,.15);color:var(--grn)}
.risk_off{background:rgba(239,68,68,.15);color:var(--red)}
.neutral{background:rgba(132,147,168,.15);color:var(--mut)}
.bar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}
button{background:var(--panel2);color:var(--txt);border:1px solid var(--line);
padding:9px 16px;border-radius:8px;cursor:pointer;font-size:13px;font-weight:500}
button:hover{border-color:var(--blu)}button:disabled{opacity:.4;cursor:not-allowed}
button.primary{background:var(--blu);border-color:var(--blu);color:#fff}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:18px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--mut);font-weight:500;padding:6px 8px;border-bottom:1px solid var(--line)}
td{padding:6px 8px;border-bottom:1px solid var(--line)}
tr:last-child td{border-bottom:none}
.pos{color:var(--grn)}.neg{color:var(--red)}
.tag{font-size:11px;padding:1px 7px;border-radius:5px}
.t-promote,.t-promoted{background:rgba(34,197,94,.15);color:var(--grn)}
.t-kill,.t-killed{background:rgba(239,68,68,.15);color:var(--red)}
.t-hold,.t-candidate{background:rgba(59,130,246,.15);color:var(--blu)}
.t-generate{background:rgba(234,179,8,.15);color:var(--yel)}
#log,#livelog{background:#070a0f;border:1px solid var(--line);border-radius:8px;padding:12px;
height:200px;overflow:auto;font:12px/1.5 ui-monospace,Consolas,monospace;color:#9fb3c8;white-space:pre-wrap}
#livelog{height:150px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--mut);margin-right:6px}
.dot.run{background:var(--yel);animation:p 1s infinite}@keyframes p{50%{opacity:.3}}
.mut{color:var(--mut)}
</style></head><body><div class="wrap">

<div class="top">
  <div><h1>🤖 Крипто-бот · панель управления</h1>
  <div class="mut" id="sub">Инфраструктура — Python · Управление — супервизор</div></div>
  <div id="macro"><span class="badge neutral">макро…</span></div>
</div>

<div class="cards" id="cards"></div>

<div class="panel">
  <h2>Управление</h2>
  <div class="bar">
    <button onclick="run('fetch')" id="b-fetch">1 · Загрузить данные</button>
    <button onclick="run('evolve')" id="b-evolve">2 · Эволюция</button>
    <button onclick="run('supervise')" id="b-supervise">3 · Супервизор</button>
    <button onclick="run('paper')" id="b-paper">4 · Бумага</button>
    <button onclick="run('run')" id="b-run" class="primary">▶ Всё подряд</button>
    <span class="mut" id="taskstate" style="align-self:center"><span class="dot"></span>простаивает</span>
  </div>
  <div id="log"></div>
</div>

<div class="panel">
  <h2>Живая торговля (реальное время · без реальных денег)</h2>
  <div class="bar">
    <button onclick="live('start')" id="b-live-start" class="primary">▶ Запустить живую торговлю</button>
    <button onclick="live('stop')" id="b-live-stop">⏹ Остановить</button>
    <span class="mut" id="livestate" style="align-self:center"><span class="dot"></span>выключена</span>
  </div>
  <div id="livelog"></div>
</div>

<div class="grid">
  <div class="panel"><h2>Агенты (топ по out-of-sample Sharpe)</h2>
    <table><thead><tr><th>#</th><th>тип</th><th>символ</th><th>статус</th>
    <th>test Sharpe</th><th>cons.</th><th>сделок</th></tr></thead>
    <tbody id="agents"></tbody></table></div>

  <div class="panel"><h2>Решения супервизора</h2>
    <table><thead><tr><th>время</th><th>агент</th><th>действие</th><th>почему</th></tr></thead>
    <tbody id="decisions"></tbody></table></div>
</div>

<div class="panel"><h2>Бумажные сделки</h2>
  <table><thead><tr><th>время</th><th>агент</th><th>символ</th><th>сторона</th>
  <th>цена</th><th>PnL</th><th>причина</th></tr></thead>
  <tbody id="trades"></tbody></table></div>

<script>
const $=s=>document.querySelector(s);
function num(x,d=2){return x==null?'—':(+x).toFixed(d)}
function cls(x){return x>0?'pos':(x<0?'neg':'')}

async function run(name){
  const r=await fetch('/api/run/'+name,{method:'POST'});
  if(r.status===409){alert('Задача уже выполняется');return}
  poll();
}
function setButtons(dis){['fetch','evolve','supervise','paper','run'].forEach(n=>$('#b-'+n).disabled=dis)}

async function live(action){
  const r=await fetch('/api/live/'+action,{method:'POST'});
  setTimeout(refresh,300);
}

async function poll(){
  const t=await (await fetch('/api/task')).json();
  $('#log').textContent=(t.log||[]).join('\n');$('#log').scrollTop=1e9;
  setButtons(t.running);
  $('#taskstate').innerHTML=t.running
    ?'<span class="dot run"></span>выполняется: '+t.name
    :'<span class="dot"></span>простаивает'+(t.finished_at?' · готово '+t.finished_at:'');
  if(t.running)setTimeout(poll,1200);
}

async function refresh(){
  const d=await (await fetch('/api/status')).json();
  const m=d.macro||{};
  $('#macro').innerHTML='<span class="badge '+(m.bias||'neutral')+'">ETF: '+(m.bias||'—')+
    '</span> <span class="mut">'+(m.note||'')+'</span>';
  const cap=d.paper.start_capital, pnl=d.paper.realized_pnl;
  $('#cards').innerHTML=`
   <div class="card"><div class="l">Кандидаты</div><div class="v">${d.counts.candidate}</div></div>
   <div class="card"><div class="l">Продвинуто (live)</div><div class="v" style="color:var(--grn)">${d.counts.promoted}</div></div>
   <div class="card"><div class="l">Убито</div><div class="v" style="color:var(--red)">${d.counts.killed}</div></div>
   <div class="card"><div class="l">Бумажный PnL</div><div class="v ${cls(pnl)}">${pnl>0?'+':''}${num(pnl)}</div></div>
   <div class="card"><div class="l">Сделок / Win rate</div><div class="v">${d.paper.closed_trades} · ${(d.paper.win_rate*100).toFixed(0)}%</div></div>`;

  $('#agents').innerHTML=d.agents.map(a=>`<tr>
    <td>${a.id}</td><td>${a.type}</td><td>${a.symbol}</td>
    <td><span class="tag t-${a.status}">${a.status}</span></td>
    <td class="${cls(a.test_sharpe)}">${num(a.test_sharpe)}</td>
    <td>${num(a.consistency)}</td><td>${a.test_trades??'—'}</td></tr>`).join('')
    ||'<tr><td colspan=7 class="mut">пусто — запусти Эволюцию</td></tr>';

  $('#decisions').innerHTML=d.decisions.map(x=>`<tr>
    <td class="mut">${x.ts}</td><td>#${x.agent_id??''}</td>
    <td><span class="tag t-${x.action}">${x.action}</span></td>
    <td class="mut">${x.rationale}</td></tr>`).join('')
    ||'<tr><td colspan=4 class="mut">пока нет решений</td></tr>';

  const lv=d.live||{};
  $('#livelog').textContent=(lv.log||[]).join('\n');$('#livelog').scrollTop=1e9;
  $('#livestate').innerHTML=lv.running
    ?'<span class="dot run"></span>работает · проверка каждые '+Math.round((lv.interval||300)/60)+' мин · позиций '+lv.open_positions
    :'<span class="dot"></span>выключена';
  $('#b-live-start').disabled=lv.running;$('#b-live-stop').disabled=!lv.running;

  $('#trades').innerHTML=d.trades.map(t=>`<tr>
    <td class="mut">${t.ts}</td><td>#${t.agent_id}</td><td>${t.symbol}</td>
    <td>${t.side}</td><td>${num(t.price)}</td>
    <td class="${cls(t.pnl)}">${t.pnl==null?'—':(t.pnl>0?'+':'')+num(t.pnl)}</td>
    <td class="mut">${t.reason||''}</td></tr>`).join('')
    ||'<tr><td colspan=7 class="mut">сделок пока нет</td></tr>';
}

refresh();setInterval(refresh,3000);poll();
</script></div></body></html>
"""

if __name__ == "__main__":
    print("Дашборд: http://127.0.0.1:5000  (Ctrl+C для остановки)")
    app.run(host="127.0.0.1", port=5000, debug=False)
