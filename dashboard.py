"""
ПАНЕЛЬ-НАБЛЮДАТЕЛЬ за облачным обучением.

Только наблюдение, никаких кнопок управления. Обучение идёт само в облаке
(GitHub Actions каждые 15 минут). Панель раз в ~90 секунд подтягивает свежее
состояние из облака (git pull) и показывает его в реальном времени:
прогресс обучения, агентов, решения эволюции, сделки, макро/новостной фон.

Запуск: python dashboard.py  →  http://127.0.0.1:5000
"""
import os
import csv
import json
import time
import threading
import subprocess

import yaml
from flask import Flask, jsonify, render_template_string

from src import db, macro_feed, news_feed

app = Flask(__name__)
ROOT = os.path.dirname(os.path.abspath(__file__))
CFG = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))

_MACRO = {"ts": 0, "data": None}
_NEWS = {"ts": 0, "data": None}
SYNC = {"ts": 0, "status": "ожидание...", "last_ok": None}


# ---------- авто-синхронизация с облаком ----------
def _sync_loop():
    while True:
        try:
            r = subprocess.run(["git", "pull", "--no-edit"], cwd=ROOT,
                               capture_output=True, text=True, timeout=90)
            if r.returncode == 0:
                SYNC["status"] = "синхронизировано с облаком"
                SYNC["last_ok"] = time.strftime("%H:%M:%S")
            else:
                SYNC["status"] = "нет связи с облаком (показываю последнее)"
        except Exception:  # noqa
            SYNC["status"] = "нет интернета (показываю последнее)"
        SYNC["ts"] = time.time()
        time.sleep(90)


def _macro():
    if time.time() - _MACRO["ts"] > 300 or _MACRO["data"] is None:
        try:
            mc = CFG.get("macro", {})
            _MACRO["data"] = macro_feed.etf_flow_bias(
                mc.get("asset", "BTC"), mc.get("lookback_days", 5),
                mc.get("block_threshold_musd", 0))
        except Exception as e:  # noqa
            _MACRO["data"] = {"bias": "neutral", "note": f"ошибка: {e}"}
        _MACRO["ts"] = time.time()
    return _MACRO["data"]


def _news():
    if time.time() - _NEWS["ts"] > 300 or _NEWS["data"] is None:
        try:
            _NEWS["data"] = news_feed.news_gate(CFG)
        except Exception as e:  # noqa
            _NEWS["data"] = {"block": False, "reason": f"ошибка: {e}",
                             "fng": {"value": None, "label": "n/a"}, "news_hits": 0}
        _NEWS["ts"] = time.time()
    return _NEWS["data"]


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
            "status": a["status"], "test_sharpe": a["test_sharpe"],
            "consistency": a["consistency"], "test_trades": a["test_trades"]})

    for d in conn.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT 20").fetchall():
        out["decisions"].append({
            "ts": d["ts"][11:19], "agent_id": d["agent_id"],
            "action": d["action"], "rationale": d["rationale"]})

    trades = conn.execute("SELECT * FROM paper_trades ORDER BY id DESC LIMIT 30").fetchall()
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

    acc = conn.execute("SELECT * FROM live_account WHERE id=1").fetchone()
    npos = conn.execute("SELECT COUNT(*) c FROM live_positions").fetchone()["c"]
    out["live"] = {"capital": round(acc["capital"], 2) if acc else CFG["paper"]["starting_capital"],
                   "open_positions": npos}

    out["macro"] = _macro()
    out["news"] = _news()
    out["sync"] = {"status": SYNC["status"], "last_ok": SYNC["last_ok"]}
    conn.close()
    return jsonify(out)


@app.route("/api/track")
def api_track():
    rows = []
    path = os.path.join(ROOT, "TRACK_RECORD.csv")
    try:
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        pass
    return jsonify(rows)


@app.route("/")
def index():
    return render_template_string(HTML)


HTML = r"""
<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Крипто-бот · наблюдение за обучением</title>
<style>
:root{--bg:#0b0e14;--panel:#141a24;--panel2:#1b2230;--line:#222c3c;
--txt:#d7dee8;--mut:#8493a8;--grn:#22c55e;--red:#ef4444;--blu:#3b82f6;--yel:#eab308;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);
font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:20px}
h1{font-size:20px;margin:0}h2{font-size:14px;color:var(--mut);text-transform:uppercase;
letter-spacing:.06em;margin:0 0 10px}
.top{display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;margin-bottom:18px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:18px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.card .v{font-size:22px;font-weight:600;margin-top:4px}
.card .l{color:var(--mut);font-size:12px}
.badge{padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600}
.risk_on{background:rgba(34,197,94,.15);color:var(--grn)}
.risk_off{background:rgba(239,68,68,.15);color:var(--red)}
.neutral{background:rgba(132,147,168,.15);color:var(--mut)}
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
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--grn);margin-right:6px}
.mut{color:var(--mut)}
</style></head><body><div class="wrap">

<div class="top">
  <div>
    <h1>🤖 Крипто-бот · наблюдение за обучением</h1>
    <div class="mut">Обучение идёт само в облаке. Эта панель только показывает прогресс.</div>
    <div class="mut" id="sync" style="margin-top:4px"><span class="dot"></span>—</div>
  </div>
  <div style="text-align:right">
    <div id="macro"><span class="badge neutral">макро…</span></div>
    <div id="news" style="margin-top:6px"><span class="badge neutral">новости…</span></div>
  </div>
</div>

<div class="cards" id="cards"></div>

<div class="panel">
  <h2>Прогресс обучения (из облака, обновляется само)</h2>
  <div id="trackchart" style="margin-bottom:12px"></div>
  <div style="max-height:260px;overflow:auto">
    <table><thead><tr><th>время (UTC)</th><th>капитал</th><th>кандидатов</th>
    <th>в live</th><th>лучший Sharpe</th><th>рынок</th><th>F&amp;G</th></tr></thead>
    <tbody id="track"></tbody></table>
  </div>
</div>

<div class="grid">
  <div class="panel"><h2>Живые агенты (топ по OOS Sharpe)</h2>
    <table><thead><tr><th>#</th><th>тип</th><th>символ</th><th>статус</th>
    <th>Sharpe</th><th>устойч.</th><th>сделок</th></tr></thead>
    <tbody id="agents"></tbody></table></div>

  <div class="panel"><h2>Решения эволюции</h2>
    <table><thead><tr><th>время</th><th>агент</th><th>действие</th><th>почему</th></tr></thead>
    <tbody id="decisions"></tbody></table></div>
</div>

<div class="panel"><h2>Бумажные сделки</h2>
  <table><thead><tr><th>время</th><th>агент</th><th>символ</th><th>сторона</th>
  <th>цена</th><th>PnL</th><th>причина</th></tr></thead>
  <tbody id="trades"></tbody></table></div>

<script>
const $=s=>document.querySelector(s);
function num(x,d=2){return x==null||x===''?'—':(+x).toFixed(d)}
function cls(x){return x>0?'pos':(x<0?'neg':'')}

function lineChart(vals,w,h){
  const pts=vals.map((v,i)=>[i,v]).filter(p=>p[1]!=null&&!isNaN(p[1]));
  if(pts.length<2) return '<div class="mut">Пока мало данных для графика (нужно 2+ точки).</div>';
  const ys=pts.map(p=>p[1]); let mn=Math.min(...ys,0),mx=Math.max(...ys,0);
  if(mn===mx){mn-=1;mx+=1;}
  const pad=24,W=w,H=h;
  const xf=i=>pad+(i/(vals.length-1))*(W-2*pad);
  const yf=v=>H-pad-((v-mn)/(mx-mn))*(H-2*pad);
  const poly=pts.map(p=>`${xf(p[0]).toFixed(1)},${yf(p[1]).toFixed(1)}`).join(' ');
  const z=yf(0);
  return `<svg width="100%" viewBox="0 0 ${W} ${H}" style="background:#070a0f;border:1px solid var(--line);border-radius:8px">
    <line x1="${pad}" y1="${z}" x2="${W-pad}" y2="${z}" stroke="#33415580" stroke-dasharray="4"/>
    <text x="4" y="${z-3}" fill="#64748b" font-size="10">0</text>
    <text x="4" y="14" fill="#64748b" font-size="10">${mx.toFixed(1)}</text>
    <text x="4" y="${H-6}" fill="#64748b" font-size="10">${mn.toFixed(1)}</text>
    <polyline fill="none" stroke="#3b82f6" stroke-width="2" points="${poly}"/></svg>`;
}

async function renderTrack(){
  let d=[]; try{ d=await (await fetch('/api/track')).json(); }catch(e){ return; }
  const sh=d.map(r=>r.best_test_sharpe===''?null:parseFloat(r.best_test_sharpe));
  $('#trackchart').innerHTML='<div class="mut" style="margin-bottom:4px">Лучший Sharpe во времени (растёт к 0 и выше = бот находит преимущество)</div>'+lineChart(sh,640,150);
  $('#track').innerHTML=d.slice().reverse().slice(0,120).map(r=>`<tr>
    <td class="mut">${r.date}</td><td>${num(r.capital,0)}</td>
    <td>${r.candidates}</td><td>${r.promoted}</td>
    <td class="${(parseFloat(r.best_test_sharpe)||0)>0?'pos':'neg'}">${r.best_test_sharpe||'—'}</td>
    <td class="mut">${r.macro_bias||''}</td><td class="mut">${r.fear_greed||''}</td></tr>`).join('')
    ||'<tr><td colspan=7 class="mut">журнал пуст — облако ещё не присылало данные</td></tr>';
}

async function refresh(){
  let d; try{ d=await (await fetch('/api/status')).json(); }catch(e){ return; }
  const m=d.macro||{},nw=d.news||{},fng=(nw.fng||{}),sy=d.sync||{};
  $('#sync').innerHTML='<span class="dot"></span>'+(sy.status||'—')+(sy.last_ok?(' · последняя синхронизация '+sy.last_ok):'');
  $('#macro').innerHTML='<span class="badge '+(m.bias||'neutral')+'">ETF: '+(m.bias||'—')+
    '</span> <span class="mut">'+(m.note||'')+'</span>';
  const ncls=nw.block?'risk_off':'risk_on';
  $('#news').innerHTML='<span class="badge '+ncls+'">Новости: '+(nw.block?'входы стоп':'спокойно')+
    '</span> <span class="mut">F&amp;G '+(fng.value??'—')+' '+(fng.label||'')+'</span>';

  const cap=d.live.capital,start=d.paper.start_capital,ret=cap/start-1;
  $('#cards').innerHTML=`
   <div class="card"><div class="l">Капитал (бумага)</div><div class="v ${cls(ret)}">${num(cap,0)}</div></div>
   <div class="card"><div class="l">Доходность</div><div class="v ${cls(ret)}">${(ret*100).toFixed(2)}%</div></div>
   <div class="card"><div class="l">Кандидаты</div><div class="v">${d.counts.candidate}</div></div>
   <div class="card"><div class="l">Продвинуто (live)</div><div class="v" style="color:var(--grn)">${d.counts.promoted}</div></div>
   <div class="card"><div class="l">Убито всего</div><div class="v" style="color:var(--red)">${d.counts.killed}</div></div>
   <div class="card"><div class="l">Открытых позиций</div><div class="v">${d.live.open_positions}</div></div>`;

  $('#agents').innerHTML=d.agents.map(a=>`<tr>
    <td>${a.id}</td><td>${a.type}</td><td>${a.symbol}</td>
    <td><span class="tag t-${a.status}">${a.status}</span></td>
    <td class="${cls(a.test_sharpe)}">${num(a.test_sharpe)}</td>
    <td>${a.consistency==null?'—':Math.round(a.consistency*100)+'%'}</td>
    <td>${a.test_trades??'—'}</td></tr>`).join('')
    ||'<tr><td colspan=7 class="mut">пока нет живых агентов</td></tr>';

  $('#decisions').innerHTML=d.decisions.map(x=>`<tr>
    <td class="mut">${x.ts}</td><td>#${x.agent_id??''}</td>
    <td><span class="tag t-${x.action}">${x.action}</span></td>
    <td class="mut">${x.rationale}</td></tr>`).join('')
    ||'<tr><td colspan=4 class="mut">пока нет решений</td></tr>';

  $('#trades').innerHTML=d.trades.map(t=>`<tr>
    <td class="mut">${t.ts}</td><td>#${t.agent_id}</td><td>${t.symbol}</td>
    <td>${t.side}</td><td>${num(t.price)}</td>
    <td class="${cls(t.pnl)}">${t.pnl==null?'—':(t.pnl>0?'+':'')+num(t.pnl)}</td>
    <td class="mut">${t.reason||''}</td></tr>`).join('')
    ||'<tr><td colspan=7 class="mut">сделок пока нет</td></tr>';
}

refresh();renderTrack();
setInterval(()=>{refresh();renderTrack();},5000);
</script></div></body></html>
"""

if __name__ == "__main__":
    threading.Thread(target=_sync_loop, daemon=True).start()
    print("Панель-наблюдатель: http://127.0.0.1:5000  (Ctrl+C для остановки)")
    app.run(host="127.0.0.1", port=5000, debug=False)
