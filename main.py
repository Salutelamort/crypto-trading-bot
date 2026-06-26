"""
CLI торгового бота для криптовалют.

Архитектура (из r/algotrading, комментарий piratastuertos с фото):
  - Инфраструктура = детерминированный Python (data, indicators, backtest, risk)
  - Управление = супервизор (правила или Claude): kill / promote / generate
  - Эволюция = генерация и отбор тысяч стратегий, выживают единицы
  - Всё в SQLite. Сначала бумага, потом (может быть) реальные деньги.

Типичный рабочий цикл:
  python main.py fetch       # 1. скачать рыночные данные (Binance, без ключей)
  python main.py evolve      # 2. сгенерировать и отобрать агентов
  python main.py supervise   # 3. супервизор продвигает/убивает агентов
  python main.py paper       # 4. форвард-тест продвинутых на out-of-sample
  python main.py status      # показать состояние популяции
  python main.py run         # всё подряд: fetch+evolve+supervise+paper
"""
import sys
import json
from datetime import datetime, timezone

import yaml

from src import db
from src import data_feed as feed
from src import evolution
from src import supervisor
from src import paper_trade
from src import live_trade


def load_cfg(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _timeframes(cfg):
    return cfg.get("timeframes", [cfg["timeframe"]])


def cmd_fetch(conn, cfg):
    print("Загрузка рыночных данных с Binance (публичный API, без ключей)...")
    feed.fetch_all(conn, cfg["symbols"], _timeframes(cfg), cfg["history_days"])


def _load_data(conn, cfg):
    """Возвращает {(symbol, timeframe): DataFrame} по всем парам (мультитаймфрейм)."""
    data = feed.load_all(conn, cfg["symbols"], _timeframes(cfg))
    if not data:
        print("  [!] Нет данных. Запусти: python main.py fetch")
    return data


def cmd_evolve(conn, cfg):
    data = _load_data(conn, cfg)
    if not data:
        return
    evolution.evolve(conn, cfg, data)


def cmd_supervise(conn, cfg):
    supervisor.supervise(conn, cfg)


def cmd_paper(conn, cfg):
    data = _load_data(conn, cfg)
    if not data:
        return
    paper_trade.run_paper(conn, cfg, data)


def cmd_live(conn, cfg):
    live_trade.run_live(conn, cfg)


def cmd_status(conn, cfg):
    for status in ("candidate", "promoted", "killed"):
        agents = db.get_agents(conn, status)
        print(f"\n[{status}] всего: {len(agents)}")
        for a in sorted(agents, key=lambda x: x["test_sharpe"] or -99,
                        reverse=True)[:10]:
            g = json.loads(a["genome"])
            ts = a["test_sharpe"] if a["test_sharpe"] is not None else 0
            cons = a["consistency"] if a["consistency"] is not None else 0
            print(f"   #{a['id']:<4} {g['type']:14s} {a['symbol']:8s} "
                  f"test_sharpe={ts:6.2f} consistency={cons:.2f}")
    q = db.quarantined_symbols(conn)
    if q:
        print(f"\nКарантин символов: {sorted(q)}")


def cmd_run(conn, cfg):
    cmd_fetch(conn, cfg)
    cmd_evolve(conn, cfg)
    cmd_supervise(conn, cfg)
    cmd_paper(conn, cfg)


def _parse_ts(s):
    """ISO-строка из БД → datetime (aware, UTC). None если не разобрать."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _last_price(symbol, cfg, _cache={}):
    """Свежая цена символа (mark-to-market открытых позиций). None если нет сети —
    тогда analyze честно откатится на цену входа (нулевая переоценка)."""
    if symbol in _cache:
        return _cache[symbol]
    px = None
    try:
        df = feed.fetch_recent(symbol, cfg["timeframe"], 2)
        px = float(df["close"].iloc[-1])
    except Exception:  # noqa — нет сети/данных → фолбэк на вход
        px = None
    _cache[symbol] = px
    return px


def _pos_value(notional, direction, entry, price):
    """Стоимость позиции mark-to-market — та же формула, что rk.Position.value."""
    return notional * (1 + direction * (price / entry - 1))


def _agent_type(conn, agent_id, _cache={}):
    """Тип стратегии (геном) по id агента, с кэшем."""
    if agent_id in _cache:
        return _cache[agent_id]
    row = conn.execute("SELECT genome FROM agents WHERE id=?", (agent_id,)).fetchone()
    t = "?"
    if row:
        try:
            t = json.loads(row["genome"]).get("type", "?")
        except (ValueError, TypeError):
            pass
    _cache[agent_id] = t
    return t


def cmd_analyze(conn, cfg):
    """Анализ форвард-теста по data/bot.db (только чтение): счёт, закрытые
    сделки, открытые позиции, решения супервизора, популяция."""
    start_cap = cfg["paper"]["starting_capital"]
    now = datetime.now(timezone.utc)
    print("=" * 60)
    print("  АНАЛИЗ ФОРВАРД-ТЕСТА  (data/bot.db, только чтение)")
    print("=" * 60)

    # ── Открытые позиции (нужны заранее для расчёта equity) ───────────
    pos = conn.execute("SELECT * FROM live_positions ORDER BY opened_at").fetchall()
    # mark-to-market каждой позиции: свежая цена, фолбэк на вход (нулевая переоценка)
    marks = []   # (row, price, price_src, value, unrealized)
    pos_value = 0.0
    for p in pos:
        entry = p["entry_price"]
        d = p["direction"] if "direction" in p.keys() and p["direction"] else 1
        notional = p["notional"] if "notional" in p.keys() and p["notional"] \
            else p["units"] * entry
        live_px = _last_price(p["symbol"], cfg)
        px = live_px if live_px else entry
        src = "рынок" if live_px else "вход (нет сети)"
        val = _pos_value(notional, d, entry, px)
        marks.append((p, px, src, val, val - notional))
        pos_value += val

    # ── Счёт ──────────────────────────────────────────────────────────
    acc = conn.execute("SELECT * FROM live_account WHERE id=1").fetchone()
    print("\n— СЧЁТ —")
    if acc:
        cash = acc["capital"]
        equity = cash + pos_value                 # настоящий счёт = кэш + позиции
        peak = acc["peak_equity"] or equity
        ret = equity / start_cap - 1 if start_cap else 0
        dd = equity / peak - 1 if peak else 0
        started = _parse_ts(acc["started_at"])
        days = (now - started).total_seconds() / 86400 if started else None
        print(f"  Equity (счёт):  {equity:,.2f}   (старт {start_cap:,.0f})   "
              f"{'+' if ret >= 0 else ''}{ret * 100:.2f}%")
        print(f"  в т.ч. кэш:     {cash:,.2f}   в позициях: {pos_value:,.2f}")
        print(f"  Пик equity:     {peak:,.2f}")
        print(f"  Просадка от пика: {dd * 100:.2f}%")
        if days is not None:
            print(f"  Возраст форварда: {days:.1f} дн. (старт {acc['started_at'][:10]})")
    else:
        print("  Счёт ещё не инициализирован (live_account пуст).")

    # ── Закрытые сделки ──────────────────────────────────────────────
    trades = conn.execute(
        "SELECT * FROM paper_trades WHERE pnl IS NOT NULL ORDER BY ts").fetchall()
    print(f"\n— ЗАКРЫТЫЕ СДЕЛКИ —  всего: {len(trades)}")
    if trades:
        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gross_win, gross_loss = sum(wins), abs(sum(losses))
        pf = gross_win / gross_loss if gross_loss else float("inf")
        wr = len(wins) / len(pnls)
        print(f"  PnL итого: {sum(pnls):+.2f}   винрейт: {wr * 100:.1f}%   "
              f"profit factor: {pf:.2f}")
        if wins:
            print(f"  ср. прибыль: +{gross_win / len(wins):.2f}  (побед {len(wins)})")
        if losses:
            print(f"  ср. убыток:  -{gross_loss / len(losses):.2f}  (убытков {len(losses)})")
        best, worst = max(trades, key=lambda t: t["pnl"]), min(trades, key=lambda t: t["pnl"])
        print(f"  лучшая: {best['pnl']:+.2f} {best['symbol']}   "
              f"худшая: {worst['pnl']:+.2f} {worst['symbol']}")

        def _group(keyfn, title):
            agg = {}
            for t in trades:
                k = keyfn(t)
                a = agg.setdefault(k, [0, 0.0])
                a[0] += 1
                a[1] += t["pnl"]
            print(f"  {title}:")
            for k, (n, s) in sorted(agg.items(), key=lambda kv: kv[1][1]):
                print(f"     {k:14s} сделок {n:3d}   PnL {s:+.2f}")

        _group(lambda t: t["symbol"], "по символам")
        _group(lambda t: _agent_type(conn, t["agent_id"]), "по типу стратегии")
        for reason in ("причина выхода",):
            _group(lambda t: t["reason"] or "?", reason)
    else:
        print("  Закрытых сделок пока нет — форвард ещё копит историю.")

    # ── Открытые позиции ─────────────────────────────────────────────
    print(f"\n— ОТКРЫТЫЕ ПОЗИЦИИ —  всего: {len(marks)}")
    for p, px, src, val, unreal in marks:
        opened = _parse_ts(p["opened_at"])
        age = f"{(now - opened).total_seconds() / 3600:.1f}ч" if opened else "?"
        d = p["direction"] if "direction" in p.keys() else None
        side = {1: "LONG", -1: "SHORT"}.get(d, str(d))
        print(f"  {p['symbol']:8s} {side:5s} вход @ {p['entry_price']:.4f}  "
              f"тек. {px:.4f} ({src})  нереализ. {unreal:+.2f}  "
              f"тип {_agent_type(conn, p['agent_id'])}  возраст {age}")

    # ── Решения супервизора ──────────────────────────────────────────
    print("\n— РЕШЕНИЯ СУПЕРВИЗОРА —")
    day_ago = (now.timestamp() - 86400)
    overall, last24 = {}, {}
    for row in conn.execute("SELECT ts, action FROM decisions").fetchall():
        a = row["action"]
        overall[a] = overall.get(a, 0) + 1
        dt = _parse_ts(row["ts"])
        if dt and dt.timestamp() >= day_ago:
            last24[a] = last24.get(a, 0) + 1
    actions = sorted(set(overall) | set(last24))
    if actions:
        print(f"  {'действие':12s} {'за 24ч':>8s} {'всего':>10s}")
        for a in actions:
            print(f"  {a:12s} {last24.get(a, 0):>8d} {overall.get(a, 0):>10d}")
    else:
        print("  Решений пока нет.")

    # ── Популяция ────────────────────────────────────────────────────
    print("\n— ПОПУЛЯЦИЯ —")
    for st in ("candidate", "promoted", "killed"):
        print(f"  {st:10s} {len(db.get_agents(conn, st))}")
    promoted = sorted(db.get_agents(conn, "promoted"),
                      key=lambda a: a["test_sharpe"] or -99, reverse=True)[:10]
    if promoted:
        print("  Топ live-агентов по OOS Sharpe:")
        for a in promoted:
            g = json.loads(a["genome"])
            print(f"     #{a['id']:<5} {g['type']:14s} {a['symbol']:8s} "
                  f"Sharpe {a['test_sharpe'] or 0:5.2f}  "
                  f"alpha {(a['test_alpha'] or 0) * 100:+5.1f}%  "
                  f"trades {a['test_trades'] or 0}")
    q = db.quarantined_symbols(conn)
    if q:
        print(f"  Карантин символов: {sorted(q)}")
    print()


COMMANDS = {
    "fetch": cmd_fetch, "evolve": cmd_evolve, "supervise": cmd_supervise,
    "paper": cmd_paper, "live": cmd_live, "status": cmd_status, "run": cmd_run,
    "analyze": cmd_analyze,
}


def main():
    cfg = load_cfg()
    conn = db.connect(cfg["db_path"])
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    fn = COMMANDS.get(cmd)
    if not fn:
        print(f"Неизвестная команда: {cmd}")
        print(f"Доступно: {', '.join(COMMANDS)}")
        sys.exit(1)
    fn(conn, cfg)
    conn.close()


if __name__ == "__main__":
    main()
