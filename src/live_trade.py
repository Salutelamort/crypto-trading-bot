"""
ЖИВОЙ бумажный трейдинг в реальном времени — БЕЗ биржи и без ключей.

Почему так (а не Binance Testnet): тестнет Binance геоблокирован (451) с машины
пользователя, а VPN на разрешённую страну включить нельзя — он нужен для доступа
к Claude. Поэтому исполнение считаем локально, а цены берём с публичного
`data-api.binance.vision`, который работает при включённом VPN.

Что делает один "тик":
  1. тянет свежие свечи по символам активных агентов;
  2. считает детерминированный сигнал (genome.signal) на последнем баре;
  3. управляет позициями (стоп/трейлинг/тейк — риск приоритетнее сигнала);
  4. открывает позиции по сигналу с учётом лимитов, макро-стража и стоп-крана;
  5. сохраняет состояние счёта и позиций в SQLite (переживает перезапуск).

Торговля 100% детерминированная. Состояние живёт в БД, поэтому можно
останавливать/запускать бота без потери позиций.
"""
import time
import json

from . import db, data_feed as feed, genome as gn, risk as rk, macro_feed
from .db import now_iso


# ---------- состояние живого счёта в SQLite ----------
def _init_account(conn, cfg):
    row = conn.execute("SELECT * FROM live_account WHERE id=1").fetchone()
    if row is None:
        cap = float(cfg["paper"]["starting_capital"])
        conn.execute("INSERT INTO live_account (id,capital,peak_equity,started_at) "
                     "VALUES (1,?,?,?)", (cap, cap, now_iso()))
        conn.commit()
        return cap, cap
    return row["capital"], row["peak_equity"]


def _save_account(conn, capital, peak):
    conn.execute("UPDATE live_account SET capital=?, peak_equity=? WHERE id=1",
                 (capital, peak))
    conn.commit()


def _load_positions(conn):
    pos = {}
    for r in conn.execute("SELECT * FROM live_positions").fetchall():
        p = rk.Position(r["agent_id"], r["symbol"], r["entry_price"], r["units"])
        p.peak_price = r["peak_price"]
        pos[r["agent_id"]] = p
    return pos


def _save_position(conn, p):
    conn.execute(
        "INSERT OR REPLACE INTO live_positions "
        "(agent_id,symbol,entry_price,units,peak_price,opened_at) "
        "VALUES (?,?,?,?,?,?)",
        (p.agent_id, p.symbol, p.entry_price, p.units, p.peak_price, now_iso()))
    conn.commit()


def _del_position(conn, agent_id):
    conn.execute("DELETE FROM live_positions WHERE agent_id=?", (agent_id,))
    conn.commit()


def _active_agents(conn, cfg):
    """Кого торгуем: продвинутых супервизором. Если их нет и разрешён демо-режим —
    берём лучших кандидатов (с явной пометкой, что они НЕ прошли отбор)."""
    promoted = db.get_agents(conn, "promoted")
    if promoted:
        return promoted, False
    live_cfg = cfg.get("live", {})
    if live_cfg.get("allow_unpromoted"):
        cands = sorted(db.get_agents(conn, "candidate"),
                       key=lambda a: a["test_sharpe"] or -99, reverse=True)
        return cands[:live_cfg.get("demo_agents", 2)], True
    return [], False


# ---------- один тик живой торговли ----------
def tick(conn, cfg, verbose=True):
    capital, peak = _init_account(conn, cfg)
    risk_cfg = cfg["risk"]
    fee = cfg["costs"]["fee_pct"]
    slip = cfg["costs"]["slippage_pct"]
    dd_limit = risk_cfg.get("max_portfolio_drawdown", 1.0)

    agents, demo = _active_agents(conn, cfg)
    if not agents:
        if verbose:
            print("Нет агентов для живой торговли. Запусти evolve+supervise, "
                  "или включи live.allow_unpromoted в config.yaml для демо.")
        return

    # макро-страж (farside работает при VPN)
    macro_block = False
    mc = cfg.get("macro", {})
    if mc.get("enabled"):
        try:
            info = macro_feed.etf_flow_bias(mc.get("asset", "BTC"),
                                            mc.get("lookback_days", 5),
                                            mc.get("block_threshold_musd", 0))
            macro_block = info["bias"] == "risk_off"
        except Exception:  # noqa
            pass

    positions = _load_positions(conn)

    # свежие данные по уникальным символам
    symbols = {a["symbol"] for a in agents}
    data = {}
    for s in symbols:
        try:
            data[s] = feed.fetch_recent(s, cfg["timeframe"], 400)
        except Exception as e:  # noqa
            if verbose:
                print(f"  [!] нет данных {s}: {e}")

    # текущий капитал и просадка
    def equity_now():
        eq = capital
        for p in positions.values():
            if p.symbol in data:
                eq += p.units * float(data[p.symbol]["close"].iloc[-1])
        return eq

    eq = equity_now()
    peak = max(peak, eq)
    dd = (peak - eq) / peak if peak else 0.0
    dd_halt = dd > dd_limit

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    actions = []

    for a in agents:
        aid, sym = a["id"], a["symbol"]
        if sym not in data:
            continue
        df = data[sym]
        price = float(df["close"].iloc[-1])
        g = json.loads(a["genome"])
        sig = int(gn.signal(g, df).iloc[-1])
        pos = positions.get(aid)

        # 1. управление позицией
        if pos is not None:
            should_exit, reason = pos.exit_check(price, risk_cfg)
            if not should_exit and sig == 0:
                should_exit, reason = True, "signal"
            if should_exit:
                fill = price * (1 - slip)
                proceeds = pos.units * fill * (1 - fee)
                pnl = proceeds - pos.units * pos.entry_price
                capital += proceeds
                db.log_paper_trade(conn, aid, sym, "SELL", fill, pos.units,
                                   pos.units * fill * fee, round(pnl, 2), reason)
                _del_position(conn, aid)
                del positions[aid]
                actions.append(f"SELL #{aid} {sym} @ {fill:.2f} ({reason}) PnL {pnl:+.2f}")
            else:
                _save_position(conn, pos)  # сохранить обновлённый peak_price

        # 2. вход по сигналу
        elif sig == 1 and not macro_block and not dd_halt \
                and rk.can_open(len(positions), risk_cfg):
            invest = rk.position_size(capital, risk_cfg)
            if 0 < invest <= capital:
                fill = price * (1 + slip)
                units = invest * (1 - fee) / fill
                capital -= invest
                p = rk.Position(aid, sym, fill, units)
                positions[aid] = p
                _save_position(conn, p)
                db.log_paper_trade(conn, aid, sym, "BUY", fill, units,
                                   invest * fee, None, "signal")
                actions.append(f"BUY  #{aid} {sym} @ {fill:.2f} (вложено {invest:.2f})")

    eq = equity_now()
    peak = max(peak, eq)
    _save_account(conn, capital, peak)

    if verbose:
        flags = []
        if demo:
            flags.append("ДЕМО: агенты НЕ прошли отбор")
        if macro_block:
            flags.append("макро risk_off — входы стоп")
        if dd_halt:
            flags.append(f"стоп-кран просадки {dd:.1%}")
        tag = "  [" + "; ".join(flags) + "]" if flags else ""
        ret = eq / float(cfg["paper"]["starting_capital"]) - 1
        print(f"[{stamp}] капитал {eq:,.2f} ({ret:+.2%}) | "
              f"кэш {capital:,.0f} | позиций {len(positions)}{tag}")
        for act in actions:
            print("   → " + act)
        if not actions and not positions:
            print("   нет позиций, ждём сигнал...")


def account_equity(conn, cfg):
    """Текущий капитал живого счёта (кэш + открытые позиции по последней цене)."""
    acc = conn.execute("SELECT * FROM live_account WHERE id=1").fetchone()
    if acc is None:
        cap = float(cfg["paper"]["starting_capital"])
        return cap, cap, 0
    capital = acc["capital"]
    eq = capital
    npos = 0
    for r in conn.execute("SELECT * FROM live_positions").fetchall():
        npos += 1
        try:
            px = float(feed.fetch_recent(r["symbol"], cfg["timeframe"], 2)["close"].iloc[-1])
        except Exception:  # noqa
            px = r["entry_price"]
        eq += r["units"] * px
    return capital, eq, npos


def run_live(conn, cfg):
    """Бесконечный цикл живой торговли. Ctrl+C для остановки."""
    interval = cfg.get("live", {}).get("interval_seconds", 300)
    print(f"Живой пейпер запущен. Интервал {interval}с. Ctrl+C для остановки.")
    print("Данные: data-api.binance.vision (работает при VPN). Реальных денег НЕТ.\n")
    try:
        while True:
            try:
                tick(conn, cfg)
            except Exception as e:  # noqa
                print(f"  [ошибка тика] {type(e).__name__}: {e}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nОстановлено. Состояние сохранено в SQLite.")
