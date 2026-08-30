"""
ЖИВОЙ бумажный трейдинг в реальном времени — БЕЗ биржи и без ключей.

Почему так (а не Binance Testnet): тестнет Binance геоблокирован (451) с машины
пользователя. Поэтому исполнение считаем локально, а цены берём с публичного
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
import json
import time
from datetime import datetime, timedelta, timezone

from . import data_feed as feed
from . import db, macro_feed, news_feed, protections
from . import genome as gn
from . import indicators as ind
from . import risk as rk
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
        keys = r.keys()
        direction = r["direction"] if "direction" in keys and r["direction"] else 1
        notional = r["notional"] if "notional" in keys and r["notional"] else r["units"] * r["entry_price"]
        atr = r["atr"] if "atr" in keys else None
        p = rk.Position(r["agent_id"], r["symbol"], r["entry_price"], r["units"],
                        direction=direction, notional=notional, atr=atr,
                        entry_fee_paid=bool(r["entry_fee_paid"]) if "entry_fee_paid" in keys else False)
        p.peak_price = r["peak_price"]
        p.opened_at = r["opened_at"] if "opened_at" in keys else None
        p.experiment_id = r["experiment_id"] if "experiment_id" in keys else "legacy"
        p.code_sha = r["code_sha"] if "code_sha" in keys else None
        p.config_hash = r["config_hash"] if "config_hash" in keys else None
        p.mark_price = r["mark_price"] if "mark_price" in keys else r["entry_price"]
        pos[r["agent_id"]] = p
    return pos


def _save_position(conn, p):
    # opened_at сохраняем РАЗ (при открытии), при последующих сохранениях не
    # затираем — иначе «возраст позиции» врёт и любая логика по времени ломается.
    opened = getattr(p, "opened_at", None) or now_iso()
    provenance = db.current_provenance(conn)
    experiment_id = getattr(p, "experiment_id", None) or provenance["experiment_id"]
    code_sha = getattr(p, "code_sha", provenance["code_sha"])
    config_hash = getattr(p, "config_hash", provenance["config_hash"])
    p.experiment_id, p.code_sha, p.config_hash = experiment_id, code_sha, config_hash
    conn.execute(
        "INSERT OR REPLACE INTO live_positions "
        "(agent_id,symbol,entry_price,units,peak_price,opened_at,direction,notional,atr,"
        "experiment_id,code_sha,config_hash,last_checked_at,mark_price,entry_fee_paid) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (p.agent_id, p.symbol, p.entry_price, p.units, p.peak_price, opened,
         p.direction, p.notional, p.atr, experiment_id, code_sha, config_hash,
         now_iso(), getattr(p, "mark_price", p.entry_price), int(p.entry_fee_paid)))
    conn.commit()


def _del_position(conn, agent_id):
    conn.execute("DELETE FROM live_positions WHERE agent_id=?", (agent_id,))
    conn.commit()


def _position_provenance(position):
    return {"experiment_id": getattr(position, "experiment_id", "legacy"),
            "code_sha": getattr(position, "code_sha", None),
            "config_hash": getattr(position, "config_hash", None)}


def _position_mode(position):
    return "legacy" if getattr(position, "experiment_id", "legacy") == "legacy" else "live"


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
    db.ensure_experiment(conn, cfg)
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
    macro_unavailable = False
    mc = cfg.get("macro", {})
    if mc.get("enabled"):
        try:
            info = macro_feed.etf_flow_bias(mc.get("asset", "BTC"),
                                            mc.get("lookback_days", 5),
                                            mc.get("block_threshold_musd", 0))
            macro_block = info["bias"] == "risk_off"
            macro_unavailable = not info.get("available", False)
            if macro_unavailable and mc.get("fail_closed", False):
                macro_block = True
        except Exception:  # noqa
            macro_unavailable = True
            macro_block = bool(mc.get("fail_closed", False))

    # новостной страж (индекс страха/жадности + негативные катализаторы)
    news_block = False
    news_reason = ""
    if cfg.get("news", {}).get("enabled"):
        try:
            ng = news_feed.news_gate(cfg)
            news_block = ng["block"]
            news_reason = ng["reason"]
        except Exception:  # noqa
            pass

    # ЗАЩИТЫ (Freqtrade-стиль): пауза после серии убытков + блок плохих символов
    guard_block, guard_reason = False, ""
    locked = set()
    try:
        guard_block, guard_reason = protections.stoploss_guard(conn, cfg)
        locked = protections.locked_symbols(conn, cfg)
    except Exception:  # noqa
        pass

    positions = _load_positions(conn)
    interval = cfg.get("live", {}).get("interval_seconds", 300)
    last_tick_raw = db.get_runtime_state(conn, "last_tick_at")
    try:
        last_tick = datetime.fromisoformat(last_tick_raw) if last_tick_raw else None
        if last_tick and last_tick.tzinfo is None:
            last_tick = last_tick.replace(tzinfo=timezone.utc)
    except ValueError:
        last_tick = None
    if last_tick is None:
        last_tick = datetime.now(timezone.utc) - timedelta(seconds=interval)
    max_minutes = int(cfg.get("live", {}).get("max_catchup_minutes", 10080))
    catchup_truncated = (datetime.now(timezone.utc) - last_tick).total_seconds() > max_minutes * 60
    minute_cache = {}

    def minute_hl(sym, fallback_price):
        """High/Low по 1m-свечам с прошлого тика — для внутрибарного стопа."""
        if sym not in minute_cache:
            try:
                start_ms = int(last_tick.timestamp() * 1000) - 60_000
                md = feed.fetch_since(sym, "1m", start_ms, max_bars=max_minutes)
                if md.empty:
                    raise RuntimeError("нет минутных баров")
                minute_cache[sym] = (float(md["high"].max()), float(md["low"].min()))
            except Exception:  # noqa
                minute_cache[sym] = (fallback_price, fallback_price)
        return minute_cache[sym]

    # свежие данные по уникальным парам (символ × таймфрейм) — мультитаймфрейм
    pairs = {(a["symbol"], a["timeframe"]) for a in agents}
    data = {}            # (sym, tf) -> DataFrame
    last_close = {}      # sym -> последняя цена (для mark-to-market позиций)
    for s, tf in pairs:
        try:
            df = feed.fetch_recent(s, tf, 400)
            data[(s, tf)] = df
            last_close[s] = float(df["close"].iloc[-1])
        except Exception as e:  # noqa
            if verbose:
                print(f"  [!] нет данных {s} {tf}: {e}")

    allow_short = risk_cfg.get("allow_short", False)
    actions = []

    # РАСШИВКА КОНЦЕНТРАЦИИ: входной гейт не пускает новые дубли по символу, но
    # уже открытые (легаси / занесённые иначе) ничем не вычищаются и зря держат
    # слоты max_open_positions. На каждом тике приводим состояние к инварианту
    # max_positions_per_symbol: лишние позиции по символу закрываем по рынку,
    # оставляя агента с лучшим OOS Sharpe.
    sym_limit = risk_cfg.get("max_positions_per_symbol", 99)
    sharpe_by_id = {a["id"]: (a["test_sharpe"] if a["test_sharpe"] is not None else -99)
                    for a in agents}
    by_symbol = {}
    for aid, p in positions.items():
        by_symbol.setdefault(p.symbol, []).append(aid)
    for sym, aids in by_symbol.items():
        if len(aids) <= sym_limit or sym not in last_close:
            continue
        ranked = sorted(aids, key=lambda i: sharpe_by_id.get(i, -99), reverse=True)
        for aid in ranked[sym_limit:]:               # всё сверх лимита — закрыть
            pos = positions[aid]
            fill = last_close[sym] * (1 - slip * pos.direction)
            pnl = rk.close_pnl(pos, fill, fee)
            capital += pos.notional + pnl
            side = "SELL" if pos.direction == 1 else "COVER"
            db.log_paper_trade(conn, aid, sym, side, fill, pos.units,
                               pos.units * fill * fee, round(pnl, 2), "deconcentrate",
                               mode=_position_mode(pos), provenance=_position_provenance(pos))
            _del_position(conn, aid)
            del positions[aid]
            actions.append(f"{side} #{aid} {sym} @ {fill:.2f} (расшивка дубля) PnL {pnl:+.2f}")

    # текущий капитал и просадка (mark-to-market, работает для long и short)
    def equity_now():
        eq = capital
        for p in positions.values():
            if p.symbol in last_close:
                eq += p.value(last_close[p.symbol])
        return eq

    eq = equity_now()
    peak = max(peak, eq)
    dd = (peak - eq) / peak if peak else 0.0
    dd_halt = dd > dd_limit

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")

    for a in agents:
        aid, sym = a["id"], a["symbol"]
        g = json.loads(a["genome"])
        key = (sym, g["timeframe"])
        if key not in data:
            continue
        df = data[key]
        price = float(df["close"].iloc[-1])
        # реагируем на сигнал УЖЕ ЗАКРЫТОГО бара (не на текущий, формирующийся) —
        # не зависим от скорости доступа к бирже.
        delay = cfg.get("execution", {}).get("signal_delay_bars", 1)
        sig = int(gn.signal(g, df, allow_short).shift(delay).fillna(0).iloc[-1])
        pos = positions.get(aid)
        # сколько позиций уже открыто по этой монете (анти-концентрация на исполнении)
        sym_count = sum(1 for p in positions.values() if p.symbol == sym)

        # 1. управление позицией — внутрибарно по 1m свечам (стоп как реальный ордер)
        if pos is not None:
            pos.mark_price = price
            hi, lo = minute_hl(sym, price)
            should_exit, reason, exit_price = pos.exit_check_hl(hi, lo, price, risk_cfg)
            if not should_exit and sig != pos.direction:  # сигнал ушёл/развернулся
                should_exit, reason, exit_price = True, "signal", price
            if should_exit:
                fill = exit_price * (1 - slip * pos.direction)
                pnl = rk.close_pnl(pos, fill, fee)
                capital += pos.notional + pnl
                side = "SELL" if pos.direction == 1 else "COVER"
                db.log_paper_trade(conn, aid, sym, side, fill, pos.units,
                                   pos.units * fill * fee, round(pnl, 2), reason,
                                   mode=_position_mode(pos), provenance=_position_provenance(pos))
                _del_position(conn, aid)
                del positions[aid]
                actions.append(f"{side} #{aid} {sym} @ {fill:.2f} ({reason}) PnL {pnl:+.2f}")
            else:
                _save_position(conn, pos)  # сохранить обновлённый extreme

        # 2. вход по сигналу (long или short)
        elif sig != 0 and not macro_block and not news_block and not dd_halt \
                and not guard_block and sym not in locked \
                and sym_count < risk_cfg.get("max_positions_per_symbol", 99) \
                and rk.can_open(len(positions), risk_cfg):
            atr_val = float(ind.atr(df, risk_cfg.get("atr_period", 14)).iloc[-1]) \
                if risk_cfg.get("atr_stop") else None
            # волатильность-таргетинг: размер от риска до стопа (стоп — ген агента)
            eff_risk = dict(risk_cfg)
            if g.get("stop_atr"):
                eff_risk["atr_stop_mult"] = g["stop_atr"]
            invest = rk.position_size(capital, eff_risk, atr_val, price)
            invest = min(invest, capital / (1 + fee))
            if 0 < invest <= capital:
                fill = price * (1 + slip * sig)
                units = invest / fill
                rr = risk_cfg.get("fixed_rr", g.get("rr"))
                take_mult = (g["stop_atr"] * rr) if g.get("stop_atr") and rr else None
                entry_fee = invest * fee
                capital -= invest + entry_fee
                p = rk.Position(aid, sym, fill, units, direction=sig,
                                notional=invest, atr=atr_val,
                                stop_mult=g.get("stop_atr"), take_mult=take_mult,
                                trail_mult=g.get("trail_atr"), entry_fee_paid=True)
                p.opened_at = now_iso()
                p.mark_price = price
                positions[aid] = p
                _save_position(conn, p)
                side = "BUY" if sig == 1 else "SHORT"
                db.log_paper_trade(conn, aid, sym, side, fill, units,
                                   entry_fee, None, "signal")
                actions.append(f"{side} #{aid} {sym} @ {fill:.2f} (вложено {invest:.2f})")

    eq = equity_now()
    peak = max(peak, eq)
    _save_account(conn, capital, peak)
    db.set_runtime_state(conn, "last_tick_at", now_iso())

    if verbose:
        flags = []
        if demo:
            flags.append("ДЕМО: агенты НЕ прошли отбор")
        if macro_block:
            flags.append("макро risk_off — входы стоп")
        elif macro_unavailable:
            flags.append("макро недоступно — входы разрешены по fail-open")
        if catchup_truncated:
            flags.append(f"минутная история ограничена {max_minutes} мин")
        if news_block:
            flags.append(f"новости: {news_reason} — входы стоп")
        if dd_halt:
            flags.append(f"стоп-кран просадки {dd:.1%}")
        if guard_block:
            flags.append(guard_reason)
        if locked:
            flags.append(f"заблокированы символы: {', '.join(sorted(locked))}")
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
    for r in conn.execute(
            "SELECT p.*, a.timeframe position_timeframe FROM live_positions p "
            "LEFT JOIN agents a ON a.id=p.agent_id").fetchall():
        npos += 1
        try:
            timeframe = r["position_timeframe"] or cfg["timeframe"]
            px = float(feed.fetch_recent(r["symbol"], timeframe, 2)["close"].iloc[-1])
        except Exception:  # noqa
            px = r["mark_price"] or r["entry_price"]
        direction = r["direction"] or 1
        notional = r["notional"] or r["units"] * r["entry_price"]
        eq += notional * (1 + direction * (px / r["entry_price"] - 1))
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
