"""
Бумажная торговля. Принцип из треда: "Сначала демо-счёт, минимум пару месяцев,
прежде чем вкладывать реальные деньги. Бумага игнорирует проскальзывание и
спред — мы их учитываем."

Здесь — ПОРТФЕЛЬНАЯ форвард-симуляция: берём агентов, продвинутых супервизором,
и торгуем ими совместно на out-of-sample данных (которые не использовались при
отборе). Это и есть честный форвард-тест перед реальными деньгами.

Общий капитал, общий лимит позиций, риск-менеджмент на каждой позиции.
Все сделки пишутся в SQLite (отслеживаемость). LLM здесь не участвует —
торговля 100% детерминированная.
"""
import json
import pandas as pd

from . import db
from . import backtest as bt
from . import genome as gn
from . import risk as rk
from . import macro_feed
from . import indicators as ind


def _macro_blocks_entries(cfg) -> bool:
    """Макро-страж: при сильном оттоке из ETF блокируем НОВЫЕ входы."""
    mc = cfg.get("macro", {})
    if not mc.get("enabled"):
        return False
    info = macro_feed.etf_flow_bias(
        mc.get("asset", "BTC"), mc.get("lookback_days", 5),
        mc.get("block_threshold_musd", 0))
    print(f"  Макро-страж: {info.get('note', info)}")
    blocked = info["bias"] == "risk_off"
    if blocked:
        print("  [!] risk_off — новые входы заблокированы (выходы работают как обычно).")
    return blocked


def run_paper(conn, cfg, data_by_symbol):
    promoted = db.get_agents(conn, "promoted")
    if not promoted:
        print("Нет продвинутых агентов. Запусти эволюцию и супервизора.")
        return

    macro_block = _macro_blocks_entries(cfg)
    capital = float(cfg["paper"]["starting_capital"])
    risk_cfg = cfg["risk"]
    fee = cfg["costs"]["fee_pct"]
    slip = cfg["costs"]["slippage_pct"]

    allow_short = risk_cfg.get("allow_short", False)
    atr_period = risk_cfg.get("atr_period", 14)
    use_atr = risk_cfg.get("atr_stop", False)

    # Готовим для каждого агента: out-of-sample df + предрассчитанный сигнал + ATR.
    streams = []
    for a in promoted:
        sym = a["symbol"]
        if sym not in data_by_symbol:
            continue
        _, test_df = bt.split_train_test(data_by_symbol[sym], cfg["train_ratio"])
        g = json.loads(a["genome"])
        delay = cfg.get("execution", {}).get("signal_delay_bars", 1)
        sig = gn.signal(g, test_df, allow_short).shift(delay).fillna(0).astype(int)
        atr_ser = ind.atr(test_df, atr_period) if use_atr else None
        streams.append({"agent": a, "df": test_df, "sig": sig, "g": g, "atr": atr_ser})

    if not streams:
        print("Нет данных для продвинутых агентов.")
        return

    # Единая временная шкала (объединение индексов всех потоков).
    timeline = sorted(set().union(*[set(s["df"].index) for s in streams]))
    open_positions = {}     # agent_id -> Position
    cooldown_left = {}      # agent_id -> сколько баров ещё «отдыхать» после выхода
    realized_pnl = 0.0
    trade_count = 0
    wins = 0

    # Портфельный стоп-кран (правило из чата: просадка >5% → стоп новых входов).
    dd_limit = risk_cfg.get("max_portfolio_drawdown", 1.0)
    peak_equity = capital
    last_price = {}
    dd_halt_bars = 0

    print(f"\nБумажная торговля: {len(streams)} агентов, "
          f"капитал {capital:.0f} USDT, {len(timeline)} баров...")

    for ts in timeline:
        for s in streams:
            a = s["agent"]
            aid = a["id"]
            if ts not in s["df"].index:
                continue
            price = float(s["df"].loc[ts, "close"])
            last_price[a["symbol"]] = price
            pos = open_positions.get(aid)

            # Текущий капитал портфеля (mark-to-market) и просадка.
            equity = capital + sum(
                p.value(last_price.get(p.symbol, p.entry_price))
                for p in open_positions.values())
            peak_equity = max(peak_equity, equity)
            dd_halt = (peak_equity - equity) / peak_equity > dd_limit if peak_equity else False
            if dd_halt:
                dd_halt_bars += 1

            sig_now = int(s["sig"].loc[ts])

            # 1. Управление открытой позицией (риск приоритетнее сигнала).
            if pos is not None:
                should_exit, reason, exit_price = pos.exit_check_hl(price, price, price, risk_cfg)
                if not should_exit and sig_now != pos.direction:
                    should_exit, reason, exit_price = True, "signal", price
                if should_exit:
                    fill = exit_price * (1 - slip * pos.direction)
                    pnl = rk.close_pnl(pos, fill, fee)
                    capital += pos.notional + pnl
                    realized_pnl += pnl
                    trade_count += 1
                    wins += 1 if pnl > 0 else 0
                    side = "SELL" if pos.direction == 1 else "COVER"
                    db.log_paper_trade(conn, aid, a["symbol"], side,
                                       fill, pos.units, pos.units * fill * fee,
                                       round(pnl, 2), reason)
                    del open_positions[aid]
                    cooldown_left[aid] = int(s["g"].get("cooldown", 0))
            elif aid not in open_positions and cooldown_left.get(aid, 0) > 0:
                cooldown_left[aid] -= 1   # «отдыхаем» после выхода

            # 2. Вход (long или short): лимит, макро-страж, стоп-кран И кулдаун.
            if aid not in open_positions and (not macro_block) and (not dd_halt) \
                    and sig_now != 0 and cooldown_left.get(aid, 0) <= 0 \
                    and rk.can_open(len(open_positions), risk_cfg):
                invest = rk.position_size(capital, risk_cfg)
                if invest <= 0 or invest > capital:
                    continue
                g = s["g"]
                fill = price * (1 + slip * sig_now)
                units = invest / fill
                atr_val = float(s["atr"].loc[ts]) if s["atr"] is not None and ts in s["atr"].index else None
                take_mult = (g["stop_atr"] * g["rr"]) if g.get("stop_atr") and g.get("rr") else None
                capital -= invest
                open_positions[aid] = rk.Position(
                    aid, a["symbol"], fill, units, direction=sig_now, notional=invest,
                    atr=atr_val, stop_mult=g.get("stop_atr"), take_mult=take_mult,
                    trail_mult=g.get("trail_atr"))
                side = "BUY" if sig_now == 1 else "SHORT"
                db.log_paper_trade(conn, aid, a["symbol"], side,
                                   fill, units, invest * fee, None, "signal")

    # Закрываем остатки по последней цене (mark-to-market, long и short).
    for aid, pos in list(open_positions.items()):
        px = float(streams[0]["df"]["close"].iloc[-1])
        for s in streams:
            if s["agent"]["id"] == aid:
                px = float(s["df"]["close"].iloc[-1])
        capital += pos.value(px)

    start = float(cfg["paper"]["starting_capital"])
    total_ret = capital / start - 1
    wr = wins / trade_count if trade_count else 0
    print("\n--- Результат бумажной торговли (out-of-sample) ---")
    print(f"  Стартовый капитал:  {start:,.0f} USDT")
    print(f"  Итоговый капитал:   {capital:,.2f} USDT")
    print(f"  Доходность:         {total_ret:+.2%}")
    print(f"  Реализованный PnL:  {realized_pnl:+,.2f} USDT")
    print(f"  Сделок:             {trade_count} | Win rate: {wr:.1%}")
    if dd_halt_bars:
        print(f"  Стоп-кран просадки сработал на {dd_halt_bars} барах "
              f"(блокировал новые входы при просадке >{dd_limit:.0%})")
    print("\n  Напоминание из треда: это форвард-тест на данных, которых агенты")
    print("  не видели при отборе. Прежде чем рисковать реальными деньгами —")
    print("  гоняй демо НЕДЕЛИ. Бэктест и даже бумага не учитывают дрейф рынка.")
    return {"start": start, "final": capital, "return": total_ret,
            "trades": trade_count, "win_rate": wr, "dd_halt_bars": dd_halt_bars}
