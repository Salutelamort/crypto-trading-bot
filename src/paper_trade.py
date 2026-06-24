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

    # Готовим для каждого агента: out-of-sample df + предрассчитанный сигнал.
    streams = []
    for a in promoted:
        sym = a["symbol"]
        if sym not in data_by_symbol:
            continue
        _, test_df = bt.split_train_test(data_by_symbol[sym], cfg["train_ratio"])
        g = json.loads(a["genome"])
        sig = gn.signal(g, test_df)
        streams.append({"agent": a, "df": test_df, "sig": sig, "g": g})

    if not streams:
        print("Нет данных для продвинутых агентов.")
        return

    # Единая временная шкала (объединение индексов всех потоков).
    timeline = sorted(set().union(*[set(s["df"].index) for s in streams]))
    open_positions = {}     # agent_id -> Position
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
                p.units * last_price.get(p.symbol, p.entry_price)
                for p in open_positions.values())
            peak_equity = max(peak_equity, equity)
            dd_halt = (peak_equity - equity) / peak_equity > dd_limit if peak_equity else False
            if dd_halt:
                dd_halt_bars += 1

            # 1. Управление открытой позицией (риск приоритетнее сигнала).
            if pos is not None:
                should_exit, reason = pos.exit_check(price, risk_cfg)
                if not should_exit and int(s["sig"].loc[ts]) == 0:
                    should_exit, reason = True, "signal"
                if should_exit:
                    fill = price * (1 - slip)
                    proceeds = pos.units * fill * (1 - fee)
                    cost_basis = pos.units * pos.entry_price
                    pnl = proceeds - cost_basis
                    capital += proceeds
                    realized_pnl += pnl
                    trade_count += 1
                    wins += 1 if pnl > 0 else 0
                    db.log_paper_trade(conn, aid, a["symbol"], "SELL",
                                       fill, pos.units, pos.units * fill * fee,
                                       round(pnl, 2), reason)
                    del open_positions[aid]

            # 2. Вход: с учётом лимита позиций, макро-стража И стоп-крана просадки.
            elif (not macro_block) and (not dd_halt) and int(s["sig"].loc[ts]) == 1 \
                    and rk.can_open(len(open_positions), risk_cfg):
                invest = rk.position_size(capital, risk_cfg)
                if invest <= 0 or invest > capital:
                    continue
                fill = price * (1 + slip)
                units = (invest * (1 - fee)) / fill
                capital -= invest
                open_positions[aid] = rk.Position(aid, a["symbol"], fill, units)
                db.log_paper_trade(conn, aid, a["symbol"], "BUY",
                                   fill, units, invest * fee, None, "signal")

    # Закрываем остатки по последней цене (mark-to-market).
    for aid, pos in list(open_positions.items()):
        last_price = float(streams[0]["df"]["close"].iloc[-1])
        for s in streams:
            if s["agent"]["id"] == aid:
                last_price = float(s["df"]["close"].iloc[-1])
        capital += pos.units * last_price

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
