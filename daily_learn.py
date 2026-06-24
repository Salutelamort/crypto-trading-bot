"""
АВТООБУЧЕНИЕ (запускается роботом GitHub Actions каждые 15 минут, без человека).

За один прогон — "пачка" циклов в пределах бюджета времени:
  1. качает свежие цены (один раз);
  2. эволюционирует стратегии (рождение/смерть агентов) — несколько циклов подряд;
  3. отбор + тик живой бумажной торговли;
  4. дописывает строку с временной меткой в журнал TRACK_RECORD.csv.

Популяция сохраняется в БД между прогонами, поэтому эволюция продолжается
непрерывно. За 3 месяца накапливается forward-track-record на невиданных данных.
По нему ЧЕЛОВЕК потом решает про реальные деньги. Автоперехода НЕТ — мера безопасности.
"""
import os
import csv
import time
import datetime
import yaml

from src import db, data_feed as feed
from src import evolution, supervisor, live_trade, macro_feed, news_feed

CSV_PATH = "TRACK_RECORD.csv"
HEADER = ["date", "equity", "capital", "open_positions",
          "candidates", "promoted", "best_test_sharpe", "macro_bias",
          "fear_greed", "news_hits"]


def _journal(conn, cfg):
    """Записывает строку в журнал (только чтение метрик, без побочных действий).
    best = лучший найденный OOS Sharpe среди ВСЕХ агентов (показывает прогресс поиска)."""
    capital, equity, npos = live_trade.account_equity(conn, cfg)
    cand = len(db.get_agents(conn, "candidate"))
    prom = len(db.get_agents(conn, "promoted"))
    best = conn.execute("SELECT MAX(test_sharpe) m FROM agents").fetchone()["m"]
    try:
        bias = macro_feed.etf_flow_bias(cfg.get("macro", {}).get("asset", "BTC"))["bias"]
    except Exception:  # noqa
        bias = "n/a"
    try:
        ng = news_feed.news_gate(cfg)
        fng, nhits = ng["fng"]["value"], ng["news_hits"]
    except Exception:  # noqa
        fng, nhits = "", ""

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
    row = [ts, round(equity, 2), round(capital, 2), npos, cand, prom,
           round(best, 3) if best is not None else "", bias, fng, nhits]
    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(HEADER)
        w.writerow(row)
    print(f"[{ts}] лучший Sharpe {best if best is not None else '—'} | "
          f"кандидатов {cand} | продвинуто {prom} | капитал {equity:,.0f}")


def main():
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    conn = db.connect(cfg["db_path"])

    # бюджет времени на один облачный прогон (минут). Репо публичный → минуты
    # бесплатны, поэтому за один слот делаем несколько циклов эволюции "пачкой".
    budget_min = float(os.environ.get("LEARN_BUDGET_MIN", "8"))
    deadline = time.time() + budget_min * 60

    print(f"== Автообучение (пачка, бюджет {budget_min} мин) ==")
    for sym in cfg["symbols"]:
        feed.fetch_ohlcv(conn, sym, cfg["timeframe"], cfg["history_days"])
    data = {s: feed.load_ohlcv(conn, s, cfg["timeframe"]) for s in cfg["symbols"]}
    data = {s: d for s, d in data.items() if not d.empty}

    # Цикл: эволюция накапливает и улучшает популяцию (рождение/смерть внутри неё).
    cycles = 0
    while True:
        evolution.evolve(conn, cfg, data)
        _journal(conn, cfg)
        cycles += 1
        if time.time() >= deadline:
            break

    # Допуск к живой торговле (без убийства) + один тик бумажной торговли.
    supervisor.supervise(conn, cfg)
    live_trade.tick(conn, cfg)
    _journal(conn, cfg)
    print(f"\nГотово: циклов эволюции за прогон — {cycles}")
    conn.close()


if __name__ == "__main__":
    main()
