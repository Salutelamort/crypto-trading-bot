"""
ЕЖЕДНЕВНОЕ АВТООБУЧЕНИЕ (запускается роботом GitHub Actions, без участия человека).

Раз в день:
  1. качает свежие цены;
  2. эволюционирует и отбирает стратегии (супервизор);
  3. делает один "тик" живой бумажной торговли;
  4. дописывает строку в журнал TRACK_RECORD.csv — честная история на реальном рынке.

За ~90 запусков (3 месяца) накапливается forward-track-record на ранее невиданных
данных. По нему ЧЕЛОВЕК потом решает, допускать ли стратегию к реальным деньгам.
Автоматического перехода на реальные деньги НЕТ — это сознательная мера безопасности.
"""
import os
import csv
import datetime
import yaml

from src import db, data_feed as feed
from src import evolution, supervisor, live_trade, macro_feed

CSV_PATH = "TRACK_RECORD.csv"
HEADER = ["date", "equity", "capital", "open_positions",
          "candidates", "promoted", "best_test_sharpe", "macro_bias"]


def main():
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    conn = db.connect(cfg["db_path"])

    print("== Ежедневное автообучение ==")
    # 1. свежие данные
    for sym in cfg["symbols"]:
        feed.fetch_ohlcv(conn, sym, cfg["timeframe"], cfg["history_days"])
    data = {s: feed.load_ohlcv(conn, s, cfg["timeframe"]) for s in cfg["symbols"]}
    data = {s: d for s, d in data.items() if not d.empty}

    # 2. эволюция + отбор
    evolution.evolve(conn, cfg, data)
    supervisor.supervise(conn, cfg)

    # 3. тик живой бумажной торговли
    live_trade.tick(conn, cfg)

    # 4. запись в журнал
    capital, equity, npos = live_trade.account_equity(conn, cfg)
    cand = len(db.get_agents(conn, "candidate"))
    prom = len(db.get_agents(conn, "promoted"))
    row_best = conn.execute(
        "SELECT MAX(test_sharpe) m FROM agents WHERE status IN ('candidate','promoted')"
    ).fetchone()["m"]
    try:
        bias = macro_feed.etf_flow_bias(cfg.get("macro", {}).get("asset", "BTC"))["bias"]
    except Exception:  # noqa
        bias = "n/a"

    row = [datetime.date.today().isoformat(), round(equity, 2), round(capital, 2),
           npos, cand, prom, round(row_best, 3) if row_best is not None else "",
           bias]

    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(HEADER)
        w.writerow(row)

    print(f"\nЗаписано в {CSV_PATH}: {dict(zip(HEADER, row))}")
    start = cfg["paper"]["starting_capital"]
    print(f"Итог дня: капитал {equity:,.2f} ({equity/start-1:+.2%}), "
          f"продвинуто {prom}, кандидатов {cand}")
    conn.close()


if __name__ == "__main__":
    main()
