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
import csv
import datetime
import json
import os
import time

import yaml

from src import data_feed as feed
from src import db, evolution, live_trade, macro_feed, news_feed, supervisor, execution_report

CSV_PATH = "TRACK_RECORD.csv"
HEADER = ["date", "equity", "capital", "open_positions",
          "candidates", "promoted", "best_test_sharpe", "macro_bias",
          "fear_greed", "news_hits"]


def _write_state_summary(conn, cfg, cycles, snapshot):
    """Маленький отслеживаемый снимок; тяжёлая SQLite живёт в release-asset."""
    quality = execution_report.build(conn)
    payload = {
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "experiment_id": db.get_runtime_state(conn, "current_experiment", "legacy"),
        **snapshot, "cycles": cycles,
        "execution_health": quality["health"],
        "experiment_metrics": quality["current"],
        "cash_reconciliation": quality["reconciliation"],
    }
    os.makedirs("state", exist_ok=True)
    tmp = "state/latest.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, "state/latest.json")


def _journal(conn, cfg):
    """Записывает строку в журнал (только чтение метрик, без побочных действий).
    best = лучший найденный OOS Sharpe среди ВСЕХ агентов (показывает прогресс поиска)."""
    capital, equity, npos = live_trade.account_equity(conn, cfg)
    cand = len(db.get_agents(conn, "candidate"))
    prom = len(db.get_agents(conn, "promoted"))
    best = db.best_sharpe_ever(conn)   # из agent_stats — переживает прунинг истории
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
    return {"equity": round(equity, 2), "free_cash": round(capital, 2),
            "open_positions": npos, "candidates": cand, "promoted": prom,
            "best_test_sharpe": best}


def main():
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    conn = db.connect(cfg["db_path"])

    # бюджет времени на один облачный прогон (минут). Репо публичный → минуты
    # бесплатны, поэтому за один слот делаем несколько циклов эволюции "пачкой".
    budget_min = float(os.environ.get("LEARN_BUDGET_MIN", "8"))
    deadline = time.time() + budget_min * 60

    print(f"== Автообучение (пачка, бюджет {budget_min} мин) ==")
    timeframes = cfg.get("timeframes", [cfg["timeframe"]])
    feed.fetch_all(conn, cfg["symbols"], timeframes, cfg["history_days"])
    data = feed.load_all(conn, cfg["symbols"], timeframes)   # ключ (symbol, timeframe)

    # Цикл: эволюция накапливает и улучшает популяцию (рождение/смерть внутри неё).
    # Журнал пишем НЕ каждый цикл (это плодило десятки почти одинаковых строк за
    # прогон), а один раз в конце прогона — после тика.
    cycles = 0
    while True:
        evolution.evolve(conn, cfg, data)
        cycles += 1
        if time.time() >= deadline:
            break

    # Гигиена пула: освежить OOS-метрики допущенных (иначе они замирают на
    # момент допуска), затем отбор с демоцией + один тик бумажной торговли.
    evolution.reevaluate_promoted(conn, cfg, data)
    supervisor.supervise(conn, cfg)
    live_trade.tick(conn, cfg)
    snapshot = _journal(conn, cfg)
    _write_state_summary(conn, cfg, cycles, snapshot)
    print(f"\nГотово: циклов эволюции за прогон — {cycles}")

    # RETENTION — база теперь хранится сжатым release-asset, не в git-истории:
    #  - свечи сохраняем для инкрементального обновления, старый хвост подрезаем;
    #  - мёртвых агентов и старые решения старше 7 дней сворачиваем в agent_stats
    #    (компактная память об испытаниях) и удаляем сырьё.
    # Память «что работает / что нет» при этом НЕ теряется — она в agent_stats.
    cutoff_ms = int((time.time() - (cfg["history_days"] + 7) * 86400) * 1000)
    conn.execute("DELETE FROM candles WHERE open_time < ?", (cutoff_ms,))
    conn.commit()
    da, dd = db.prune_history(conn, keep_killed=3000, keep_decisions=8000)
    print(f"Retention: убрано killed-агентов {da}, старых решений {dd}")
    conn.execute("PRAGMA optimize")
    conn.close()


if __name__ == "__main__":
    main()
