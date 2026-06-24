"""
A/B-эксперимент: сравнение соотношений риск/прибыль (R:R).

Чат настойчиво советует R:R >= 3, у нас по умолчанию 2:1. Этот скрипт честно
проверяет оба варианта: для каждого take_profit прогоняет ПОЛНЫЙ конвейер
(эволюция -> супервизор -> бумага) с одинаковым seed, в отдельной временной базе,
и сравнивает результаты.

Принцип из треда: не верить мнению ("надо R:R 3"), а измерять на данных.
Запуск:  python experiment_rr.py
"""
import os
import copy
import statistics
import yaml

from src import db, data_feed as feed, evolution, supervisor, paper_trade

SCRATCH = os.environ.get("TEMP", ".")

# Варианты: (название, take_profit_pct). stop_loss_pct = 0.03 в обоих → R:R = tp/sl.
VARIANTS = [
    ("R:R 2:1", 0.06),
    ("R:R 3:1", 0.09),
]


def agent_stats(conn):
    """Сводка по живым/продвинутым агентам после прогона."""
    rows = conn.execute(
        "SELECT test_sharpe, test_return, consistency, status FROM agents "
        "WHERE status IN ('candidate','promoted')").fetchall()
    sharpes = [r["test_sharpe"] for r in rows if r["test_sharpe"] is not None]
    rets = [r["test_return"] for r in rows if r["test_return"] is not None]
    return {
        "alive": len(rows),
        "best_sharpe": round(max(sharpes), 2) if sharpes else None,
        "median_sharpe": round(statistics.median(sharpes), 2) if sharpes else None,
        "best_return": round(max(rets), 4) if rets else None,
        "positive_sharpe": sum(1 for s in sharpes if s > 0),
    }


def run_variant(name, take_profit, base_cfg, data):
    cfg = copy.deepcopy(base_cfg)
    cfg["risk"]["take_profit_pct"] = take_profit
    cfg["macro"]["enabled"] = False   # выключаем сетевой фильтр для чистоты сравнения
    cfg["db_path"] = os.path.join(SCRATCH, f"rr_{int(take_profit*100)}.db")

    if os.path.exists(cfg["db_path"]):
        os.remove(cfg["db_path"])
    conn = db.connect(cfg["db_path"])

    print(f"\n{'='*60}\n  ВАРИАНТ: {name}  (take_profit={take_profit}, stop=0.03)\n{'='*60}")
    evolution.evolve(conn, cfg, data)
    sup = supervisor.supervise(conn, cfg)
    paper = paper_trade.run_paper(conn, cfg, data)

    res = {"name": name, "take_profit": take_profit,
           "promoted": sup.get("promote", 0), "agents": agent_stats(conn), "paper": paper}
    conn.close()
    return res


def main():
    base_cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    # данные грузим ОДИН раз из основной базы (кэш свечей общий)
    main_conn = db.connect(base_cfg["db_path"])
    data = {}
    for sym in base_cfg["symbols"]:
        df = feed.load_ohlcv(main_conn, sym, base_cfg["timeframe"])
        if df.empty:
            print(f"[!] Нет данных {sym}. Сначала: python main.py fetch")
            return
        data[sym] = df
    main_conn.close()

    results = [run_variant(n, tp, base_cfg, data) for n, tp in VARIANTS]

    # --- итоговая таблица ---
    print(f"\n\n{'#'*64}\n  СРАВНЕНИЕ R:R\n{'#'*64}")
    hdr = f"{'метрика':28s}" + "".join(f"{r['name']:>16s}" for r in results)
    print(hdr); print("-" * len(hdr))

    def row(label, fn):
        print(f"{label:28s}" + "".join(f"{fn(r):>16s}" for r in results))

    row("Живых агентов", lambda r: str(r["agents"]["alive"]))
    row("С положит. Sharpe", lambda r: str(r["agents"]["positive_sharpe"]))
    row("Лучший test Sharpe", lambda r: str(r["agents"]["best_sharpe"]))
    row("Медианный test Sharpe", lambda r: str(r["agents"]["median_sharpe"]))
    row("Лучшая test доходность", lambda r: f"{(r['agents']['best_return'] or 0):.1%}")
    row("Продвинуто в live", lambda r: str(r["promoted"]))
    row("Бумага: доходность",
        lambda r: f"{r['paper']['return']:+.2%}" if r["paper"] else "—")
    row("Бумага: сделок",
        lambda r: str(r["paper"]["trades"]) if r["paper"] else "—")
    row("Бумага: win rate",
        lambda r: f"{r['paper']['win_rate']:.0%}" if r["paper"] else "—")

    print("\nВывод: смотри на 'лучший/медианный test Sharpe' и 'с положит. Sharpe' —")
    print("это качество отобранных стратегий на out-of-sample. Если у одного R:R")
    print("стабильно выше — он лучше подходит ТЕКУЩИМ данным (не гарантия будущего).")


if __name__ == "__main__":
    main()
