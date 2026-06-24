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


def cmd_fetch(conn, cfg):
    print("Загрузка рыночных данных с Binance (публичный API, без ключей)...")
    for sym in cfg["symbols"]:
        feed.fetch_ohlcv(conn, sym, cfg["timeframe"], cfg["history_days"])


def _load_data(conn, cfg):
    data = {}
    for sym in cfg["symbols"]:
        df = feed.load_ohlcv(conn, sym, cfg["timeframe"])
        if df.empty:
            print(f"  [!] Нет данных для {sym}. Запусти: python main.py fetch")
        else:
            data[sym] = df
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


COMMANDS = {
    "fetch": cmd_fetch, "evolve": cmd_evolve, "supervise": cmd_supervise,
    "paper": cmd_paper, "live": cmd_live, "status": cmd_status, "run": cmd_run,
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
