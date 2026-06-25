"""
SQLite-хранилище. Принцип из треда (piratastuertos): "SQLite для всего,
никакого Postgres, никакого Redis". Каждое решение записывается с обоснованием
и ожидаемым результатом — архитектура построена вокруг отслеживаемости.
"""
import sqlite3
import json
import os
from datetime import datetime, timezone


SCHEMA = """
-- Агенты = торговые стратегии с конкретными параметрами (геном).
-- Это НЕ LLM. Это детерминированные гипотезы о поведении рынка.
CREATE TABLE IF NOT EXISTS agents (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    genome       TEXT NOT NULL,        -- JSON генома (тип стратегии + параметры)
    symbol       TEXT NOT NULL,
    timeframe    TEXT NOT NULL,
    status       TEXT NOT NULL,        -- candidate | promoted | killed
    born_at      TEXT NOT NULL,
    killed_at    TEXT,
    -- метрики in-sample (train)
    train_sharpe REAL, train_return REAL, train_winrate REAL, train_trades INTEGER,
    -- метрики out-of-sample (test) — главный критерий честности
    test_sharpe  REAL, test_return REAL, test_winrate REAL, test_trades INTEGER,
    test_maxdd   REAL,
    test_buyhold REAL,                 -- доходность "купи и держи" за OOS-период
    test_alpha   REAL,                 -- обгон рынка = test_return - test_buyhold
    consistency  REAL                  -- доля OOS-окон с положительной alpha
);

-- Решения супервизора. Каждое — с обоснованием (отслеживаемость).
CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    agent_id    INTEGER,
    action      TEXT NOT NULL,         -- kill | promote | generate | hold
    backend     TEXT NOT NULL,         -- rules | claude
    rationale   TEXT NOT NULL,         -- ПОЧЕМУ принято решение
    FOREIGN KEY(agent_id) REFERENCES agents(id)
);

-- Бумажные сделки (с учётом комиссий и проскальзывания).
CREATE TABLE IF NOT EXISTS paper_trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    INTEGER NOT NULL,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,         -- BUY | SELL
    ts          TEXT NOT NULL,
    price       REAL NOT NULL,
    qty         REAL NOT NULL,
    fee         REAL NOT NULL,
    pnl         REAL,                  -- реализованный PnL при закрытии
    reason      TEXT,                  -- signal | stop_loss | trailing | take_profit
    FOREIGN KEY(agent_id) REFERENCES agents(id)
);

-- Кэш рыночных данных (OHLCV).
CREATE TABLE IF NOT EXISTS candles (
    symbol      TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    open_time   INTEGER NOT NULL,      -- unix ms
    open        REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (symbol, timeframe, open_time)
);

-- Карантин символов (отрицательная историческая PnL → блок продвижения).
CREATE TABLE IF NOT EXISTS symbol_quarantine (
    symbol      TEXT PRIMARY KEY,
    reason      TEXT,
    ts          TEXT
);

-- ЖИВОЙ бумажный счёт (форвард-тест в реальном времени, без биржи).
CREATE TABLE IF NOT EXISTS live_account (
    id          INTEGER PRIMARY KEY CHECK (id=1),
    capital     REAL,            -- свободный кэш
    peak_equity REAL,            -- пик капитала (для стоп-крана просадки)
    started_at  TEXT
);

-- Открытые позиции живого счёта (переживают перезапуск бота).
CREATE TABLE IF NOT EXISTS live_positions (
    agent_id    INTEGER PRIMARY KEY,
    symbol      TEXT,
    entry_price REAL,
    units       REAL,
    peak_price  REAL,
    opened_at   TEXT
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn):
    """Лёгкие миграции для существующих БД (добавление новых колонок)."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(agents)").fetchall()}
    for col in ("test_buyhold", "test_alpha"):
        if col not in cols:
            conn.execute(f"ALTER TABLE agents ADD COLUMN {col} REAL")
    pcols = {r["name"] for r in conn.execute("PRAGMA table_info(live_positions)").fetchall()}
    if "direction" not in pcols:
        conn.execute("ALTER TABLE live_positions ADD COLUMN direction INTEGER DEFAULT 1")
    if "notional" not in pcols:
        conn.execute("ALTER TABLE live_positions ADD COLUMN notional REAL")
    if "atr" not in pcols:
        conn.execute("ALTER TABLE live_positions ADD COLUMN atr REAL")
    conn.commit()


# ---------- агенты ----------
def insert_agent(conn, genome: dict, symbol: str, timeframe: str) -> int:
    cur = conn.execute(
        "INSERT INTO agents (genome, symbol, timeframe, status, born_at) "
        "VALUES (?,?,?,?,?)",
        (json.dumps(genome), symbol, timeframe, "candidate", now_iso()),
    )
    conn.commit()
    return cur.lastrowid


def update_agent_metrics(conn, agent_id: int, train: dict, test: dict, consistency: float):
    conn.execute(
        """UPDATE agents SET
            train_sharpe=?, train_return=?, train_winrate=?, train_trades=?,
            test_sharpe=?,  test_return=?,  test_winrate=?,  test_trades=?, test_maxdd=?,
            test_buyhold=?, test_alpha=?,
            consistency=?
           WHERE id=?""",
        (train["sharpe"], train["total_return"], train["win_rate"], train["num_trades"],
         test["sharpe"],  test["total_return"],  test["win_rate"],  test["num_trades"], test["max_drawdown"],
         test.get("buy_hold", 0.0), test.get("alpha", 0.0),
         consistency, agent_id),
    )
    conn.commit()


def set_agent_status(conn, agent_id: int, status: str):
    killed_at = now_iso() if status == "killed" else None
    conn.execute("UPDATE agents SET status=?, killed_at=? WHERE id=?",
                 (status, killed_at, agent_id))
    conn.commit()


def get_agents(conn, status: str = None):
    if status:
        rows = conn.execute("SELECT * FROM agents WHERE status=? ORDER BY id", (status,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM agents ORDER BY id").fetchall()
    return [dict(r) for r in rows]


# ---------- решения ----------
def log_decision(conn, agent_id, action: str, backend: str, rationale: str):
    conn.execute(
        "INSERT INTO decisions (ts, agent_id, action, backend, rationale) VALUES (?,?,?,?,?)",
        (now_iso(), agent_id, action, backend, rationale),
    )
    conn.commit()


# ---------- карантин ----------
def quarantine_symbol(conn, symbol: str, reason: str):
    conn.execute(
        "INSERT OR REPLACE INTO symbol_quarantine (symbol, reason, ts) VALUES (?,?,?)",
        (symbol, reason, now_iso()),
    )
    conn.commit()


def quarantined_symbols(conn) -> set:
    rows = conn.execute("SELECT symbol FROM symbol_quarantine").fetchall()
    return {r["symbol"] for r in rows}


# ---------- сделки ----------
def log_paper_trade(conn, agent_id, symbol, side, price, qty, fee, pnl, reason):
    conn.execute(
        "INSERT INTO paper_trades (agent_id, symbol, side, ts, price, qty, fee, pnl, reason) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (agent_id, symbol, side, now_iso(), price, qty, fee, pnl, reason),
    )
    conn.commit()
