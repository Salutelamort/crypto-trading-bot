"""
SQLite-хранилище. Принцип из треда (piratastuertos): "SQLite для всего,
никакого Postgres, никакого Redis". Каждое решение записывается с обоснованием
и ожидаемым результатом — архитектура построена вокруг отслеживаемости.
"""
import sqlite3
import json
import os
from datetime import datetime, timezone, timedelta


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
    test_sortino REAL,                 -- Sortino (штраф только за просадки)
    test_calmar  REAL,                 -- Calmar (доход / макс. просадка)
    test_pf      REAL,                 -- profit factor (прибыли / убытки)
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

-- КОМПАКТНАЯ ПАМЯТЬ ОБ ИСПЫТАНИЯХ (агрегат по тип×символ×таймфрейм).
-- Заменяет «кладбище» сырых killed-агентов: храним не миллионы строк, а сводку,
-- которой достаточно для (1) списка доказанных символов, (2) планки Deflated
-- Sharpe (нужны count+разброс), (3) лучшего Sharpe за всё время. Размер ограничен
-- числом комбинаций (типы×символы×ТФ), растёт на КБ, а не на ГБ.
CREATE TABLE IF NOT EXISTS agent_stats (
    type         TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    timeframe    TEXT NOT NULL,
    n_trials     INTEGER NOT NULL DEFAULT 0,  -- сколько агентов этой комбо оценено
    n_promoted   INTEGER NOT NULL DEFAULT 0,  -- сколько допущено к торговле
    sum_sharpe   REAL NOT NULL DEFAULT 0,     -- Σ test_sharpe (для среднего)
    sumsq_sharpe REAL NOT NULL DEFAULT 0,     -- Σ test_sharpe² (для дисперсии → Deflated)
    max_sharpe   REAL,                        -- лучший test_sharpe за всё время
    sum_alpha    REAL NOT NULL DEFAULT 0,     -- Σ alpha (для среднего обгона рынка)
    updated_at   TEXT,
    PRIMARY KEY (type, symbol, timeframe)
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
    for col in ("test_buyhold", "test_alpha", "test_sortino", "test_calmar", "test_pf"):
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


def update_agent_metrics(conn, agent_id: int, train: dict, test: dict, consistency: float,
                         count_trial: bool = True):
    conn.execute(
        """UPDATE agents SET
            train_sharpe=?, train_return=?, train_winrate=?, train_trades=?,
            test_sharpe=?,  test_return=?,  test_winrate=?,  test_trades=?, test_maxdd=?,
            test_buyhold=?, test_alpha=?, test_sortino=?, test_calmar=?, test_pf=?,
            consistency=?
           WHERE id=?""",
        (train["sharpe"], train["total_return"], train["win_rate"], train["num_trades"],
         test["sharpe"],  test["total_return"],  test["win_rate"],  test["num_trades"], test["max_drawdown"],
         test.get("buy_hold", 0.0), test.get("alpha", 0.0),
         test.get("sortino", 0.0), test.get("calmar", 0.0), test.get("profit_factor", 0.0),
         consistency, agent_id),
    )
    # Сворачиваем испытание в компактную память (один агент = одно испытание).
    # count_trial=False — при ПЕРЕОЦЕНКЕ уже существующего агента (не новое
    # испытание, счётчик Deflated Sharpe раздувать нельзя).
    row = conn.execute("SELECT genome, symbol, timeframe FROM agents WHERE id=?",
                       (agent_id,)).fetchone()
    if row is not None and count_trial:
        try:
            stype = json.loads(row["genome"]).get("type", "?")
        except (ValueError, TypeError):
            stype = "?"
        record_trial(conn, stype, row["symbol"], row["timeframe"],
                     test.get("sharpe"), test.get("alpha", 0.0))
    conn.commit()


def set_agent_status(conn, agent_id: int, status: str):
    killed_at = now_iso() if status == "killed" else None
    conn.execute("UPDATE agents SET status=?, killed_at=? WHERE id=?",
                 (status, killed_at, agent_id))
    if status == "promoted":
        r = conn.execute("SELECT genome, symbol, timeframe FROM agents WHERE id=?",
                         (agent_id,)).fetchone()
        if r is not None:
            try:
                stype = json.loads(r["genome"]).get("type", "?")
            except (ValueError, TypeError):
                stype = "?"
            conn.execute(
                "UPDATE agent_stats SET n_promoted=n_promoted+1 "
                "WHERE type=? AND symbol=? AND timeframe=?",
                (stype, r["symbol"], r["timeframe"]))
    conn.commit()


# ---------- компактная память об испытаниях (agent_stats) ----------
def record_trial(conn, stype, symbol, timeframe, sharpe, alpha=0.0):
    """Учитывает ОДНО испытание в сводке (тип×символ×ТФ). Без commit (вызывающий
    коммитит). Невалидный Sharpe не учитываем, чтобы не портить разброс."""
    if sharpe is None or not (-90 < sharpe < 90):
        return
    alpha = alpha or 0.0
    conn.execute(
        """INSERT INTO agent_stats
             (type,symbol,timeframe,n_trials,n_promoted,sum_sharpe,sumsq_sharpe,max_sharpe,sum_alpha,updated_at)
           VALUES (?,?,?,1,0,?,?,?,?,?)
           ON CONFLICT(type,symbol,timeframe) DO UPDATE SET
             n_trials     = n_trials + 1,
             sum_sharpe   = sum_sharpe + excluded.sum_sharpe,
             sumsq_sharpe = sumsq_sharpe + excluded.sumsq_sharpe,
             max_sharpe   = MAX(COALESCE(max_sharpe, -1e9), excluded.max_sharpe),
             sum_alpha    = sum_alpha + excluded.sum_alpha,
             updated_at   = excluded.updated_at""",
        (stype, symbol, timeframe, sharpe, sharpe * sharpe, sharpe, alpha, now_iso()))


def proven_symbols_from_stats(conn, bar: float) -> set:
    """Символы, где хоть одна комбо когда-либо дала test_sharpe выше планки."""
    rows = conn.execute(
        "SELECT DISTINCT symbol FROM agent_stats WHERE max_sharpe > ?", (bar,)).fetchall()
    return {r["symbol"] for r in rows}


def trial_global_stats(conn):
    """(T, sigma) по ВСЕМ испытаниям из сводки — для планки Deflated Sharpe."""
    r = conn.execute(
        "SELECT SUM(n_trials) n, SUM(sum_sharpe) s, SUM(sumsq_sharpe) ss "
        "FROM agent_stats").fetchone()
    n = r["n"] or 0
    if n < 2:
        return n, 0.0
    mean = r["s"] / n
    var = (r["ss"] - r["s"] * mean) / (n - 1)   # выборочная дисперсия
    sigma = var ** 0.5 if var > 0 else 0.0
    return n, sigma


def best_sharpe_ever(conn):
    """Лучший test_sharpe за всю историю (из сводки). None если пусто."""
    return conn.execute("SELECT MAX(max_sharpe) m FROM agent_stats").fetchone()["m"]


def backfill_agent_stats(conn, force=False):
    """Одноразовый перенос истории из сырых agents в сводку agent_stats.
    Идемпотентен: по умолчанию ничего не делает, если сводка уже заполнена."""
    have = conn.execute("SELECT COUNT(*) c FROM agent_stats").fetchone()["c"]
    if have and not force:
        return 0
    if force:
        conn.execute("DELETE FROM agent_stats")
    agg = {}
    for r in conn.execute(
            "SELECT genome, symbol, timeframe, status, test_sharpe, test_alpha "
            "FROM agents WHERE test_sharpe IS NOT NULL").fetchall():
        s = r["test_sharpe"]
        if s is None or not (-90 < s < 90):
            continue
        try:
            stype = json.loads(r["genome"]).get("type", "?")
        except (ValueError, TypeError):
            stype = "?"
        k = (stype, r["symbol"], r["timeframe"])
        a = agg.setdefault(k, [0, 0, 0.0, 0.0, None, 0.0])
        a[0] += 1
        a[1] += 1 if r["status"] == "promoted" else 0
        a[2] += s
        a[3] += s * s
        a[4] = s if a[4] is None else max(a[4], s)
        a[5] += r["test_alpha"] or 0.0
    ts = now_iso()
    for (stype, sym, tf), a in agg.items():
        conn.execute(
            "INSERT OR REPLACE INTO agent_stats "
            "(type,symbol,timeframe,n_trials,n_promoted,sum_sharpe,sumsq_sharpe,max_sharpe,sum_alpha,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (stype, sym, tf, a[0], a[1], a[2], a[3], a[4], a[5], ts))
    conn.commit()
    return len(agg)


def prune_history(conn, keep_killed=3000, keep_decisions=8000):
    """Retention ПО КОЛИЧЕСТВУ (а не по возрасту): держим N последних killed-агентов
    и N последних решений, остальное сырьё удаляем. Возрастное окно не годится —
    эволюция плодит ~45k killed/сутки, и любой разумный срок переполнит лимит
    GitHub 100 МБ. Лимит по числу делает размер базы стабильным при любой скорости.
    Живые агенты (candidate/promoted) и сделки НЕ трогаем. Память «что работает /
    что нет» сохранена в agent_stats, поэтому удаление сырых killed ничего не теряет.
    Возвращает (удалено_агентов, удалено_решений)."""
    # Агентов, на которых ссылаются сделки, не удаляем НИКОГДА (их единицы):
    # иначе анализ сделок по типам стратегий теряет геном (тип становится «?»).
    da = conn.execute(
        "DELETE FROM agents WHERE status='killed' AND id NOT IN "
        "(SELECT id FROM agents WHERE status='killed' ORDER BY id DESC LIMIT ?) "
        "AND id NOT IN (SELECT DISTINCT agent_id FROM paper_trades)",
        (keep_killed,)).rowcount
    dd = conn.execute(
        "DELETE FROM decisions WHERE id NOT IN "
        "(SELECT id FROM decisions ORDER BY id DESC LIMIT ?)",
        (keep_decisions,)).rowcount
    conn.commit()
    return da, dd


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
