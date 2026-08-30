"""
Получение рыночных данных. Принцип из треда (piratastuertos):
"только публичный REST Binance + websocket. Ничего особенного. Преимущество
не в данных, а в том, как агенты их используют."

Используем ТОЛЬКО публичный endpoint /api/v3/klines — БЕЗ API-ключей.
Данные кэшируются в SQLite, чтобы не дёргать биржу повторно.
"""
import time

import pandas as pd
import requests

BINANCE_BASE = "https://api.binance.com"
# Зеркало для регионов, где основной домен заблокирован:
BINANCE_FALLBACKS = ["https://data-api.binance.vision", "https://api.binance.com"]

_TF_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
}


def _fetch_klines_page(symbol, timeframe, start_ms, limit=1000):
    last_err = None
    for base in BINANCE_FALLBACKS:
        try:
            r = requests.get(
                f"{base}/api/v3/klines",
                params={"symbol": symbol, "interval": timeframe,
                        "startTime": start_ms, "limit": limit},
                timeout=20,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa
            last_err = e
            continue
    raise RuntimeError(f"Не удалось получить данные с Binance: {last_err}")


def fetch_ohlcv(conn, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Тянет историю свечей, кэширует в SQLite, возвращает DataFrame."""
    if timeframe not in _TF_MS:
        raise ValueError(f"Неподдерживаемый timeframe: {timeframe}")

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 86_400_000
    step = _TF_MS[timeframe]

    cached = conn.execute(
        "SELECT MIN(open_time) first_ms, MAX(open_time) last_ms FROM candles "
        "WHERE symbol=? AND timeframe=?", (symbol, timeframe),
    ).fetchone()
    # После первого полного снимка загружаем только новые бары. Если старой
    # истории недостаточно, INSERT OR REPLACE безопасно дозаполнит весь диапазон.
    have_full_history = cached["first_ms"] is not None and cached["first_ms"] <= start_ms + step
    cursor = max(start_ms, (cached["last_ms"] or (start_ms - step)) + step) \
        if have_full_history else start_ms
    rows = []
    mode = "обновление кэша" if have_full_history else f"загрузка ~{days}д истории"
    print(f"  [{symbol} {timeframe}] {mode}...")
    while cursor < now_ms:
        page = _fetch_klines_page(symbol, timeframe, cursor)
        if not page:
            break
        for k in page:
            rows.append((symbol, timeframe, int(k[0]),
                         float(k[1]), float(k[2]), float(k[3]),
                         float(k[4]), float(k[5])))
        cursor = int(page[-1][0]) + step
        if len(page) < 1000:
            break
        time.sleep(0.2)  # вежливость к публичному API

    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO candles "
            "(symbol,timeframe,open_time,open,high,low,close,volume) "
            "VALUES (?,?,?,?,?,?,?,?)", rows)
        conn.commit()
    print(f"  [{symbol} {timeframe}] сохранено свечей: {len(rows)}")
    return load_ohlcv(conn, symbol, timeframe)


def fetch_since(symbol: str, timeframe: str, start_ms: int,
                end_ms: int | None = None, max_bars: int = 10_080) -> pd.DataFrame:
    """Все бары начиная с start_ms, с пагинацией. Нужен live-стопам после
    задержанного/пропущенного GitHub Actions запуска; `fetch_recent(limit)` не
    покрывает произвольный разрыв."""
    if timeframe not in _TF_MS:
        raise ValueError(f"Неподдерживаемый timeframe: {timeframe}")
    step = _TF_MS[timeframe]
    end_ms = end_ms or int(time.time() * 1000)
    earliest = max(0, end_ms - max_bars * step)
    cursor = max(int(start_ms), earliest)
    rows = []
    while cursor <= end_ms and len(rows) < max_bars:
        page = _fetch_klines_page(symbol, timeframe, cursor,
                                  min(1000, max_bars - len(rows)))
        if not page:
            break
        for k in page:
            opened = int(k[0])
            if opened > end_ms:
                break
            rows.append((opened, float(k[1]), float(k[2]), float(k[3]),
                         float(k[4]), float(k[5])))
        next_cursor = int(page[-1][0]) + step
        if next_cursor <= cursor or len(page) < 1000:
            break
        cursor = next_cursor
        time.sleep(0.05)
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.drop_duplicates("open_time").set_index("dt")


def fetch_all(conn, symbols, timeframes, days: int):
    """Качает историю по всем парам (символ × таймфрейм) для мультитаймфрейма."""
    for sym in symbols:
        for tf in timeframes:
            fetch_ohlcv(conn, sym, tf, days)


def load_all(conn, symbols, timeframes) -> dict:
    """Загружает кэш по всем парам. Ключ словаря — кортеж (symbol, timeframe)."""
    data = {}
    for sym in symbols:
        for tf in timeframes:
            df = load_ohlcv(conn, sym, tf)
            if not df.empty:
                data[(sym, tf)] = df
    return data


def load_ohlcv(conn, symbol: str, timeframe: str) -> pd.DataFrame:
    """Читает закэшированные свечи из SQLite."""
    df = pd.read_sql_query(
        "SELECT open_time, open, high, low, close, volume FROM candles "
        "WHERE symbol=? AND timeframe=? ORDER BY open_time",
        conn, params=(symbol, timeframe),
    )
    if df.empty:
        return df
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("dt")
    return df


def fetch_recent(symbol: str, timeframe: str, limit: int = 400) -> pd.DataFrame:
    """Последние N свечей в реальном времени (для живого пейпера).
    Не кэширует — всегда свежие данные. Возвращает DataFrame с индексом-временем."""
    last_err = None
    for base in BINANCE_FALLBACKS:
        try:
            r = requests.get(f"{base}/api/v3/klines",
                             params={"symbol": symbol, "interval": timeframe,
                                     "limit": limit}, timeout=20)
            r.raise_for_status()
            page = r.json()
            rows = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]),
                     float(k[4]), float(k[5])) for k in page]
            df = pd.DataFrame(rows, columns=["open_time", "open", "high",
                                             "low", "close", "volume"])
            df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            return df.set_index("dt")
        except Exception as e:  # noqa
            last_err = e
            continue
    raise RuntimeError(f"Не удалось получить свежие свечи: {last_err}")


def latest_price(symbol: str) -> float:
    """Текущая цена для бумажной торговли (публичный endpoint)."""
    for base in BINANCE_FALLBACKS:
        try:
            r = requests.get(f"{base}/api/v3/ticker/price",
                             params={"symbol": symbol}, timeout=10)
            r.raise_for_status()
            return float(r.json()["price"])
        except (requests.RequestException, KeyError, TypeError, ValueError):
            continue
    raise RuntimeError("Не удалось получить текущую цену")
