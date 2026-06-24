"""
Геном стратегии. Принцип из треда (piratastuertos):
"Я компилирую вывод LLM в ДЕТЕРМИНИРОВАННЫЙ Python. Сам трейдинг — 100%
детерминированный Python. LLM только генерирует и проверяет стратегии,
он никогда не принимает решения по отдельным сделкам."

Геном = небольшой словарь параметров. Намеренно МАЛО параметров —
принцип из треда: "Стратегии с 15 настраиваемыми параметрами легче всего
переоптимизировать". Здесь у каждой стратегии 1-3 параметра.

База типов стратегий заложена по принципам технического анализа, которые
систематизирует Adam Grimes в "The Art and Science of Technical Analysis"
(сама книга защищена авторским правом — здесь только общеизвестные концепции):
  - рынок чередует ТРЕНД и ДИАПАЗОН (сжатие/расширение волатильности);
  - торговать С трендом через ОТКАТЫ (pullback continuation);
  - FAILURE TEST — фейдить ложные пробои;
  - пробой уровня + ретест;
  - каналы Дончиана (трендследование).

Эволюция (evolution.py) АВТОМАТИЧЕСКИ перебирает это пространство — пользователю
не нужно придумывать стратегии вручную. Он задаёт лишь рамки (символы, риск).

Сигнал: +1 = быть в позиции (long), 0 = вне рынка (flat).
Только long/flat — потому что это spot-рынок (как у автора, Binance spot).
"""
import random
import numpy as np
import pandas as pd
from . import indicators as ind

STRATEGY_TYPES = [
    # классика
    "mean_reversion", "momentum", "breakout", "ma_cross",
    # принципы Grimes (Искусство и наука технического анализа)
    "pullback_trend", "failure_test", "breakout_retest", "donchian_trend",
    "vol_breakout",
    # из библиотеки пользователя:
    "mtf_trend",          # Брайан Шеннон — Мультитаймфреймный технический анализ
    "wyckoff_breakout",   # Вайкофф / Дэвид Вайс (Сделки на горизонте) — объём+цена
    "williams_volatility",# Ларри Вильямс — Секреты торговли на фьючерсном рынке
]


def random_genome(symbol: str, timeframe: str, rng: random.Random) -> dict:
    """Создаёт случайный геном. Это и есть 'генерация кандидата'."""
    stype = rng.choice(STRATEGY_TYPES)
    g = {"type": stype, "symbol": symbol, "timeframe": timeframe}

    if stype == "mean_reversion":
        g["period"] = rng.choice([14, 20, 30, 50])
        g["z_entry"] = round(rng.uniform(-2.5, -1.0), 2)
    elif stype == "momentum":
        g["rsi_period"] = rng.choice([7, 14, 21])
        g["rsi_entry"] = rng.choice([50, 55, 60])
    elif stype == "breakout":
        g["lookback"] = rng.choice([20, 40, 55, 100])
    elif stype == "ma_cross":
        fast = rng.choice([10, 20, 30])
        slow = rng.choice([50, 100, 150])
        g["fast"], g["slow"] = fast, max(slow, fast + 20)

    # --- Grimes: тренд + откат. Покупаем просадку RSI в восходящем тренде. ---
    elif stype == "pullback_trend":
        g["trend_ma"] = rng.choice([50, 100, 150, 200])
        g["rsi_period"] = rng.choice([2, 3, 5, 14])     # короткий RSI ловит откат
        g["rsi_dip"] = rng.choice([20, 25, 30, 35])

    # --- Grimes: failure test. Ложный пробой вниз + возврат над уровень → лонг. ---
    elif stype == "failure_test":
        g["lookback"] = rng.choice([10, 20, 30, 55])

    # --- Grimes: пробой максимума + подтверждение (импульс продолжается). ---
    elif stype == "breakout_retest":
        g["lookback"] = rng.choice([20, 40, 55])
        g["confirm"] = rng.choice([1, 2, 3])            # сколько баров держаться выше

    # --- трендследование по каналу Дончиана (вход и выход — разные каналы). ---
    elif stype == "donchian_trend":
        g["entry_ch"] = rng.choice([20, 40, 55])
        g["exit_ch"] = rng.choice([10, 20])

    # --- Grimes: сжатие волатильности → расширение. Пробой после "тихого" рынка. ---
    elif stype == "vol_breakout":
        g["atr_period"] = rng.choice([14, 20])
        g["lookback"] = rng.choice([20, 40, 55])
        g["squeeze"] = round(rng.uniform(0.6, 0.9), 2)  # atr ниже squeeze*средн.atr

    # --- Шеннон: мультитаймфрейм. Торгуем только при выстроенном стеке трендов. ---
    elif stype == "mtf_trend":
        g["fast"] = rng.choice([10, 20])
        g["mid"] = rng.choice([50, 60])
        g["slow"] = rng.choice([100, 150, 200])

    # --- Вайкофф/Вайс: пробой диапазона ПОДТВЕРЖДЁННЫЙ всплеском объёма. ---
    elif stype == "wyckoff_breakout":
        g["lookback"] = rng.choice([20, 40, 55])
        g["vol_mult"] = round(rng.uniform(1.5, 2.5), 2)  # объём выше среднего в N раз

    # --- Ларри Вильямс: пробой волатильности (диапазон предыдущего бара × k). ---
    elif stype == "williams_volatility":
        g["atr_period"] = rng.choice([5, 10, 14])
        g["k"] = round(rng.uniform(0.5, 1.5), 2)
    return g


def mutate(genome: dict, rng: random.Random) -> dict:
    """Небольшая мутация одного параметра — для эволюции выживших."""
    g = dict(genome)
    keys = [k for k in g if k not in ("type", "symbol", "timeframe")]
    if not keys:
        return g
    k = rng.choice(keys)
    v = g[k]
    if isinstance(v, int):
        g[k] = max(2, v + rng.choice([-10, -5, -2, 2, 5, 10]))
    else:  # float (z_entry, squeeze)
        g[k] = round(v + rng.choice([-0.3, -0.1, 0.1, 0.3]), 2)
    # инварианты
    if g["type"] == "ma_cross":
        g["slow"] = max(g["slow"], g["fast"] + 20)
    if g["type"] == "donchian_trend":
        g["exit_ch"] = min(g.get("exit_ch", 10), g.get("entry_ch", 20))
    if g["type"] == "mtf_trend":  # стек должен сохранять порядок fast<mid<slow
        g["mid"] = max(g["mid"], g["fast"] + 10)
        g["slow"] = max(g["slow"], g["mid"] + 20)
    return g


def _state_from_entries(index, entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Строит позицию {0,1} из булевых условий входа/выхода (вход приоритетнее)."""
    raw = pd.Series(np.nan, index=index, dtype="float64")
    raw[exit_.fillna(False).to_numpy()] = 0.0
    raw[entry.fillna(False).to_numpy()] = 1.0
    return raw.ffill().fillna(0.0).astype(int)


def signal(genome: dict, df: pd.DataFrame) -> pd.Series:
    """
    КОМПИЛЯЦИЯ генома в детерминированный сигнал.
    Возвращает Series из {0, 1}: 1 = держим long, 0 = вне рынка.
    Чистая математика, без всякого LLM.
    """
    close = df["close"]
    t = genome["type"]

    if t == "mean_reversion":
        z = ind.zscore(close, genome["period"])
        sig = _state_from_entries(df.index, z <= genome["z_entry"], z >= 0)

    elif t == "momentum":
        r = ind.rsi(close, genome["rsi_period"])
        sig = (r >= genome["rsi_entry"]).astype(int)

    elif t == "breakout":
        hh = ind.rolling_high(close.shift(1), genome["lookback"])
        ll = ind.rolling_low(close.shift(1), genome["lookback"])
        sig = _state_from_entries(df.index, close > hh, close < ll)

    elif t == "ma_cross":
        sig = (ind.ema(close, genome["fast"]) > ind.ema(close, genome["slow"])).astype(int)

    elif t == "pullback_trend":
        uptrend = close > ind.ema(close, genome["trend_ma"])
        r = ind.rsi(close, genome["rsi_period"])
        entry = uptrend & (r <= genome["rsi_dip"])      # откат внутри тренда
        exit_ = (~uptrend) | (r >= 55)                  # тренд кончился или откат отыгран
        sig = _state_from_entries(df.index, entry, exit_)

    elif t == "failure_test":
        lb = genome["lookback"]
        ll = ind.rolling_low(close.shift(1), lb)
        hh = ind.rolling_high(close.shift(1), lb)
        # ложный пробой вниз: минимум бара пробил уровень, но закрылись ВЫШЕ него
        entry = (df["low"] < ll) & (close > ll)
        exit_ = (close >= hh) | (close < ll)            # цель достигнута / реальный пробой
        sig = _state_from_entries(df.index, entry, exit_)

    elif t == "breakout_retest":
        hh = ind.rolling_high(close.shift(1), genome["lookback"])
        above = (close > hh)
        entry = above & (above.rolling(genome["confirm"]).sum() >= genome["confirm"])
        exit_ = close < hh
        sig = _state_from_entries(df.index, entry, exit_)

    elif t == "donchian_trend":
        hh = ind.rolling_high(close.shift(1), genome["entry_ch"])
        ll = ind.rolling_low(close.shift(1), genome["exit_ch"])
        sig = _state_from_entries(df.index, close > hh, close < ll)

    elif t == "vol_breakout":
        a = ind.atr(df, genome["atr_period"])
        squeeze = a < (a.rolling(genome["lookback"]).mean() * genome["squeeze"])
        hh = ind.rolling_high(close.shift(1), genome["lookback"])
        entry = squeeze.shift(1).fillna(False) & (close > hh)  # тихий рынок → пробой
        exit_ = close < ind.ema(close, genome["lookback"])
        sig = _state_from_entries(df.index, entry, exit_)

    elif t == "mtf_trend":
        # Шеннон: вход только когда быстрая > средней > медленной (тренды
        # разных горизонтов выстроены в одну сторону); выход — когда стек ломается.
        f = ind.ema(close, genome["fast"])
        m = ind.ema(close, genome["mid"])
        s = ind.ema(close, genome["slow"])
        sig = ((f > m) & (m > s)).astype(int)

    elif t == "wyckoff_breakout":
        # Вайкофф/Вайс: пробой максимума, ПОДТВЕРЖДЁННЫЙ всплеском объёма
        # (объём = "топливо" движения; пробой без объёма не доверяем).
        lb = genome["lookback"]
        hh = ind.rolling_high(close.shift(1), lb)
        vol_ok = df["volume"] > df["volume"].rolling(lb).mean() * genome["vol_mult"]
        entry = (close > hh) & vol_ok
        exit_ = close < ind.ema(close, lb)
        sig = _state_from_entries(df.index, entry, exit_)

    elif t == "williams_volatility":
        # Ларри Вильямс: вход при выходе цены за диапазон волатильности
        # (close прошлого бара + k*ATR); выход — симметрично вниз.
        a = ind.atr(df, genome["atr_period"])
        up = close.shift(1) + genome["k"] * a.shift(1)
        dn = close.shift(1) - genome["k"] * a.shift(1)
        sig = _state_from_entries(df.index, close > up, close < dn)

    else:
        raise ValueError(f"Неизвестный тип стратегии: {t}")

    return sig.fillna(0).astype(int)
