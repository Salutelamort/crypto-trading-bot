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

Сигнал: +1 = long (рынок растёт), -1 = short (рынок падает), 0 = кэш (вне рынка).
ТРИ состояния. Раньше было только long/flat — и на медвежьем рынке система была
обречена (см. разбор). Теперь стратегия может зарабатывать на падении (short) и,
главное, УХОДИТЬ В КЭШ, когда сигнала нет. Short включается флагом allow_short
(для spot он отключается, для фьючерс-/демо-режима — включён).
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
    # добавлено из обзора конкурентов (docs/competitors_ideas.md):
    "supertrend",         # ATR-трендследование (популярно в Freqtrade/TradingView)
    "macd_adx",           # MACD-кросс + фильтр силы тренда ADX (не торгуем в боковике)
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

    # --- Supertrend: ATR-трендследование (направление = сторона позиции). ---
    elif stype == "supertrend":
        g["st_period"] = rng.choice([7, 10, 14])
        g["st_mult"] = round(rng.uniform(2.0, 4.0), 1)

    # --- MACD-кросс, отфильтрованный силой тренда ADX (в боковике — кэш). ---
    elif stype == "macd_adx":
        g["adx_min"] = rng.choice([20, 25, 30])   # MACD держим классический 12/26/9

    # --- ГЕНЫ РИСКА (общие для всех типов; их подбирает эволюция) ---
    # Карвер: риск измеряется ОТ ВОЛАТИЛЬНОСТИ, а не вслепую. Поэтому стоп/тейк/трейл
    # заданы в единицах ATR и являются частью генома — бот сам ищет оптимум.
    g["stop_atr"] = round(rng.uniform(1.5, 3.5), 2)    # жёсткий стоп = stop_atr × ATR
    g["rr"] = round(rng.uniform(2.0, 5.0), 1)          # R:R; тейк = stop_atr × rr × ATR
    g["trail_atr"] = round(rng.uniform(1.5, 4.0), 2)   # trailing = trail_atr × ATR
    # КУЛДАУН против переторговли ("тысяча порезов"): минимум баров между сделками.
    g["cooldown"] = rng.choice([0, 3, 6, 12, 24])
    return g


def mutate(genome: dict, rng: random.Random) -> dict:
    """Небольшая мутация одного параметра — для эволюции выживших."""
    g = dict(genome)
    keys = [k for k in g if k not in ("type", "symbol", "timeframe")]
    if not keys:
        return g
    k = rng.choice(keys)
    v = g[k]
    if k == "cooldown":
        g[k] = max(0, v + rng.choice([-12, -6, -3, 3, 6, 12]))
    elif isinstance(v, int):
        g[k] = max(2, v + rng.choice([-10, -5, -2, 2, 5, 10]))
    else:  # float (z_entry, squeeze, гены риска)
        g[k] = round(v + rng.choice([-0.3, -0.1, 0.1, 0.3]), 2)
    # инварианты генов риска: стоп/трейл > 0, R:R >= 1
    if "stop_atr" in g:
        g["stop_atr"] = round(max(0.5, g["stop_atr"]), 2)
    if "trail_atr" in g:
        g["trail_atr"] = round(max(0.5, g["trail_atr"]), 2)
    if "rr" in g:
        g["rr"] = round(max(1.0, g["rr"]), 1)
    if "st_mult" in g:
        g["st_mult"] = round(max(1.0, g["st_mult"]), 1)
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


def _directional(index, long_entry, long_exit, short_entry, short_exit,
                 allow_short: bool) -> pd.Series:
    """
    Строит позицию {-1, 0, +1} из условий входа/выхода для ДВУХ направлений.
    +1 = long, -1 = short, 0 = кэш. Вход приоритетнее выхода в один бар.
    Если allow_short=False — короткие входы игнорируются (поведение spot long/flat).
    """
    def b(x):
        return x.fillna(False).to_numpy()
    raw = pd.Series(np.nan, index=index, dtype="float64")
    # сначала выходы (закрытие в кэш), затем входы перетирают их
    raw[b(long_exit)] = 0.0
    raw[b(short_exit)] = 0.0
    raw[b(long_entry)] = 1.0
    if allow_short:
        raw[b(short_entry)] = -1.0
    return raw.ffill().fillna(0.0).astype(int)


def signal(genome: dict, df: pd.DataFrame, allow_short: bool = False) -> pd.Series:
    """
    КОМПИЛЯЦИЯ генома в детерминированный сигнал.
    Возвращает Series из {-1, 0, +1}: +1 = long, -1 = short, 0 = кэш.
    Чистая математика, без всякого LLM. allow_short=False → только long/flat.
    """
    close = df["close"]
    t = genome["type"]

    if t == "mean_reversion":
        # лонг на перепроданности (z низкий), шорт на перекупленности (z высокий)
        z = ind.zscore(close, genome["period"])
        ze = abs(genome["z_entry"])
        sig = _directional(df.index,
                           long_entry=z <= -ze, long_exit=z >= 0,
                           short_entry=z >= ze, short_exit=z <= 0,
                           allow_short=allow_short)

    elif t == "momentum":
        r = ind.rsi(close, genome["rsi_period"])
        hi = genome["rsi_entry"]
        lo = 100 - hi
        sig = pd.Series(np.where(r >= hi, 1, np.where(r <= lo, -1, 0)),
                        index=df.index)
        if not allow_short:
            sig = sig.clip(lower=0)

    elif t == "breakout":
        hh = ind.rolling_high(close.shift(1), genome["lookback"])
        ll = ind.rolling_low(close.shift(1), genome["lookback"])
        sig = _directional(df.index,
                           long_entry=close > hh, long_exit=close < ll,
                           short_entry=close < ll, short_exit=close > hh,
                           allow_short=allow_short)

    elif t == "ma_cross":
        f = ind.ema(close, genome["fast"])
        s = ind.ema(close, genome["slow"])
        sig = pd.Series(np.where(f > s, 1, -1), index=df.index)
        if not allow_short:
            sig = sig.clip(lower=0)

    elif t == "pullback_trend":
        ma = ind.ema(close, genome["trend_ma"])
        uptrend = close > ma
        downtrend = close < ma
        r = ind.rsi(close, genome["rsi_period"])
        dip = genome["rsi_dip"]
        sig = _directional(df.index,
                           long_entry=uptrend & (r <= dip),
                           long_exit=(~uptrend) | (r >= 55),
                           short_entry=downtrend & (r >= 100 - dip),
                           short_exit=(~downtrend) | (r <= 45),
                           allow_short=allow_short)

    elif t == "failure_test":
        lb = genome["lookback"]
        ll = ind.rolling_low(close.shift(1), lb)
        hh = ind.rolling_high(close.shift(1), lb)
        # ложный пробой вниз → лонг; ложный пробой вверх → шорт
        sig = _directional(df.index,
                           long_entry=(df["low"] < ll) & (close > ll),
                           long_exit=(close >= hh) | (close < ll),
                           short_entry=(df["high"] > hh) & (close < hh),
                           short_exit=(close <= ll) | (close > hh),
                           allow_short=allow_short)

    elif t == "breakout_retest":
        lb, cf = genome["lookback"], genome["confirm"]
        hh = ind.rolling_high(close.shift(1), lb)
        ll = ind.rolling_low(close.shift(1), lb)
        above = (close > hh)
        below = (close < ll)
        sig = _directional(df.index,
                           long_entry=above & (above.rolling(cf).sum() >= cf),
                           long_exit=close < hh,
                           short_entry=below & (below.rolling(cf).sum() >= cf),
                           short_exit=close > ll,
                           allow_short=allow_short)

    elif t == "donchian_trend":
        hh = ind.rolling_high(close.shift(1), genome["entry_ch"])
        ll = ind.rolling_low(close.shift(1), genome["entry_ch"])
        hh_x = ind.rolling_high(close.shift(1), genome["exit_ch"])
        ll_x = ind.rolling_low(close.shift(1), genome["exit_ch"])
        sig = _directional(df.index,
                           long_entry=close > hh, long_exit=close < ll_x,
                           short_entry=close < ll, short_exit=close > hh_x,
                           allow_short=allow_short)

    elif t == "vol_breakout":
        a = ind.atr(df, genome["atr_period"])
        squeeze = (a < (a.rolling(genome["lookback"]).mean() * genome["squeeze"])
                   ).shift(1, fill_value=False)
        ema = ind.ema(close, genome["lookback"])
        hh = ind.rolling_high(close.shift(1), genome["lookback"])
        ll = ind.rolling_low(close.shift(1), genome["lookback"])
        sig = _directional(df.index,
                           long_entry=squeeze & (close > hh), long_exit=close < ema,
                           short_entry=squeeze & (close < ll), short_exit=close > ema,
                           allow_short=allow_short)

    elif t == "mtf_trend":
        f = ind.ema(close, genome["fast"])
        m = ind.ema(close, genome["mid"])
        s = ind.ema(close, genome["slow"])
        up = (f > m) & (m > s)
        dn = (f < m) & (m < s)
        sig = pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)
        if not allow_short:
            sig = sig.clip(lower=0)

    elif t == "wyckoff_breakout":
        lb = genome["lookback"]
        hh = ind.rolling_high(close.shift(1), lb)
        ll = ind.rolling_low(close.shift(1), lb)
        ema = ind.ema(close, lb)
        vol_ok = df["volume"] > df["volume"].rolling(lb).mean() * genome["vol_mult"]
        sig = _directional(df.index,
                           long_entry=(close > hh) & vol_ok, long_exit=close < ema,
                           short_entry=(close < ll) & vol_ok, short_exit=close > ema,
                           allow_short=allow_short)

    elif t == "williams_volatility":
        a = ind.atr(df, genome["atr_period"])
        up = close.shift(1) + genome["k"] * a.shift(1)
        dn = close.shift(1) - genome["k"] * a.shift(1)
        sig = _directional(df.index,
                           long_entry=close > up, long_exit=close < dn,
                           short_entry=close < dn, short_exit=close > up,
                           allow_short=allow_short)

    elif t == "supertrend":
        # направление supertrend = сторона позиции (+1 long / -1 short)
        trend = ind.supertrend(df, genome["st_period"], genome["st_mult"])
        sig = trend.astype(int)
        if not allow_short:
            sig = sig.clip(lower=0)

    elif t == "macd_adx":
        # MACD-кросс, но ТОЛЬКО при сильном тренде (ADX>=adx_min); иначе кэш
        ml, sl = ind.macd(close)
        strong = ind.adx(df, 14) >= genome["adx_min"]
        up = (ml > sl) & strong
        dn = (ml < sl) & strong
        sig = pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)
        if not allow_short:
            sig = sig.clip(lower=0)

    else:
        raise ValueError(f"Неизвестный тип стратегии: {t}")

    return sig.fillna(0).astype(int)
