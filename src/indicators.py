"""
Технические индикаторы — ЧИСТЫЙ детерминированный numpy/pandas.
Принцип из треда: "LLM ужасно справляются с сырыми математическими расчётами
и галлюцинируют результаты". Поэтому вся математика тут — обычный код,
никаких LLM. Один и тот же вход → всегда один и тот же выход.
"""
import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def zscore(series: pd.Series, period: int) -> pd.Series:
    """Z-оценка: на сколько стандартных отклонений цена ушла от среднего.
    База для mean-reversion (торговля против экстремальных отклонений)."""
    mean = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - mean) / std.replace(0, np.nan)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — волатильность."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def rolling_high(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).max()


def rolling_low(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).min()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD: (быстрая EMA - медленная EMA) и его сигнальная линия. Классика 12/26/9."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ADX — СИЛА тренда (не направление). Высокий ADX = выраженный тренд,
    низкий = боковик/шум. Используем как фильтр: торговать только в тренде.
    Детерминированное приближение (скользящие средние вместо сглаживания Уайлдера).
    """
    high, low = df["high"], df["low"]
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down
    tr = atr(df, period).replace(0, np.nan)
    plus_di = 100 * (plus_dm.rolling(period).mean() / tr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / tr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.Series:
    """
    Supertrend — ATR-трендследование. Возвращает направление тренда: +1 (вверх) / -1 (вниз).
    Линия идёт под ценой в аптренде и над ценой в даунтренде; пробой = смена тренда.
    """
    hl2 = (df["high"] + df["low"]) / 2.0
    a = atr(df, period)
    upper = (hl2 + mult * a).values
    lower = (hl2 - mult * a).values
    close = df["close"].values
    n = len(df)
    fu = upper.copy()
    fl = lower.copy()
    trend = np.ones(n)
    for i in range(1, n):
        fu[i] = min(upper[i], fu[i - 1]) if close[i - 1] <= fu[i - 1] else upper[i]
        fl[i] = max(lower[i], fl[i - 1]) if close[i - 1] >= fl[i - 1] else lower[i]
        if close[i] > fu[i - 1]:
            trend[i] = 1
        elif close[i] < fl[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    return pd.Series(trend, index=df.index)
