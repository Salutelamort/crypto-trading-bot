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
