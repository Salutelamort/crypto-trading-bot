"""
GOLDEN-регрессионный тест ядра бэктеста.

Зачем: при любом рефакторинге (особенно оптимизации скорости) результаты бэктеста
должны оставаться ПОБИТОВО теми же. Этот тест фиксирует "эталон" (golden) на
ДЕТЕРМИНИРОВАННЫХ синтетических данных и сравнивает с ним.

Использование:
  python tests/golden_backtest.py            # проверить (упадёт при расхождении)
  python tests/golden_backtest.py --update    # перегенерировать эталон (осознанно!)

Данные синтетические (сид фиксирован) → тест НЕ зависит от сети/биржи.
"""
import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import backtest as bt   # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_backtest.json")


def synthetic_df(n=2500, seed=42):
    """Детерминированный OHLCV: случайное блуждание с дрейфом и режимами."""
    rng = np.random.RandomState(seed)
    # смесь режимов: тренд вверх, вниз, боковик — чтобы задеть разные ветки стратегий
    rets = np.concatenate([
        rng.normal(0.0008, 0.02, n // 3),    # бычий
        rng.normal(-0.0010, 0.025, n // 3),  # медвежий
        rng.normal(0.0000, 0.015, n - 2 * (n // 3)),  # боковик
    ])
    close = 100 * np.exp(np.cumsum(rets))
    # high/low/open вокруг close детерминированно
    spread = np.abs(rng.normal(0, 0.01, len(close))) + 0.002
    high = close * (1 + spread)
    low = close * (1 - spread)
    openp = close * (1 + rng.normal(0, 0.003, len(close)))
    vol = np.abs(rng.normal(1000, 200, len(close)))
    idx = pd.date_range("2022-01-01", periods=len(close), freq="4h", tz="UTC")
    return pd.DataFrame({"open": openp, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


def cfg():
    """Фиксированный конфиг ядра (не зависит от config.yaml, чтобы тест был стабилен)."""
    return {
        "train_ratio": 0.5,
        "risk": {
            "allow_short": True, "position_fraction": 0.10,
            "atr_stop": True, "atr_period": 14,
            "atr_stop_mult": 2.0, "atr_trail_mult": 2.5, "atr_take_mult": 6.0,
            "stop_loss_pct": 0.025, "trailing_stop_pct": 0.015, "take_profit_pct": 0.05,
        },
        "costs": {"fee_pct": 0.001, "slippage_pct": 0.0005},
        "execution": {"signal_delay_bars": 1},
        "validation": {"walk_forward_windows": 4},
    }


def genomes():
    """Набор геномов, задевающий все типы стратегий и ветки риска (ATR/cooldown/short)."""
    base = {"symbol": "SYN", "timeframe": "4h",
            "stop_atr": 2.0, "rr": 3.0, "trail_atr": 2.5, "cooldown": 6}
    out = [
        {**base, "type": "mean_reversion", "period": 20, "z_entry": -1.5},
        {**base, "type": "momentum", "rsi_period": 14, "rsi_entry": 55},
        {**base, "type": "breakout", "lookback": 40},
        {**base, "type": "ma_cross", "fast": 20, "slow": 100},
        {**base, "type": "pullback_trend", "trend_ma": 100, "rsi_period": 3, "rsi_dip": 25},
        {**base, "type": "failure_test", "lookback": 20},
        {**base, "type": "breakout_retest", "lookback": 40, "confirm": 2},
        {**base, "type": "donchian_trend", "entry_ch": 40, "exit_ch": 20},
        {**base, "type": "vol_breakout", "atr_period": 14, "lookback": 40, "squeeze": 0.8},
        {**base, "type": "mtf_trend", "fast": 20, "mid": 50, "slow": 150},
        {**base, "type": "wyckoff_breakout", "lookback": 40, "vol_mult": 1.8},
        {**base, "type": "williams_volatility", "atr_period": 14, "k": 1.0},
        {**base, "type": "supertrend", "st_period": 10, "st_mult": 3.0},
        {**base, "type": "macd_adx", "adx_min": 25},
        {**base, "type": "breakout", "lookback": 40, "cooldown": 0},   # без кулдауна
    ]
    return out


METRIC_KEYS = ["sharpe", "sortino", "calmar", "profit_factor", "total_return",
               "max_drawdown", "win_rate", "num_trades", "alpha"]


def compute():
    df = synthetic_df()
    c = cfg()
    result = {}
    for g in genomes():
        train_m, test_m, rob = bt.walk_forward_eval(g, df, c)
        key = f"{g['type']}_cd{g.get('cooldown')}"
        result[key] = {
            "train": {k: train_m.get(k) for k in METRIC_KEYS},
            "test": {k: test_m.get(k) for k in METRIC_KEYS},
            "robustness": rob,
        }
    return result


def main():
    update = "--update" in sys.argv
    current = compute()
    if update or not os.path.exists(GOLDEN_PATH):
        with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2, ensure_ascii=False)
        print(f"Эталон записан: {GOLDEN_PATH} ({len(current)} геномов)")
        return 0

    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = json.load(f)

    mismatches = []
    for key, cur in current.items():
        if key not in golden:
            mismatches.append(f"  НОВЫЙ геном (нет в эталоне): {key}")
            continue
        for split in ("train", "test"):
            for mk in METRIC_KEYS:
                a = cur[split][mk]
                b = golden[key][split][mk]
                if a is None or b is None:
                    if a != b:
                        mismatches.append(f"  {key}.{split}.{mk}: {b} -> {a}")
                elif abs(float(a) - float(b)) > 1e-6:
                    mismatches.append(f"  {key}.{split}.{mk}: {b} -> {a}")
        if abs(float(cur["robustness"]) - float(golden[key]["robustness"])) > 1e-6:
            mismatches.append(f"  {key}.robustness: {golden[key]['robustness']} -> {cur['robustness']}")

    if mismatches:
        print("РЕГРЕССИЯ! Результаты бэктеста изменились:")
        print("\n".join(mismatches))
        return 1
    print(f"OK: все {len(current)} геномов совпадают с эталоном (регрессий нет).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
