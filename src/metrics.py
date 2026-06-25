"""
Метрики производительности — чистый numpy. Никакого LLM.

Ключевой принцип из фото-комментария (piratastuertos):
"Я разделил метрики для ПРОДВИЖЕНИЯ от метрик для УБИЙСТВА. Подделать одну
метрику можно. Подделать две независимые метрики одновременно намного сложнее."
+ метрика consistency (backtest winrate vs реальный winrate) ловит переобучение.
"""
import numpy as np
import pandas as pd

# Кол-во периодов в году для годового Sharpe (зависит от таймфрейма).
PERIODS_PER_YEAR = {
    "1m": 525_600, "5m": 105_120, "15m": 35_040,
    "1h": 8_760, "4h": 2_190, "1d": 365,
}


def compute_metrics(equity: pd.Series, returns: pd.Series,
                    trade_results: list, timeframe: str,
                    buy_hold: float = 0.0) -> dict:
    """
    equity        — кривая капитала (Series)
    returns       — поэкземплярные доходности портфеля
    trade_results — список PnL завершённых сделок (для winrate)
    buy_hold      — доходность пассивного "купи и держи" за тот же период.
                    alpha = доходность стратегии минус buy_hold (сколько мы добавили
                    сверх рынка). В медвежий рынок положительная alpha = сохранение
                    капитала, даже если абсолютная доходность отрицательна.
    """
    ann = PERIODS_PER_YEAR.get(timeframe, 8_760)

    r = returns.fillna(0).values
    if r.std() > 0:
        sharpe = float(np.sqrt(ann) * r.mean() / r.std())
    else:
        sharpe = 0.0

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) else 0.0

    # Максимальная просадка
    if len(equity):
        running_max = equity.cummax()
        dd = (equity - running_max) / running_max
        max_drawdown = float(-dd.min())
    else:
        max_drawdown = 0.0

    wins = [p for p in trade_results if p > 0]
    win_rate = float(len(wins) / len(trade_results)) if trade_results else 0.0

    return {
        "sharpe": round(sharpe, 3),
        "total_return": round(total_return, 4),
        "max_drawdown": round(max_drawdown, 4),
        "win_rate": round(win_rate, 3),
        "num_trades": len(trade_results),
        "buy_hold": round(buy_hold, 4),
        "alpha": round(total_return - buy_hold, 4),
    }


def consistency(train_winrate: float, test_winrate: float) -> float:
    """
    Метрика согласованности из фото-комментария.
    1.0 = out-of-sample так же хорош как in-sample.
    ~0  = агент работал только на исторических данных (переобучение).
    """
    if train_winrate <= 0:
        return 0.0
    return round(min(test_winrate / train_winrate, 2.0), 3)
