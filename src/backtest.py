"""
Бэктестер — ЧИСТЫЙ детерминированный Python.

Принципы из треда, которые здесь реализованы:
1. "Не используй агента для бэктестинга, используй его для создания хорошей
   СРЕДЫ для бэктестинга" — это та самая среда.
2. "Бумажная торговля игнорирует проскальзывание и расширение спреда. Эти
   скрытые расходы сожрут твою прибыль" — здесь учитываются комиссия и слиппедж.
3. Риск-менеджмент встроен: жёсткий стоп, ratcheting trailing stop, take-profit.
4. Long/flat на spot (как у автора).

Один вход → один детерминированный выход. LLM здесь нет вообще.
"""
import pandas as pd
from . import genome as gn
from . import metrics as mt


def run(genome: dict, df: pd.DataFrame, cfg: dict, sig=None) -> dict:
    """
    Симулирует одного агента на исторических данных.
    Возвращает словарь метрик + кривую equity.
    sig — необязательный готовый сигнал (для walk-forward, чтобы не терять прогрев
    индикаторов на границах окон). Если None — считается из генома.
    """
    risk = cfg["risk"]
    costs = cfg["costs"]
    fee = costs["fee_pct"]
    slip = costs["slippage_pct"]

    # Задержка исполнения: реагируем на сигнал УЖЕ ЗАКРЫТОГО бара (не мгновенно).
    # Это убирает зависимость стратегии от скорости доступа к бирже.
    if sig is None:
        delay = cfg.get("execution", {}).get("signal_delay_bars", 1)
        sig = gn.signal(genome, df).shift(delay).fillna(0).astype(int)
    else:
        sig = sig.reindex(df.index).fillna(0).astype(int)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    cash = 1.0            # нормированный капитал
    units = 0.0           # объём в активе
    in_pos = False
    entry_price = 0.0
    peak_price = 0.0      # для trailing stop (ratcheting)

    equity_curve = []
    period_returns = []
    trade_results = []
    prev_equity = cash

    for i in range(n):
        price = close[i]

        # --- управление открытой позицией: риск имеет приоритет над сигналом ---
        exit_reason = None
        if in_pos:
            peak_price = max(peak_price, high[i])

            stop_price = entry_price * (1 - risk["stop_loss_pct"])
            trail_price = peak_price * (1 - risk["trailing_stop_pct"])
            take_price = entry_price * (1 + risk["take_profit_pct"])
            effective_stop = max(stop_price, trail_price)  # ratcheting: стоп только растёт

            if low[i] <= effective_stop:
                exit_price = effective_stop
                exit_reason = "stop_loss" if effective_stop == stop_price else "trailing"
            elif high[i] >= take_price:
                exit_price = take_price
                exit_reason = "take_profit"
            elif sig.iloc[i] == 0:
                exit_price = price
                exit_reason = "signal"

            if exit_reason:
                fill = exit_price * (1 - slip)            # проскальзывание против нас
                proceeds = units * fill * (1 - fee)       # минус комиссия
                pnl = proceeds - (units * entry_price)
                trade_results.append(pnl)
                cash += proceeds                          # возврат в общий кэш (не затираем остаток!)
                units = 0.0
                in_pos = False

        # --- вход по сигналу ---
        if (not in_pos) and sig.iloc[i] == 1:
            fill = price * (1 + slip)                      # проскальзывание против нас
            invest = cash * risk["position_fraction"]
            units = (invest * (1 - fee)) / fill
            cash -= invest
            entry_price = fill
            peak_price = high[i]
            in_pos = True

        # --- учёт текущего капитала ---
        equity = cash + units * price
        equity_curve.append(equity)
        period_returns.append(equity / prev_equity - 1 if prev_equity else 0.0)
        prev_equity = equity

    eq = pd.Series(equity_curve, index=df.index)
    rets = pd.Series(period_returns, index=df.index)
    m = mt.compute_metrics(eq, rets, trade_results, genome["timeframe"])
    m["equity"] = eq
    return m


def split_train_test(df: pd.DataFrame, train_ratio: float):
    """Разделение на in-sample / out-of-sample. Агент при отборе видит только train."""
    cut = int(len(df) * train_ratio)
    return df.iloc[:cut], df.iloc[cut:]


def walk_forward_eval(genome: dict, df: pd.DataFrame, cfg: dict):
    """
    WALK-FORWARD валидация (совет из треда против переобучения).

    Первая половина данных = in-sample (фитнес отбора). Вторая половина режется
    на N окон, и агент проверяется в КАЖДОМ окне отдельно. Хорошая стратегия
    прибыльна в большинстве окон, а не в одном удачном.

    Возвращает (train_metrics, test_metrics_aggregated, robustness), где
    robustness = доля out-of-sample окон с положительной доходностью (0..1).
    Эта robustness используется как метрика consistency при отборе.
    """
    import numpy as np
    delay = cfg.get("execution", {}).get("signal_delay_bars", 1)
    full_sig = gn.signal(genome, df).shift(delay).fillna(0).astype(int)

    cut = int(len(df) * cfg["train_ratio"])
    train_df = df.iloc[:cut]
    oos_df = df.iloc[cut:]

    train_m = run(genome, train_df, cfg, sig=full_sig.iloc[:cut])

    nwin = cfg.get("validation", {}).get("walk_forward_windows", 4)
    idx = list(range(len(oos_df)))
    windows = [w for w in np.array_split(idx, nwin) if len(w) >= 50]

    results = []
    for w in windows:
        a, b = int(w[0]), int(w[-1]) + 1
        seg = oos_df.iloc[a:b]
        seg_sig = full_sig.iloc[cut + a:cut + b]
        results.append(run(genome, seg, cfg, sig=seg_sig))

    if not results:  # данных мало — откат на единый OOS
        test_m = run(genome, oos_df, cfg, sig=full_sig.iloc[cut:])
        return train_m, test_m, 0.0

    test_m = {
        "sharpe": round(float(np.mean([r["sharpe"] for r in results])), 3),
        "total_return": round(float(np.mean([r["total_return"] for r in results])), 4),
        "win_rate": round(float(np.mean([r["win_rate"] for r in results])), 3),
        "num_trades": int(sum(r["num_trades"] for r in results)),
        "max_drawdown": round(float(max(r["max_drawdown"] for r in results)), 4),
    }
    robustness = round(sum(1 for r in results if r["total_return"] > 0) / len(results), 3)
    return train_m, test_m, robustness
