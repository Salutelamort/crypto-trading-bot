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


def run(genome: dict, df: pd.DataFrame, cfg: dict) -> dict:
    """
    Симулирует одного агента на исторических данных.
    Возвращает словарь метрик + кривую equity.
    """
    risk = cfg["risk"]
    costs = cfg["costs"]
    fee = costs["fee_pct"]
    slip = costs["slippage_pct"]

    # Задержка исполнения: реагируем на сигнал УЖЕ ЗАКРЫТОГО бара (не мгновенно).
    # Это убирает зависимость стратегии от скорости доступа к бирже.
    delay = cfg.get("execution", {}).get("signal_delay_bars", 1)
    sig = gn.signal(genome, df).shift(delay).fillna(0).astype(int)
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
