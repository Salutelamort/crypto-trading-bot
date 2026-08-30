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
from . import indicators as ind
from . import metrics as mt
from . import risk as rk


def _effective_risk(genome, risk):
    """Накладывает ГЕНЫ РИСКА агента поверх базового config (если они есть).
    Так каждый агент носит собственные стоп/тейк/трейл в единицах ATR,
    а эволюция их подбирает (совет Карвера: риск от волатильности)."""
    r = dict(risk)
    if genome.get("stop_atr"):
        r["atr_stop_mult"] = genome["stop_atr"]
    if genome.get("trail_atr"):
        r["atr_trail_mult"] = genome["trail_atr"]
    rr = risk.get("fixed_rr", genome.get("rr"))
    if genome.get("stop_atr") and rr:
        r["atr_take_mult"] = round(genome["stop_atr"] * rr, 3)
    return r


def _exit_levels(direction, entry, extreme, atr_val, risk):
    """
    Абсолютные уровни выхода для позиции в направлении direction (+1 long / -1 short).
    extreme — пик (для long) или дно (для short) цены с момента входа (ratcheting trail).
    Если включён ATR-стоп и atr_val валиден — стоп/тейк/трейл считаются от
    волатильности (k*ATR), иначе — от фиксированных процентов (старое поведение).
    Возвращает (effective_stop, take_price) уже как цены.
    """
    use_atr = risk.get("atr_stop") and atr_val and atr_val > 0
    if use_atr:
        s_off = risk.get("atr_stop_mult", 2.0) * atr_val
        t_off = risk.get("atr_take_mult", 4.0) * atr_val
        tr_off = risk.get("atr_trail_mult", 2.5) * atr_val
    else:
        s_off = entry * risk["stop_loss_pct"]
        t_off = entry * risk["take_profit_pct"]
        tr_off = extreme * risk["trailing_stop_pct"]

    if direction == 1:
        stop = entry - s_off
        trail = extreme - tr_off
        take = entry + t_off
        return max(stop, trail), take          # ratcheting вверх
    else:  # short
        stop = entry + s_off
        trail = extreme + tr_off
        take = entry - t_off
        return min(stop, trail), take          # ratcheting вниз


def run(genome: dict, df: pd.DataFrame, cfg: dict, sig=None) -> dict:
    """
    Симулирует одного агента на исторических данных. Поддерживает ТРИ состояния:
    long (+1), short (-1), кэш (0). Возвращает словарь метрик + кривую equity.
    sig — необязательный готовый сигнал (для walk-forward, чтобы не терять прогрев
    индикаторов на границах окон). Если None — считается из генома.
    """
    risk = _effective_risk(genome, cfg["risk"])
    costs = cfg["costs"]
    fee = costs["fee_pct"]
    slip = costs["slippage_pct"]
    allow_short = risk.get("allow_short", False)
    cooldown = int(genome.get("cooldown", 0))   # баров «отдыха» после выхода

    # Задержка исполнения: реагируем на сигнал УЖЕ ЗАКРЫТОГО бара (не мгновенно).
    if sig is None:
        delay = cfg.get("execution", {}).get("signal_delay_bars", 1)
        sig = gn.signal(genome, df, allow_short).shift(delay).fillna(0).astype(int)
    else:
        sig = sig.reindex(df.index).fillna(0).astype(int)

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    # ОПТИМИЗАЦИЯ: работаем с numpy-массивами в цикле. sig.iloc[i] (доступ pandas
    # по элементу) был главным тормозом — переходим на sig_arr[i]. Логика та же.
    sig_arr = sig.to_numpy()
    n = len(df)
    atr_arr = ind.atr(df, risk.get("atr_period", 14)).values if risk.get("atr_stop") else [None] * n

    cash = 1.0
    in_pos = False
    direction = 0
    entry_exec = 0.0      # цена входа с проскальзыванием
    notional = 0.0        # вложенный капитал (маржа)
    extreme = 0.0         # пик(long)/дно(short) с момента входа — для trailing
    atr_entry = None

    equity_curve = []
    period_returns = []
    trade_results = []
    prev_equity = cash
    cooldown_until = 0    # до этого бара новые входы запрещены (анти-переторговля)

    def close_pos(exit_price, reason):
        nonlocal cash, in_pos, direction, cooldown_until
        exit_exec = exit_price * (1 - slip * direction)       # слиппедж всегда против нас
        gross = direction * (exit_exec / entry_exec - 1)      # доходность с учётом стороны
        pnl = notional * gross - notional * (exit_exec / entry_exec) * fee
        trade_results.append(pnl)
        cash += notional + pnl
        in_pos = False
        direction = 0
        cooldown_until = i + cooldown

    for i in range(n):
        price = close[i]
        s = int(sig_arr[i])
        closed_this_bar = False

        # --- управление открытой позицией: риск приоритетнее сигнала ---
        if in_pos:
            # OHLC не хранит порядок high/low. Уровни берём с предыдущего бара;
            # новый экстремум применяем только если позиция пережила текущий.
            eff_stop, take = _exit_levels(direction, entry_exec, extreme, atr_entry, risk)
            if direction == 1:
                if low[i] <= eff_stop:
                    close_pos(eff_stop, "stop")
                    closed_this_bar = True
                elif high[i] >= take:
                    close_pos(take, "take_profit")
                    closed_this_bar = True
                elif s != 1:
                    close_pos(price, "signal")
                    closed_this_bar = True
                else:
                    extreme = max(extreme, high[i])
            else:  # short
                if high[i] >= eff_stop:
                    close_pos(eff_stop, "stop")
                    closed_this_bar = True
                elif low[i] <= take:
                    close_pos(take, "take_profit")
                    closed_this_bar = True
                elif s != -1:
                    close_pos(price, "signal")
                    closed_this_bar = True
                else:
                    extreme = min(extreme, low[i])

        # --- вход по сигналу (после кулдауна; разворот тоже ждёт «отдыха») ---
        if (not in_pos) and not closed_this_bar and s != 0 and i >= cooldown_until:
            direction = s
            # волатильность-таргетинг: доля от риска до стопа (risk parity)
            frac_eff = rk.sized_fraction(risk, atr_arr[i], price)
            notional = min(cash * frac_eff, cash / (1 + fee))
            cash -= notional + notional * fee
            entry_exec = price * (1 + slip * direction)
            # Вход исполняется на close; high/low этой свечи уже в прошлом.
            extreme = entry_exec
            atr_entry = atr_arr[i] if atr_arr[i] is not None else None
            in_pos = True

        # --- учёт текущего капитала ---
        if in_pos:
            unreal = notional * (direction * (price / entry_exec - 1))
            equity = cash + notional + unreal
        else:
            equity = cash
        equity_curve.append(equity)
        period_returns.append(equity / prev_equity - 1 if prev_equity else 0.0)
        prev_equity = equity

    eq = pd.Series(equity_curve, index=df.index)
    rets = pd.Series(period_returns, index=df.index)
    buy_hold = float(close[-1] / close[0] - 1) if n else 0.0
    m = mt.compute_metrics(eq, rets, trade_results, genome["timeframe"], buy_hold)
    m["equity"] = eq
    return m


def split_train_test(df: pd.DataFrame, train_ratio: float):
    """Разделение train/validation (validation повторно используется поиском)."""
    cut = int(len(df) * train_ratio)
    return df.iloc[:cut], df.iloc[cut:]


def walk_forward_eval(genome: dict, df: pd.DataFrame, cfg: dict):
    """
    WALK-FORWARD валидация (совет из треда против переобучения).

    Первая половина данных = train (фитнес отбора). Вторая половина режется
    на N окон, и агент проверяется в КАЖДОМ окне отдельно. Хорошая стратегия
    прибыльна в большинстве окон, а не в одном удачном.

    Возвращает (train_metrics, test_metrics_aggregated, robustness), где
    robustness = доля validation-окон с положительной alpha (0..1).
    Эта robustness используется как метрика consistency при отборе.
    """
    import numpy as np
    delay = cfg.get("execution", {}).get("signal_delay_bars", 1)
    allow_short = cfg["risk"].get("allow_short", False)
    full_sig = gn.signal(genome, df, allow_short).shift(delay).fillna(0).astype(int)

    cut = int(len(df) * cfg["train_ratio"])
    # EMBARGO (López de Prado, purged CV): зазор между train и test, чтобы индикаторы
    # на границе не "подсматривали" данные обучения (утечка). Пропускаем N баров.
    embargo = int(cfg.get("validation", {}).get("embargo_bars", 0))
    oos_start = min(cut + embargo, len(df))
    train_df = df.iloc[:cut]
    oos_df = df.iloc[oos_start:]

    train_m = run(genome, train_df, cfg, sig=full_sig.iloc[:cut])

    nwin = cfg.get("validation", {}).get("walk_forward_windows", 4)
    idx = list(range(len(oos_df)))
    windows = [w for w in np.array_split(idx, nwin) if len(w) >= 50]

    # Итоговые OOS-метрики считаются единым непрерывным прогоном. Усреднять
    # Sharpe/PF/доходности по окнам математически некорректно.
    test_m = run(genome, oos_df, cfg, sig=full_sig.iloc[oos_start:])
    results = []
    for w in windows:
        a, b = int(w[0]), int(w[-1]) + 1
        seg = oos_df.iloc[a:b]
        seg_sig = full_sig.iloc[oos_start + a:oos_start + b]
        results.append(run(genome, seg, cfg, sig=seg_sig))

    if not results:  # данных мало — единый OOS без оценки устойчивости
        return train_m, test_m, 0.0

    # Агент, который НЕ торгует в validation, бесполезен → худший балл, не 0.0.
    if test_m["num_trades"] == 0:
        test_m = {"sharpe": -99.0, "sortino": -99.0, "calmar": -99.0,
                  "profit_factor": 0.0, "total_return": 0.0, "win_rate": 0.0,
                  "num_trades": 0, "max_drawdown": 0.0, "buy_hold": 0.0, "alpha": 0.0}
        return train_m, test_m, 0.0

    # УСТОЙЧИВОСТЬ = доля окон, где агент ОБОГНАЛ "купи и держи" (alpha > 0).
    # Так ценится не только заработок, но и сохранение капитала в падающем рынке:
    # потерять -5%, когда рынок упал -40%, — это победа (alpha = +35%).
    # Окно без сделок — провал, а не исчезнувшее наблюдение.
    robustness = round(
        sum(1 for r in results if r["num_trades"] > 0 and r["alpha"] > 0)
        / len(results), 3)
    return train_m, test_m, robustness
