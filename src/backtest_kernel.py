"""Strict scalar simulation for research; the detailed Python backtest is the oracle."""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def simulate(opened, close, high, low, signals, atr, fee, slip, cooldown,
             atr_stop, stop_mult, take_mult, trail_mult, stop_pct, take_pct,
             trail_pct, fraction, vol_target, risk_per_trade):
    n = len(close)
    equities = np.empty(n)
    returns = np.empty(n)
    trades = np.empty(n)
    count = 0
    cash = previous = 1.0
    direction = 0
    entry = notional = extreme = 0.0
    atr_entry = np.nan
    cooldown_until = 0
    for i in range(n):
        price, signal = close[i], signals[i]
        closed = False
        if direction:
            if atr_stop and np.isfinite(atr_entry) and atr_entry > 0:
                stop, take, trail = stop_mult * atr_entry, take_mult * atr_entry, trail_mult * atr_entry
            else:
                stop, take, trail = entry * stop_pct, entry * take_pct, extreme * trail_pct
            if direction == 1:
                stop, take = max(entry - stop, extreme - trail), entry + take
            else:
                stop, take = min(entry + stop, extreme + trail), entry - take
            exit_price = 0.0
            if (direction == 1 and opened[i] <= stop) or (direction == -1 and opened[i] >= stop):
                exit_price, closed = opened[i], True
            elif (direction == 1 and opened[i] >= take) or (direction == -1 and opened[i] <= take):
                exit_price, closed = take, True
            elif (direction == 1 and low[i] <= stop) or (direction == -1 and high[i] >= stop):
                exit_price, closed = stop, True
            elif (direction == 1 and high[i] >= take) or (direction == -1 and low[i] <= take):
                exit_price, closed = take, True
            elif signal != direction:
                exit_price, closed = price, True
            if closed:
                if not np.isfinite(exit_price) or exit_price <= 0:
                    raise ValueError("invalid fill reference or side")
                execution = exit_price * (1 + slip * -direction)
                gross = direction * (execution / entry - 1)
                pnl = notional * gross - notional * (execution / entry) * fee
                trades[count] = pnl - notional * fee
                count += 1
                cash += notional + pnl
                direction = 0
                cooldown_until = i + max(1, cooldown)
            else:
                extreme = max(extreme, high[i]) if direction == 1 else min(extreme, low[i])
        if not direction and not closed and signal != 0 and i >= cooldown_until:
            direction = signal
            sized = fraction
            if vol_target and np.isfinite(atr[i]) and atr[i] > 0 and price > 0:
                relative = atr[i] / price
                if stop_mult > 0 and relative > 0:
                    sized = max(0.0, min(risk_per_trade / (stop_mult * relative), fraction))
            notional = min(cash * sized, cash / (1 + fee))
            cash -= notional + notional * fee
            entry = price * (1 + slip * direction)
            extreme, atr_entry = entry, atr[i]
        equity = cash + notional + notional * (direction * (price / entry - 1)) if direction else cash
        equities[i] = equity
        returns[i] = equity / previous - 1 if previous else 0.0
        previous = equity
    return equities, returns, trades[:count]


def eligible(opened, close, high, low, signals, fee, slip):
    return (np.isfinite(fee) and fee >= 0 and np.isfinite(slip) and 0 <= slip < 1
            and np.isin(signals, [-1, 0, 1]).all()
            and all(np.isfinite(values).all() and (values > 0).all() for values in (opened, close, high, low)))


def run_arrays(opened, close, high, low, signals, atr, risk, fee, slip, cooldown):
    return simulate(opened.astype(float), close.astype(float), high.astype(float), low.astype(float),
                    signals, np.asarray(atr, dtype=float), fee, slip, cooldown,
                    bool(risk.get("atr_stop")), risk.get("atr_stop_mult", 2.0),
                    risk.get("atr_take_mult", 6.0), risk.get("atr_trail_mult", 2.5),
                    risk["stop_loss_pct"], risk["take_profit_pct"], risk["trailing_stop_pct"],
                    risk["position_fraction"], bool(risk.get("vol_target")),
                    risk.get("risk_per_trade", 0.005))
