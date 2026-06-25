"""
Риск-менеджмент. Принцип из треда — повторённый почти всеми:
"Управление рисками — это ВСЁ. Ты сольёшь свой счёт без него. Даже с лучшим
преимуществом, плохие риски сольют твой счёт." (milic_001, Downtown-Feeling-206)

Этот модуль управляет ЖИВЫМИ (бумажными) позициями: размер позиции,
лимит одновременных позиций, расчёт цен выхода с ratcheting trailing stop.
"""


class Position:
    """
    Открытая бумажная позиция с динамическими уровнями выхода.
    direction: +1 = long (зарабатываем на росте), -1 = short (на падении).
    notional = вложенный капитал (маржа). units = notional/entry_price (для логов qty).
    atr = ATR на момент входа (для стопа от волатильности, если включён).
    """
    def __init__(self, agent_id, symbol, entry_price, units,
                 direction=1, notional=None, atr=None,
                 stop_mult=None, take_mult=None, trail_mult=None):
        self.agent_id = agent_id
        self.symbol = symbol
        self.entry_price = entry_price
        self.units = units
        self.direction = direction
        self.notional = notional if notional is not None else units * entry_price
        self.atr = atr
        # персональные гены риска агента (если заданы) — иначе берём из config
        self.stop_mult = stop_mult
        self.take_mult = take_mult
        self.trail_mult = trail_mult
        # экстремум в нашу пользу: пик (long) или дно (short) — для ratcheting trail
        self.peak_price = entry_price

    def update_extreme(self, high, low):
        if self.direction == 1:
            self.peak_price = max(self.peak_price, high)
        else:
            self.peak_price = min(self.peak_price, low)

    def _levels(self, risk_cfg):
        """Абсолютные (effective_stop, take) с учётом направления и ATR/процентов."""
        e = self.entry_price
        use_atr = risk_cfg.get("atr_stop") and self.atr and self.atr > 0
        if use_atr:
            sm = self.stop_mult if self.stop_mult else risk_cfg.get("atr_stop_mult", 2.0)
            tm = self.take_mult if self.take_mult else risk_cfg.get("atr_take_mult", 6.0)
            trm = self.trail_mult if self.trail_mult else risk_cfg.get("atr_trail_mult", 2.5)
            s_off = sm * self.atr
            t_off = tm * self.atr
            tr_off = trm * self.atr
        else:
            s_off = e * risk_cfg["stop_loss_pct"]
            t_off = e * risk_cfg["take_profit_pct"]
            tr_off = self.peak_price * risk_cfg["trailing_stop_pct"]
        if self.direction == 1:
            return max(e - s_off, self.peak_price - tr_off), e + t_off
        else:
            return min(e + s_off, self.peak_price + tr_off), e - t_off

    def value(self, price):
        """Текущая стоимость позиции (mark-to-market) для расчёта капитала."""
        return self.notional * (1 + self.direction * (price / self.entry_price - 1))

    def exit_check_hl(self, high, low, price, risk_cfg):
        """
        Внутрибарная проверка по экстремумам — для живого режима, чтобы стоп
        срабатывал между тиками как настоящий резервный ордер.
        Возвращает (надо_выходить, причина, цена_выхода). Стоп приоритетнее тейка.
        """
        self.update_extreme(high, low)
        eff_stop, take = self._levels(risk_cfg)
        if self.direction == 1:
            if low <= eff_stop:
                return True, "stop", eff_stop
            if high >= take:
                return True, "take_profit", take
        else:
            if high >= eff_stop:
                return True, "stop", eff_stop
            if low <= take:
                return True, "take_profit", take
        return False, None, price

    def exit_check(self, price, risk_cfg):
        """Проверка по цене закрытия (для бар-уровневой бумажной торговли)."""
        return self.exit_check_hl(price, price, price, risk_cfg)


def close_pnl(pos: "Position", exit_fill: float, fee: float) -> float:
    """Реализованный PnL при закрытии (работает для long и short).
    exit_fill — цена выхода уже с учётом проскальзывания."""
    gross = pos.direction * pos.units * (exit_fill - pos.entry_price)
    fees = (pos.notional + pos.units * exit_fill) * fee
    return gross - fees


def sized_fraction(risk_cfg: dict, atr=None, price=None) -> float:
    """
    ВОЛАТИЛЬНОСТЬ-ТАРГЕТИНГ (risk parity по позиции).
    Доля капитала на сделку считается так, чтобы РИСК ДО СТОПА был одинаковым для
    любой монеты/таймфрейма: спокойная монета → больше, дёрганая → меньше.

    risk_per_trade — целевой риск (доля капитала), который мы теряем, если цена
    дойдёт до стопа. Стоп = stop_mult × ATR, значит относительный риск позиции =
    stop_mult × (ATR/price). Отсюда доля = risk_per_trade / (stop_mult × ATR/price).
    position_fraction работает как ПОТОЛОК (не вкладываем больше него).

    Если vol_target выключен или нет ATR — откат на фиксированную position_fraction.
    """
    cap = risk_cfg["position_fraction"]   # потолок доли на сделку
    # atr != atr → NaN (ранние бары, где ATR ещё не посчитан) → откат на потолок
    if (not risk_cfg.get("vol_target") or atr is None or price is None
            or atr != atr or price != price or atr <= 0 or price <= 0):
        return cap
    stop_mult = risk_cfg.get("atr_stop_mult", 2.0)
    rpt = risk_cfg.get("risk_per_trade", 0.005)
    rel_vol = atr / price
    if stop_mult <= 0 or rel_vol <= 0:
        return cap
    raw = rpt / (stop_mult * rel_vol)
    return max(0.0, min(raw, cap))


def position_size(capital: float, risk_cfg: dict, atr=None, price=None) -> float:
    """Сколько денег вложить в одну позицию (с учётом волатильность-таргетинга)."""
    return capital * sized_fraction(risk_cfg, atr, price)


def can_open(open_positions: int, risk_cfg: dict) -> bool:
    """Не превышаем лимит одновременных позиций."""
    return open_positions < risk_cfg["max_open_positions"]
