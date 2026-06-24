"""
Риск-менеджмент. Принцип из треда — повторённый почти всеми:
"Управление рисками — это ВСЁ. Ты сольёшь свой счёт без него. Даже с лучшим
преимуществом, плохие риски сольют твой счёт." (milic_001, Downtown-Feeling-206)

Этот модуль управляет ЖИВЫМИ (бумажными) позициями: размер позиции,
лимит одновременных позиций, расчёт цен выхода с ratcheting trailing stop.
"""


class Position:
    """Открытая бумажная позиция с динамическими уровнями выхода."""
    def __init__(self, agent_id, symbol, entry_price, units):
        self.agent_id = agent_id
        self.symbol = symbol
        self.entry_price = entry_price
        self.units = units
        self.peak_price = entry_price   # для ratcheting trailing stop

    def update_peak(self, price):
        self.peak_price = max(self.peak_price, price)

    def exit_check(self, price, risk_cfg):
        """Возвращает (надо_ли_выходить, причина). Риск > сигнала."""
        self.update_peak(price)
        stop = self.entry_price * (1 - risk_cfg["stop_loss_pct"])
        trail = self.peak_price * (1 - risk_cfg["trailing_stop_pct"])
        take = self.entry_price * (1 + risk_cfg["take_profit_pct"])
        effective_stop = max(stop, trail)   # ratcheting: стоп подтягивается за ценой

        if price <= effective_stop:
            return True, ("stop_loss" if effective_stop == stop else "trailing")
        if price >= take:
            return True, "take_profit"
        return False, None

    def exit_check_hl(self, high, low, price, risk_cfg):
        """
        Внутрибарная проверка по экстремумам (high/low) — для живого режима, чтобы
        стоп срабатывал на движениях МЕЖДУ тиками, как настоящий резервный ордер,
        а не только по цене закрытия. Возвращает (надо_выходить, причина, цена_выхода).
        """
        self.update_peak(high)
        stop = self.entry_price * (1 - risk_cfg["stop_loss_pct"])
        trail = self.peak_price * (1 - risk_cfg["trailing_stop_pct"])
        take = self.entry_price * (1 + risk_cfg["take_profit_pct"])
        effective_stop = max(stop, trail)

        # стоп приоритетнее тейка (консервативно — сначала защита от убытка)
        if low <= effective_stop:
            return True, ("stop_loss" if effective_stop == stop else "trailing"), effective_stop
        if high >= take:
            return True, "take_profit", take
        return False, None, price


def position_size(capital: float, risk_cfg: dict) -> float:
    """Сколько денег вложить в одну позицию (доля капитала)."""
    return capital * risk_cfg["position_fraction"]


def can_open(open_positions: int, risk_cfg: dict) -> bool:
    """Не превышаем лимит одновременных позиций."""
    return open_positions < risk_cfg["max_open_positions"]
