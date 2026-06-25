"""
ЗАЩИТЫ (protections) — паттерн из Freqtrade. Поверх обычного риск-менеджмента
добавляют «предохранители», которые ПАУЗЯТ торговлю при плохой полосе. Это прямой
ответ на правило из чата ("серия убытков → стоп") и на профиль низкого риска.

Реализация СТАТУСНАЯ по скользящему окну (без отдельных таблиц состояния):
предохранитель активен, пока в окне последних N часов выполняется плохое условие;
по мере «старения» убытков за пределы окна — снимается автоматически.

Никакого LLM — чистый SQL по журналу сделок paper_trades.
"""
from datetime import datetime, timezone, timedelta

# закрывающие сделки (по ним считаем реализованный PnL)
_CLOSE_SIDES = ("SELL", "COVER")


def _cutoff_iso(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def stoploss_guard(conn, cfg):
    """
    Если за окно последних N часов набралось >= max_losses убыточных закрытий —
    ПАУЗА всех новых входов (рынок против нас, не долбим).
    Возвращает (blocked: bool, reason: str).
    """
    p = cfg.get("protections", {}).get("stoploss_guard", {})
    if not p.get("enabled"):
        return False, ""
    window = p.get("window_hours", 24)
    max_losses = p.get("max_losses", 4)
    cutoff = _cutoff_iso(window)
    n = conn.execute(
        f"SELECT COUNT(*) FROM paper_trades "
        f"WHERE side IN {_CLOSE_SIDES} AND pnl < 0 AND ts >= ?",
        (cutoff,)).fetchone()[0]
    if n >= max_losses:
        return True, f"StoplossGuard: {n} убыточных закрытий за {window}ч (>= {max_losses})"
    return False, ""


def locked_symbols(conn, cfg) -> set:
    """
    Символы, временно заблокированные для входов: суммарный реализованный PnL по
    символу за окно последних N часов хуже порога (доля стартового капитала).
    Возвращает множество символов под блокировкой.
    """
    p = cfg.get("protections", {}).get("symbol_lock", {})
    if not p.get("enabled"):
        return set()
    window = p.get("window_hours", 72)
    max_loss_frac = p.get("max_loss_frac", 0.03)
    start_cap = float(cfg.get("paper", {}).get("starting_capital", 10000))
    limit = -abs(max_loss_frac) * start_cap
    cutoff = _cutoff_iso(window)
    rows = conn.execute(
        f"SELECT symbol, COALESCE(SUM(pnl),0) AS net FROM paper_trades "
        f"WHERE side IN {_CLOSE_SIDES} AND ts >= ? GROUP BY symbol",
        (cutoff,)).fetchall()
    return {r["symbol"] for r in rows if r["net"] <= limit}
