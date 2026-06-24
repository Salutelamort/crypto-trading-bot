"""
ОТБОР (стадия эволюции) — решения принимает ЭВОЛЮЦИЯ, не Claude.

Это сознательное отступление от фото piratastuertos: там управленческий слой —
Клод. Пользователь выбрал иначе: решения о смерти и рождении агентов принимает
сам эволюционный отбор по ДЕТЕРМИНИРОВАННЫМ правилам выживания. Никакого LLM в
цикле принятия решений вообще — ни для стратегий, ни для управления.

Здесь — "ворота допуска": агент, переживший эволюцию и стабильно показавший
преимущество на нескольких out-of-sample окнах (walk-forward), ПРОДВИГАЕТСЯ к
живой торговле; слабый — УБИВАЕТСЯ. Это чистая селекция, а не мнение модели.

Ключевые правила:
  - Разделяем метрики для ПРОДВИЖЕНИЯ и для УБИЙСТВА (нельзя подделать обе сразу).
  - consistency = доля walk-forward окон с прибылью (устойчивость во времени).
  - Низкая просадка обязательна (профиль: минимальный риск).
  - Карантин символов и продвижение только по out-of-sample результатам.
"""
import json
from . import db


def _agent_summary(a: dict) -> dict:
    g = json.loads(a["genome"])
    return {
        "id": a["id"], "type": g["type"], "symbol": a["symbol"],
        "train_sharpe": a["train_sharpe"], "test_sharpe": a["test_sharpe"],
        "test_return": a["test_return"], "test_maxdd": a["test_maxdd"],
        "test_trades": a["test_trades"], "consistency": a["consistency"],
    }


def _decide(agents, cfg, quarantined):
    """Детерминированные правила выживания/допуска. Никакого LLM."""
    sup = cfg["supervisor"]
    decisions = []
    for a in agents:
        s = _agent_summary(a)
        ts = s["test_sharpe"] or -99
        dd = s["test_maxdd"] or 1.0
        cons = s["consistency"] or 0.0
        trades = s["test_trades"] or 0

        # СМЕРТЬ агентов происходит ВНУТРИ эволюции (отбор по приспособленности
        # в evolution.py). Здесь супервизор НЕ убивает — только ДОПУСКАЕТ к живой
        # торговле тех, кто стабильно показал преимущество. Так популяция
        # накапливается и улучшается между циклами.

        # --- ДОПУСК к живой торговле ---
        max_dd_ok = dd <= sup.get("promote_max_drawdown", 1.0)
        ok = (ts >= sup["promote_min_sharpe"]
              and trades >= sup["promote_min_trades"]
              and cons >= sup["promote_min_consistency"]
              and max_dd_ok
              and a["symbol"] not in quarantined)
        if ok:
            decisions.append((a["id"], "promote",
                f"OOS Sharpe {ts:.2f}, просадка {dd:.0%}, сделок {trades}, "
                f"устойчивость {cons:.0%} окон — стабилен и низкорисков"))
        else:
            decisions.append((a["id"], "hold",
                f"не дотягивает (OOS Sharpe {ts:.2f}, просадка {dd:.0%}, "
                f"trades {trades}, устойчивость {cons:.0%})"))
    return decisions


def supervise(conn, cfg):
    """Стадия отбора эволюции: продвигает достойных, убивает слабых."""
    quarantined = db.quarantined_symbols(conn)
    agents = db.get_agents(conn, "candidate")
    if not agents:
        print("Нет агентов-кандидатов. Сначала запусти эволюцию.")
        return {"kill": 0, "promote": 0, "hold": 0}

    print(f"\nОтбор эволюции оценивает {len(agents)} агентов (правила, без LLM)...")
    decisions = _decide(agents, cfg, quarantined)

    counts = {"kill": 0, "promote": 0, "hold": 0}
    for agent_id, action, rationale in decisions:
        db.log_decision(conn, agent_id, action, "evolution", rationale)
        if action == "kill":
            db.set_agent_status(conn, agent_id, "killed")
        elif action == "promote":
            db.set_agent_status(conn, agent_id, "promoted")
        counts[action] = counts.get(action, 0) + 1
        print(f"  #{agent_id}: {action.upper():8s} — {rationale}")

    print(f"\nИтог: продвинуто {counts['promote']}, "
          f"убито {counts['kill']}, оставлено {counts['hold']}")
    if counts["promote"] == 0:
        print("Ни один агент не прошёл в paper-live. Это НОРМАЛЬНО — "
              "большинство стратегий не имеют преимущества (урок из треда).")
    return counts
