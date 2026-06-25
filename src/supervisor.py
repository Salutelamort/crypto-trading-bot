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
        alpha = a["test_alpha"] if a["test_alpha"] is not None else -99

        # СМЕРТЬ агентов происходит ВНУТРИ эволюции (отбор по приспособленности
        # в evolution.py). Здесь супервизор НЕ убивает — только ДОПУСКАЕТ к живой
        # торговле тех, кто стабильно показал преимущество. Так популяция
        # накапливается и улучшается между циклами.

        # --- ДОПУСК к живой торговле ---
        # База качества (общая для обоих путей): достаточно сделок, низкая просадка,
        # устойчивый обгон рынка по окнам, символ не в карантине.
        base_ok = (trades >= sup["promote_min_trades"]
                   and cons >= sup["promote_min_consistency"]
                   and dd <= sup.get("promote_max_drawdown", 1.0)
                   and a["symbol"] not in quarantined)
        # Достаточно ЛИБО хорошего Sharpe (гладкая кривая), ЛИБО обгона рынка (alpha).
        edge_ok = (ts >= sup["promote_min_sharpe"]
                   or alpha >= sup.get("promote_min_alpha", 99))
        if base_ok and edge_ok:
            decisions.append((a["id"], "promote",
                f"OOS Sharpe {ts:.2f}, alpha {alpha:+.1%} vs рынок, просадка {dd:.0%}, "
                f"сделок {trades}, устойчивость {cons:.0%} окон — допущен"))
        else:
            decisions.append((a["id"], "hold",
                f"не дотягивает (OOS Sharpe {ts:.2f}, alpha {alpha:+.1%}, "
                f"просадка {dd:.0%}, trades {trades}, устойчивость {cons:.0%})"))
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

    # --- ДИВЕРСИФИКАЦИЯ на уровне допуска ---
    # Даже после анти-клона ограничиваем концентрацию: не больше N стратегий на
    # один символ и не больше M всего. Из прошедших ворота берём лучших по alpha,
    # чтобы живой портфель был распределён, а не сидел в одной монете.
    sup = cfg["supervisor"]
    max_per_sym = sup.get("promote_max_per_symbol", 2)
    max_total = sup.get("promote_max_total", 6)
    by_id = {a["id"]: a for a in agents}
    promo = [d for d in decisions if d[1] == "promote"]
    promo.sort(key=lambda d: (by_id[d[0]]["test_alpha"] if by_id[d[0]]["test_alpha"] is not None else -99),
               reverse=True)
    per_sym, taken = {}, 0
    allowed = set()
    for d in promo:
        sym = by_id[d[0]]["symbol"]
        if taken >= max_total or per_sym.get(sym, 0) >= max_per_sym:
            continue
        allowed.add(d[0])
        per_sym[sym] = per_sym.get(sym, 0) + 1
        taken += 1
    # прошедшие ворота, но «лишние» по диверсификации → переводим в hold
    decisions = [
        d if not (d[1] == "promote" and d[0] not in allowed)
        else (d[0], "hold", d[2] + " | не допущен: лимит диверсификации (концентрация)")
        for d in decisions
    ]

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
