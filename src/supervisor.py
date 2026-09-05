"""
ОТБОР (стадия эволюции) — решения принимает ЭВОЛЮЦИЯ, не Claude.

Это сознательное отступление от фото piratastuertos: там управленческий слой —
Клод. Пользователь выбрал иначе: решения о смерти и рождении агентов принимает
сам эволюционный отбор по ДЕТЕРМИНИРОВАННЫМ правилам выживания. Никакого LLM в
цикле принятия решений вообще — ни для стратегий, ни для управления.

Здесь — "ворота допуска": агент, переживший эволюцию и стабильно показавший
преимущество на нескольких validation-окнах (walk-forward), ПРОДВИГАЕТСЯ к
живой торговле; слабый — УБИВАЕТСЯ. Это чистая селекция, а не мнение модели.

Ключевые правила:
  - Разделяем метрики для ПРОДВИЖЕНИЯ и для УБИЙСТВА (нельзя подделать обе сразу).
  - consistency = доля walk-forward окон с прибылью (устойчивость во времени).
  - Низкая просадка обязательна (профиль: минимальный риск).
  - Карантин символов и продвижение по validation с финальной live-проверкой.
"""
import json
import math

from . import db
from . import execution_core as core
from . import genome as gn
from . import metrics as mt


def _number(value, default):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return default
    return number if math.isfinite(number) else default


def _agent_summary(a: dict) -> dict:
    g = json.loads(a["genome"])
    return {
        "id": a["id"], "type": g["type"], "symbol": a["symbol"],
        "train_sharpe": a["train_sharpe"], "test_sharpe": a["test_sharpe"],
        "test_return": a["test_return"], "test_maxdd": a["test_maxdd"],
        "test_trades": a["test_trades"], "consistency": a["consistency"],
    }


def _decide(agents, cfg, quarantined, sr0=0.0):
    """Детерминированные правила выживания/допуска. Никакого LLM.
    sr0 — планка Deflated Sharpe: Sharpe ниже неё считаем возможной случайностью."""
    sup = cfg["supervisor"]
    decisions = []
    for a in agents:
        g = json.loads(a["genome"])
        valid, invalid_reason = gn.validate_genome(g)
        family = (g.get("type", "?"), a["symbol"], a["timeframe"])
        family_sr0 = sr0.get(family, 0.0) if isinstance(sr0, dict) else sr0
        sharpe_bar = max(sup["promote_min_sharpe"], family_sr0)
        s = _agent_summary(a)
        ts = _number(s["test_sharpe"], -99)
        dd = _number(s["test_maxdd"], 1.0)
        cons = s["consistency"] or 0.0
        trades = s["test_trades"] or 0
        alpha = a["test_alpha"] if a["test_alpha"] is not None else -99
        calmar = a["test_calmar"] if a["test_calmar"] is not None else -99
        pf = a["test_pf"] if a["test_pf"] is not None else 0

        # СМЕРТЬ агентов происходит ВНУТРИ эволюции (отбор по приспособленности
        # в evolution.py). Здесь супервизор НЕ убивает — только ДОПУСКАЕТ к живой
        # торговле тех, кто стабильно показал преимущество. Так популяция
        # накапливается и улучшается между циклами.

        # --- ДОПУСК к живой торговле ---
        # База качества (общая для обоих путей): достаточно сделок, низкая просадка,
        # устойчивый обгон рынка по окнам, символ не в карантине.
        base_ok = (valid
                   and trades >= sup["promote_min_trades"]
                   and cons >= sup["promote_min_consistency"]
                   and dd <= sup.get("promote_max_drawdown", 1.0)
                   and (s["test_return"] or -99) > 0
                   and ts > 0
                   and pf >= sup.get("promote_min_pf", 1.1)
                   and a["symbol"] not in quarantined)
        # Sharpe and DSR are mandatory; diagnostic alpha/Calmar cannot bypass them.
        stats = a.get("return_stats")
        if isinstance(stats, str):
            try:
                stats = json.loads(stats)
            except ValueError:
                stats = None
        reference = max(0.0, family_sr0) / math.sqrt(mt.PERIODS_PER_YEAR.get(a["timeframe"], 8760))
        probability = mt.deflated_sharpe_probability(stats, reference)
        edge_ok = ts >= sharpe_bar
        if sup.get("deflated_sharpe_enabled", True):
            edge_ok = edge_ok and probability >= sup.get("min_dsr_probability", .95)
        if sup.get("require_current_model", False):
            edge_ok = edge_ok and a.get("model_version") == core.MODEL_VERSION
        if sup.get("require_cost_stress", False):
            edge_ok = (edge_ok and _number(a.get("stress_return"), -1) > 0
                       and _number(a.get("stress_pf"), 0) >= sup.get("stress_min_pf", 1.05))
        if base_ok and edge_ok:
            decisions.append((a["id"], "promote",
                (f"Validation Sharpe {ts:.2f}, Calmar {calmar:.2f}, PF {pf:.2f}, "
                 f"alpha {alpha:+.1%}, просадка {dd:.0%}, сделок {trades}, "
                 f"устойчивость {cons:.0%}, DSR {probability:.1%} — допущен в демо")))
        else:
            invalid = f", невалидный геном: {invalid_reason}" if not valid else ""
            decisions.append((a["id"], "hold",
                (f"не дотягивает (Sharpe {ts:.2f}, Calmar {calmar:.2f}, "
                 f"alpha {alpha:+.1%}, PF {pf:.2f}, просадка {dd:.0%}, "
                 f"trades {trades}, устойчивость {cons:.0%}, DSR {probability:.1%}, "
                 f"stress={a.get('stress_return')}{invalid})")))
    return decisions


def _pool_hygiene(conn, cfg, quarantined, sr0):
    """ГИГИЕНА ПУЛА, шаг 2 — демоция. Без неё допущенный агент живёт вечно:
    пул рос ~18/день (за квартал был бы ~1600, 95% в одном символе).

    Правила (консервативно — пул может только СУЖАТЬСЯ, риск не растёт):
      1. Агент, чьи ОСВЕЖЁННЫЕ метрики (reevaluate_promoted) больше не проходят
         те же ворота допуска, — демоция.
      2. Капы пула: не больше pool_max_per_symbol на символ и pool_max_total
         всего (лучшие по alpha, как и при допуске).
      3. Агентов С ОТКРЫТОЙ ПОЗИЦИЕЙ не трогаем НИКОГДА (иначе позиция
         осиротеет — её некому будет вести к выходу).
    Демоция = статус killed + решение 'demote' в журнале (память об испытании
    уже в agent_stats)."""
    sup = cfg["supervisor"]
    promoted = db.get_agents(conn, "promoted")
    if not promoted:
        return 0
    in_pos = {r["agent_id"] for r in
              conn.execute("SELECT agent_id FROM live_positions").fetchall()}
    demoted = 0

    # 1. Переоценка теми же воротами допуска (на освежённых метриках).
    gate = {aid: (act, why)
            for aid, act, why in _decide(promoted, cfg, quarantined, sr0)}
    survivors = []
    for a in promoted:
        act, why = gate.get(a["id"], ("hold", ""))
        if act != "promote" and a["id"] not in in_pos:
            db.set_agent_status(conn, a["id"], "killed")
            db.log_decision(conn, a["id"], "demote", "evolution",
                            f"демоция: свежая переоценка не проходит ворота ({why})")
            demoted += 1
        else:
            survivors.append(a)

    # 2. Капы пула (сначала агенты в позиции, потом лучшие по alpha).
    # Кап на КОМБО (тип×символ×ТФ) закрывает утечку клонов: анти-клон фильтр
    # сравнивает кандидатов внутри одного прогона, а допущенные в РАЗНЫХ
    # прогонах между собой не сравниваются — одинаковые геномы копились в пуле
    # (наблюдалось: 6 идентичных donchian_trend/SOL/8h из 16 мест).
    max_per_sym = sup.get("pool_max_per_symbol", 8)
    max_total = sup.get("pool_max_total", 30)
    max_per_combo = sup.get("pool_max_per_combo", 2)
    survivors.sort(key=lambda a: (a["id"] not in in_pos,
                                  -(a["test_alpha"] if a["test_alpha"] is not None else -99)))
    per_sym, per_combo, taken = {}, {}, 0
    for a in survivors:
        sym = a["symbol"]
        try:
            stype = json.loads(a["genome"]).get("type", "?")
        except (ValueError, TypeError):
            stype = "?"
        combo = (stype, sym, a["timeframe"])
        over = (taken >= max_total or per_sym.get(sym, 0) >= max_per_sym
                or per_combo.get(combo, 0) >= max_per_combo)
        if over and a["id"] not in in_pos:
            db.set_agent_status(conn, a["id"], "killed")
            db.log_decision(conn, a["id"], "demote", "evolution",
                            f"демоция: кап пула (комбо {stype}/{sym}/{a['timeframe']} "
                            f"{per_combo.get(combo, 0)}/{max_per_combo}, символ "
                            f"{per_sym.get(sym, 0)}/{max_per_sym}, всего {taken}/{max_total})")
            demoted += 1
        else:
            per_sym[sym] = per_sym.get(sym, 0) + 1
            per_combo[combo] = per_combo.get(combo, 0) + 1
            taken += 1
    if demoted:
        print(f"  Гигиена пула: демоция {demoted}, осталось допущенных {taken}")
    return demoted


def supervise(conn, cfg):
    """Стадия отбора эволюции: продвигает достойных, убивает слабых."""
    quarantined = db.quarantined_symbols(conn)
    agents = db.get_agents(conn, "candidate")

    # Deflated Sharpe: планка "лучшего по удаче" из РАСПРЕДЕЛЕНИЯ Sharpe всех испытаний
    # (всех когда-либо оценённых агентов), чтобы не поверить везунчику среди тысяч.
    sr0 = {}
    if cfg["supervisor"].get("deflated_sharpe_enabled", True):
        # Коррелированные поиски разных типов/рынков нельзя считать миллионами
        # независимых ставок. Считаем SR0 отдельно в каждой семье.
        family_stats = db.trial_family_stats(conn)
        total_trials = sum(n for n, _ in family_stats.values())
        sr0 = {family: mt.expected_max_sharpe_from_stats(max(n, total_trials), sigma)
               for family, (n, sigma) in family_stats.items()}
        bars = list(sr0.values())
        print(f"  Deflated Sharpe: семейных планок {len(bars)}, "
              f"диапазон SR0 {min(bars, default=0):.2f}…{max(bars, default=0):.2f}")

    print(f"\nОтбор эволюции оценивает {len(agents)} агентов (правила, без LLM)...")
    decisions = _decide(agents, cfg, quarantined, sr0)

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

    counts["demote"] = _pool_hygiene(conn, cfg, quarantined, sr0)

    print(f"\nИтог: продвинуто {counts['promote']}, "
          f"убито {counts['kill']}, оставлено {counts['hold']}, "
          f"демоция из пула {counts['demote']}")
    if counts["promote"] == 0:
        print("Ни один агент не прошёл в paper-live. Это НОРМАЛЬНО — "
              "большинство стратегий не имеют преимущества (урок из треда).")
    return counts
