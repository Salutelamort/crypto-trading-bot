"""
СУПЕРВИЗОР — это и есть архитектура с ФОТО (комментарий piratastuertos):

  "В моей системе Клод не выбирает стратегии. Он решает, стоит ли УБИТЬ агента,
   который работает плохо, стоит ли ПРОДВИНУТЬ одного к реальной торговле и стоит
   ли ГЕНЕРИРОВАТЬ нового кандидата. Это решения по УПРАВЛЕНИЮ, а не по стратегии.
   Инфраструктура — Python. Уровень управления — Клод. Ни один не делает работу
   другого."

Поэтому супервизор работает ТОЛЬКО с агрегированными метриками агентов и выдаёт
управленческие действия: kill / promote / generate / hold. Он НИКОГДА не считает
индикаторы, не выбирает сделки и не трогает математику (её галлюцинирует LLM).

Два бэкенда (выбор в config.yaml -> supervisor.backend):
  * "rules"  — детерминированные правила. Работает всегда, $0, без интернета.
               (как у автора: "Общая стоимость LLM за 60 дней: $0".)
  * "claude" — те же решения принимает Claude (нужен пакет anthropic + ключ).
               Claude получает ТОЛЬКО таблицу метрик и возвращает действия.

Ключевые правила взяты из фото-комментария:
  - Разделяем метрики для ПРОДВИЖЕНИЯ и для УБИЙСТВА (нельзя подделать обе сразу).
  - Метрика consistency ловит переобучение (train winrate vs test winrate).
  - Карантин символов и продвижение только по out-of-sample результатам.
"""
import json
import os
from . import db


def _agent_summary(a: dict) -> dict:
    """Компактная сводка метрик агента для принятия решения."""
    g = json.loads(a["genome"])
    return {
        "id": a["id"], "type": g["type"], "symbol": a["symbol"],
        "train_sharpe": a["train_sharpe"], "test_sharpe": a["test_sharpe"],
        "test_return": a["test_return"], "test_maxdd": a["test_maxdd"],
        "test_trades": a["test_trades"], "consistency": a["consistency"],
    }


# ---------------------------------------------------------------------------
#  Бэкенд 1: детерминированные правила (по умолчанию)
# ---------------------------------------------------------------------------
def _decide_rules(agents, cfg, quarantined):
    sup = cfg["supervisor"]
    decisions = []
    for a in agents:
        s = _agent_summary(a)
        ts = s["test_sharpe"] or -99
        dd = s["test_maxdd"] or 1.0
        cons = s["consistency"] or 0.0
        trades = s["test_trades"] or 0

        # --- решение об УБИЙСТВЕ (своя группа метрик) ---
        if dd > sup["kill_max_drawdown"]:
            decisions.append((a["id"], "kill",
                f"просадка {dd:.0%} > лимита {sup['kill_max_drawdown']:.0%}"))
            continue
        if cons < sup["kill_min_consistency"]:
            decisions.append((a["id"], "kill",
                f"переобучение: consistency {cons:.2f} < {sup['kill_min_consistency']}"))
            continue

        # --- решение о ПРОДВИЖЕНИИ (другая группа метрик) ---
        max_dd_ok = dd <= sup.get("promote_max_drawdown", 1.0)
        ok = (ts >= sup["promote_min_sharpe"]
              and trades >= sup["promote_min_trades"]
              and cons >= sup["promote_min_consistency"]
              and max_dd_ok
              and a["symbol"] not in quarantined)
        if ok:
            decisions.append((a["id"], "promote",
                f"out-of-sample Sharpe {ts:.2f}, просадка {dd:.0%}, сделок {trades}, "
                f"consistency {cons:.2f} — стабилен и низкорисков"))
        else:
            decisions.append((a["id"], "hold",
                f"не дотягивает (test_sharpe {ts:.2f}, просадка {dd:.0%}, "
                f"trades {trades}, cons {cons:.2f})"))
    return decisions


# ---------------------------------------------------------------------------
#  Бэкенд 2: Claude принимает те же управленческие решения
# ---------------------------------------------------------------------------
def _decide_claude(agents, cfg, quarantined):
    try:
        import anthropic
    except ImportError:
        print("  [!] Пакет anthropic не установлен — откат на правила.")
        return _decide_rules(agents, cfg, quarantined)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  [!] Нет ANTHROPIC_API_KEY — откат на правила.")
        return _decide_rules(agents, cfg, quarantined)

    sup = cfg["supervisor"]
    summaries = [_agent_summary(a) for a in agents]
    prompt = f"""Ты — супервизор управления портфелем торговых агентов.
Ты НЕ выбираешь стратегии и НЕ считаешь индикаторы. Ты принимаешь только
управленческие решения по каждому агенту: kill / promote / hold.

Профиль пользователя: МИНИМАЛЬНЫЙ РИСК + СТАБИЛЬНОСТЬ важнее доходности.
Пороги (из конфигурации):
- ПРОДВИГАТЬ (promote), только если ВСЕ: test_sharpe >= {sup['promote_min_sharpe']},
  test_trades >= {sup['promote_min_trades']}, consistency >= {sup['promote_min_consistency']},
  test_maxdd <= {sup.get('promote_max_drawdown', 1.0)} (низкая просадка обязательна!),
  и символа нет в карантине.
- УБИВАТЬ (kill), если: test_maxdd > {sup['kill_max_drawdown']}
  ИЛИ consistency < {sup['kill_min_consistency']} (признак переобучения).
- Иначе hold.

Символы в карантине: {sorted(quarantined)}

Агенты (метрики; train = in-sample, test = out-of-sample):
{json.dumps(summaries, ensure_ascii=False, indent=2)}

Верни СТРОГО JSON-массив объектов вида:
[{{"id": <int>, "action": "kill|promote|hold", "rationale": "<кратко почему>"}}]
Только JSON, без пояснений."""

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=sup["model"], max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    try:
        arr = json.loads(text)
        return [(d["id"], d["action"], d.get("rationale", "")) for d in arr]
    except Exception as e:  # noqa
        print(f"  [!] Не удалось разобрать ответ Claude ({e}) — откат на правила.")
        return _decide_rules(agents, cfg, quarantined)


# ---------------------------------------------------------------------------
#  Точка входа
# ---------------------------------------------------------------------------
def supervise(conn, cfg):
    """Прогоняет всех живых кандидатов через управленческий слой."""
    backend = cfg["supervisor"]["backend"]
    quarantined = db.quarantined_symbols(conn)
    agents = db.get_agents(conn, "candidate")
    if not agents:
        print("Нет агентов-кандидатов. Сначала запусти эволюцию.")
        return

    print(f"\nСупервизор ({backend}) оценивает {len(agents)} агентов...")
    if backend == "claude":
        decisions = _decide_claude(agents, cfg, quarantined)
    else:
        decisions = _decide_rules(agents, cfg, quarantined)

    counts = {"kill": 0, "promote": 0, "hold": 0}
    for agent_id, action, rationale in decisions:
        db.log_decision(conn, agent_id, action, backend, rationale)
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
