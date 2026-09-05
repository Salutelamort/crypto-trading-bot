"""
Эволюционный отбор агентов. Прямая реализация архитектуры с фото
(piratastuertos): "эволюционная система. Агенты генерируются, оцениваются и
автоматически уничтожаются. Сгенерировано 120000+ стратегий, ~20 живы в любой
момент. Эволюционный отбор жесток и эффективен."

Что здесь реализовано из его уроков:
- Генерация → бэктест на train → оценка → отбор лучших → убийство слабых.
- Анти-клон фильтр: корреляция equity > порога → убить более слабого
  ("у 3 лучших агентов был идентичный Sharpe — никакой реальной диверсификации").
- Карантин символов: символы с отрицательной PnL блокируются от генерации.
- Метрики train и validation считаются ОТДЕЛЬНО. Validation используется
  многократно и поэтому не называется финальным невидимым тестом.

ВАЖНО: эволюция НЕ продвигает агентов в реальную торговлю. Это делает
супервизор (supervisor.py) — отдельное управленческое решение.
"""
import copy
import json
import random

import pandas as pd

from . import backtest as bt
from . import db
from . import genome as gn


def _fitness(agent, min_trades):
    """Фитнес для отбора = train_sharpe (in-sample, без утечки), НО непригодны:
    - мало сделок в обучении (одна удачная не должна давать высокий Sharpe);
    - НОЛЬ сделок в свежих данных (OOS) — стратегия мертва на актуальном рынке,
      такие не должны выживать и размножаться (это и есть 'бесполезные' агенты)."""
    try:
        valid, _ = gn.validate_genome(json.loads(agent["genome"]))
    except (ValueError, TypeError):
        valid = False
    if not valid:
        return -999.0
    if (agent["train_trades"] or 0) < min_trades:
        return -999.0
    if (agent["test_trades"] or 0) == 0:
        return -999.0
    return agent["train_sharpe"] if agent["train_sharpe"] is not None else -99.0


def _evaluate(genome, df, cfg):
    """Walk-forward оценка агента. consistency = доля прибыльных OOS окон."""
    train, test, consistency = bt.walk_forward_eval(genome, df, cfg)
    factors = cfg.get("validation", {}).get("cost_stress_multipliers", [])
    if factors and test["total_return"] > 0 and test["num_trades"] >= cfg["supervisor"].get("promote_min_trades", 20):
        start = int(len(df) * cfg["train_ratio"]) + cfg.get("validation", {}).get("embargo_bars", 0)
        signal = gn.signal(genome, df, cfg["risk"].get("allow_short", False)).shift(
            cfg.get("execution", {}).get("signal_delay_bars", 1)).fillna(0)
        stressed_results = []
        for factor in factors:
            stressed = copy.deepcopy(cfg)
            stressed["costs"] = {key: value * factor for key, value in cfg["costs"].items()}
            stressed_results.append(bt.run(genome, df.iloc[start:], stressed, sig=signal.iloc[start:]))
        test["stress_return"] = min(item["total_return"] for item in stressed_results)
        test["stress_pf"] = min(item["profit_factor"] for item in stressed_results)
    return train, test, consistency


def reevaluate_promoted(conn, cfg, data_by_key):
    """ГИГИЕНА ПУЛА, шаг 1: освежить OOS-метрики допущенных агентов на свежих
    данных (walk-forward, та же оценка, что при рождении). Без этого метрики
    promoted замирают на момент допуска, и протухший агент живёт вечно.
    count_trial=False: переоценка — не новое испытание для Deflated Sharpe.
    Лимит на прогон защищает бюджет времени облака."""
    limit = int(cfg["supervisor"].get("pool_reeval_per_run", 60))
    promoted = db.get_agents(conn, "promoted")[:limit]
    done = 0
    for a in promoted:
        g = json.loads(a["genome"])
        key = (a["symbol"], a["timeframe"])
        df = data_by_key.get(key)
        if df is None or len(df) < 400:
            continue
        train_m, test_m, cons = _evaluate(g, df, cfg)
        db.update_agent_metrics(conn, a["id"], train_m, test_m, cons,
                                count_trial=False)
        done += 1
    if done:
        print(f"Переоценка promoted на свежих данных: {done} агентов")
    return done


def _select_survivors(ranked, n, max_per_sym):
    """Выбирает n выживших с КВОТОЙ на символ, чтобы пул не схлопывался в одну
    монету (диверсификация генофонда). Если разнообразия не хватает — добивает
    лучшими из оставшихся."""
    kept, per = [], {}
    for a in ranked:
        if len(kept) >= n:
            break
        s = a["symbol"]
        if per.get(s, 0) >= max_per_sym:
            continue
        kept.append(a)
        per[s] = per.get(s, 0) + 1
    if len(kept) < n:
        ids = {a["id"] for a in kept}
        for a in ranked:
            if len(kept) >= n:
                break
            if a["id"] not in ids:
                kept.append(a)
    return kept


def _proven_symbols(conn, cfg, symbols):
    """Символы с ДОКАЗАННЫМ преимуществом (как у piratastuertos): где хоть один
    агент показал OOS Sharpe выше порога. Пока таких нет — возвращаем все (bootstrap)."""
    bar = cfg["evolution"].get("proven_min_sharpe", 0.0)
    # из компактной памяти (agent_stats), а не из сырых agents — переживает прунинг
    proven = db.proven_symbols_from_stats(conn, bar) & set(symbols)
    return sorted(proven) if proven else symbols


def _oos_returns(genome, df, cfg):
    """Доходности агента на validation-участке (для матрицы корреляций)."""
    cut = int(len(df) * cfg["train_ratio"]) + cfg.get("validation", {}).get("embargo_bars", 0)
    oos = df.iloc[cut:]
    delay = cfg.get("execution", {}).get("signal_delay_bars", 1)
    allow_short = cfg["risk"].get("allow_short", False)
    sig = gn.signal(genome, df, allow_short).shift(delay).fillna(0).astype(int).iloc[cut:]
    m = bt.run(genome, oos, cfg, sig=sig)
    return m["returns"]


def common_daily_returns(returns):
    """Compound onto complete UTC days; do not invent zero returns for missing days."""
    if returns.empty:
        return returns
    days = (1 + returns).resample("1D").prod(min_count=1) - 1
    # Boundary days may be partial and distort cross-timeframe comparisons.
    return days.iloc[1:-1].dropna()


def _anti_clone(conn, cfg, data_by_key):
    """
    ДИВЕРСИФИКАЦИЯ. Убиваем клонов по РЕАЛЬНОЙ корреляции кривых дохода
    (а не по близости Sharpe, как раньше — то пропускало однотипных).

    Принцип Тома Бассо («Всепогодный трейдер»): портфель должен пережить любой
    режим → стратегии должны быть НЕПОХОЖИ. Если два агента зарабатывают/теряют
    в одни и те же моменты (corr > порога) — оставляем более сильного (по alpha,
    затем Sharpe), второго в расход. Это касается и РАЗНЫХ символов/типов:
    важна именно совместная динамика дохода, а не формальное совпадение генома.
    """
    thresh = cfg["evolution"]["anti_clone_corr"]
    alive = db.get_agents(conn, "candidate")
    info = {}   # id -> (timeframe, returns series)
    for a in alive:
        key = (a["symbol"], a["timeframe"])
        if key not in data_by_key:
            continue
        try:
            info[a["id"]] = common_daily_returns(
                _oos_returns(json.loads(a["genome"]), data_by_key[key], cfg))
        except Exception:  # noqa
            continue
    if len(info) < 2:
        return 0

    score = {a["id"]: ((a["test_alpha"] if a["test_alpha"] is not None else -99),
                       (a["test_sharpe"] if a["test_sharpe"] is not None else -99))
             for a in alive}
    # от сильнейших к слабым: сильный занимает «нишу», похожие на него — убиваются.
    # Корреляцию считаем ТОЛЬКО внутри одного таймфрейма (у разных ТФ разная сетка
    # баров — это уже диверсификация по построению).
    order = sorted(info.keys(), key=lambda i: score[i], reverse=True)
    kept, killed = [], 0
    for i in order:
        r_i = info[i]
        clone = False
        for j in kept:
            paired = pd.concat([r_i, info[j]], axis=1, join="inner").dropna()
            if len(paired) < cfg["evolution"].get("correlation_min_days", 60):
                continue
            c = paired.iloc[:, 0].corr(paired.iloc[:, 1])
            if pd.notna(c) and c > thresh:
                clone = True
                break
        if clone:
            db.set_agent_status(conn, i, "killed")
            db.log_decision(conn, i, "kill", "rules",
                            f"анти-клон: корреляция дохода > {thresh} с более сильным агентом "
                            f"(нет диверсификации)")
            killed += 1
        else:
            kept.append(i)
    return killed


def evolve(conn, cfg, data_by_key):
    """
    data_by_key: {(symbol, timeframe): DataFrame OHLCV}
    МУЛЬТИТАЙМФРЕЙМ: таймфрейм — часть генома, эволюция ищет лучший под стратегию.
    Запускает несколько поколений эволюции.
    """
    # РАНЬШЕ здесь был фиксированный seed(42): каждый облачный прогон генерировал
    # ОДНИ И ТЕ ЖЕ случайные геномы → 16k агентов, но реального поиска не было
    # (бег на месте). Теперь seed случайный — пространство стратегий реально
    # исследуется от прогона к прогону. Выжившие накапливаются в БД (эволюция).
    ev = cfg["evolution"]
    rng = random.Random(ev.get("seed"))
    quarantined = db.quarantined_symbols(conn)
    # доступные пары (символ, таймфрейм): есть данные и символ не в карантине
    keys = [(s, tf) for (s, tf) in data_by_key
            if s in cfg["symbols"] and s not in quarantined]
    if not keys:
        print("Нет доступных пар символ/таймфрейм (карантин или нет данных).")
        return

    for gen in range(ev["generations"]):
        print(f"\n=== Поколение {gen + 1}/{ev['generations']} ===")

        # 1. Пополняем популяцию до нужного размера новыми кандидатами.
        alive = db.get_agents(conn, "candidate")
        need = ev["population_size"] - len(alive)
        min_tr = ev.get("min_trades", 0)
        max_per_sym = ev.get("max_survivors_per_symbol", ev["survivors"])
        ranked_alive = sorted(alive, key=lambda a: _fitness(a, min_tr), reverse=True)
        survivors = _select_survivors(ranked_alive, ev["survivors"], max_per_sym)

        new_genomes = []
        # мутации выживших (символ И таймфрейм сохраняются)
        for s in survivors:
            for _ in range(ev["mutations_per_survivor"]):
                new_genomes.append(gn.mutate(json.loads(s["genome"]), rng))
        # добиваем случайными по всем парам (символ × таймфрейм)
        while len(new_genomes) < need:
            sym, tf = rng.choice(keys)
            new_genomes.append(gn.random_genome(sym, tf, rng))
        new_genomes = new_genomes[:max(need, 0)]

        # 2. Оцениваем новых кандидатов через walk-forward.
        for g in new_genomes:
            key = (g["symbol"], g["timeframe"])
            if key not in data_by_key:
                continue
            df = data_by_key[key]
            if len(df) < 400:
                continue
            train_m, test_m, cons = _evaluate(g, df, cfg)
            aid = db.insert_agent(conn, g, g["symbol"], g["timeframe"])
            db.update_agent_metrics(conn, aid, train_m, test_m, cons)

        # 3. Отбор: выживают лучшие С КВОТОЙ на символ (диверсификация генофонда).
        alive = db.get_agents(conn, "candidate")
        ranked = sorted(alive, key=lambda a: _fitness(a, min_tr), reverse=True)
        keep_ids = {a["id"] for a in _select_survivors(ranked, ev["survivors"], max_per_sym)}
        for a in ranked:
            if a["id"] in keep_ids:
                continue
            db.set_agent_status(conn, a["id"], "killed")
            reason = (f"мало сделок ({a['train_trades']} < {min_tr})"
                      if (a["train_trades"] or 0) < min_tr
                      else f"train_sharpe {a['train_sharpe']} вне топ-{ev['survivors']} (с квотой на символ)")
            db.log_decision(conn, a["id"], "kill", "rules",
                            f"эволюционный отбор: {reason}")

        # 4. Анти-клон фильтр (по реальной корреляции дохода → диверсификация).
        cloned = _anti_clone(conn, cfg, data_by_key)

        survivors_now = db.get_agents(conn, "candidate")
        print(f"  Живых агентов: {len(survivors_now)} | убито клонов: {cloned}")
        for a in sorted(survivors_now,
                        key=lambda x: x["test_sharpe"] or -99, reverse=True)[:5]:
            g = json.loads(a["genome"])
            print(f"   #{a['id']} {g['type']:14s} {a['symbol']:8s} {a['timeframe']:>3s} "
                  f"train_sh={a['train_sharpe']:.2f} test_sh={a['test_sharpe']:.2f} "
                  f"cons={a['consistency']:.2f} trades={a['test_trades']}")
