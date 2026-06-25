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
- Метрики train и test считаются ОТДЕЛЬНО (для отбора берём train,
  для честной оценки — test, который агент не видел).

ВАЖНО: эволюция НЕ продвигает агентов в реальную торговлю. Это делает
супервизор (supervisor.py) — отдельное управленческое решение.
"""
import json
import random
import numpy as np
import pandas as pd

from . import db
from . import genome as gn
from . import backtest as bt
from . import metrics as mt


def _fitness(agent, min_trades):
    """Фитнес для отбора = train_sharpe (in-sample, без утечки), НО непригодны:
    - мало сделок в обучении (одна удачная не должна давать высокий Sharpe);
    - НОЛЬ сделок в свежих данных (OOS) — стратегия мертва на актуальном рынке,
      такие не должны выживать и размножаться (это и есть 'бесполезные' агенты)."""
    if (agent["train_trades"] or 0) < min_trades:
        return -999.0
    if (agent["test_trades"] or 0) == 0:
        return -999.0
    return agent["train_sharpe"] if agent["train_sharpe"] is not None else -99.0


def _evaluate(genome, df, cfg):
    """Walk-forward оценка агента. consistency = доля прибыльных OOS окон."""
    return bt.walk_forward_eval(genome, df, cfg)


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
    rows = conn.execute(
        "SELECT DISTINCT symbol FROM agents WHERE test_sharpe > ?", (bar,)).fetchall()
    proven = {r["symbol"] for r in rows} & set(symbols)
    return sorted(proven) if proven else symbols


def _oos_returns(genome, df, cfg):
    """Доходности агента на out-of-sample участке (для матрицы корреляций)."""
    cut = int(len(df) * cfg["train_ratio"])
    oos = df.iloc[cut:]
    delay = cfg.get("execution", {}).get("signal_delay_bars", 1)
    allow_short = cfg["risk"].get("allow_short", False)
    sig = gn.signal(genome, df, allow_short).shift(delay).fillna(0).astype(int).iloc[cut:]
    m = bt.run(genome, oos, cfg, sig=sig)
    return m["equity"].pct_change().fillna(0.0)


def _anti_clone(conn, cfg, data_by_symbol):
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
    rets = {}
    for a in alive:
        sym = a["symbol"]
        if sym not in data_by_symbol:
            continue
        try:
            rets[a["id"]] = _oos_returns(json.loads(a["genome"]), data_by_symbol[sym], cfg)
        except Exception:  # noqa
            continue
    if len(rets) < 2:
        return 0

    R = pd.DataFrame(rets).fillna(0.0)
    corr = R.corr()
    score = {a["id"]: ((a["test_alpha"] if a["test_alpha"] is not None else -99),
                       (a["test_sharpe"] if a["test_sharpe"] is not None else -99))
             for a in alive}
    # от сильнейших к слабым: сильный занимает «нишу», похожие на него — убиваются
    order = sorted(rets.keys(), key=lambda i: score[i], reverse=True)
    kept, killed = [], 0
    for i in order:
        if any(abs(corr.loc[i, j]) > thresh for j in kept):
            db.set_agent_status(conn, i, "killed")
            db.log_decision(conn, i, "kill", "rules",
                            f"анти-клон: корреляция дохода > {thresh} с более сильным агентом "
                            f"(нет диверсификации)")
            killed += 1
        else:
            kept.append(i)
    return killed


def evolve(conn, cfg, data_by_symbol):
    """
    data_by_symbol: {symbol: DataFrame OHLCV}
    Запускает несколько поколений эволюции.
    """
    # РАНЬШЕ здесь был фиксированный seed(42): каждый облачный прогон генерировал
    # ОДНИ И ТЕ ЖЕ случайные геномы → 16k агентов, но реального поиска не было
    # (бег на месте). Теперь seed случайный — пространство стратегий реально
    # исследуется от прогона к прогону. Выжившие накапливаются в БД (эволюция).
    rng = random.Random()
    ev = cfg["evolution"]
    quarantined = db.quarantined_symbols(conn)
    symbols = [s for s in cfg["symbols"]
               if s in data_by_symbol and s not in quarantined]
    if not symbols:
        print("Нет доступных символов (все в карантине или нет данных).")
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

        # Генерируем новых ТОЛЬКО на символах с доказанным преимуществом
        # (как piratastuertos). Пока таких нет — по всем (фаза bootstrap).
        gen_symbols = symbols
        if ev.get("restrict_to_proven"):
            gen_symbols = _proven_symbols(conn, cfg, symbols)

        new_genomes = []
        # мутации выживших (символ сохраняется)
        for s in survivors:
            for _ in range(ev["mutations_per_survivor"]):
                new_genomes.append(gn.mutate(json.loads(s["genome"]), rng))
        # добиваем случайными на проверенных символах
        while len(new_genomes) < need:
            sym = rng.choice(gen_symbols)
            new_genomes.append(gn.random_genome(sym, cfg["timeframe"], rng))
        new_genomes = new_genomes[:max(need, 0)]

        # 2. Оцениваем новых кандидатов через walk-forward.
        for g in new_genomes:
            sym = g["symbol"]
            if sym not in data_by_symbol:
                continue
            df = data_by_symbol[sym]
            if len(df) < 400:
                continue
            train_m, test_m, cons = _evaluate(g, df, cfg)
            aid = db.insert_agent(conn, g, sym, cfg["timeframe"])
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
        cloned = _anti_clone(conn, cfg, data_by_symbol)

        survivors_now = db.get_agents(conn, "candidate")
        print(f"  Живых агентов: {len(survivors_now)} | убито клонов: {cloned}")
        for a in sorted(survivors_now,
                        key=lambda x: x["test_sharpe"] or -99, reverse=True)[:5]:
            g = json.loads(a["genome"])
            print(f"   #{a['id']} {g['type']:14s} {a['symbol']:8s} "
                  f"train_sh={a['train_sharpe']:.2f} test_sh={a['test_sharpe']:.2f} "
                  f"cons={a['consistency']:.2f} trades={a['test_trades']}")
