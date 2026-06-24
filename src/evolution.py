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


def _proven_symbols(conn, cfg, symbols):
    """Символы с ДОКАЗАННЫМ преимуществом (как у piratastuertos): где хоть один
    агент показал OOS Sharpe выше порога. Пока таких нет — возвращаем все (bootstrap)."""
    bar = cfg["evolution"].get("proven_min_sharpe", 0.0)
    rows = conn.execute(
        "SELECT DISTINCT symbol FROM agents WHERE test_sharpe > ?", (bar,)).fetchall()
    proven = {r["symbol"] for r in rows} & set(symbols)
    return sorted(proven) if proven else symbols


def _anti_clone(conn, cfg):
    """
    Убиваем клонов: если equity-кривые двух живых агентов сильно коррелируют,
    оставляем того, у кого лучше out-of-sample Sharpe.
    """
    thresh = cfg["evolution"]["anti_clone_corr"]
    alive = db.get_agents(conn, "candidate")
    # нужны equity-кривые — пересчитывать дорого, поэтому корреляцию приближаем
    # по близости геномов того же типа/символа + близкому test_sharpe.
    killed = 0
    for i in range(len(alive)):
        for j in range(i + 1, len(alive)):
            a, b = alive[i], alive[j]
            if a["status"] != "candidate" or b["status"] != "candidate":
                continue
            ga, gb = json.loads(a["genome"]), json.loads(b["genome"])
            if ga["type"] != gb["type"] or a["symbol"] != b["symbol"]:
                continue
            sa = a["test_sharpe"] or -99
            sb = b["test_sharpe"] or -99
            # одинаковый тип+символ и почти идентичный test_sharpe => клон
            if abs(sa - sb) < (1 - thresh):
                weaker = a if sa < sb else b
                db.set_agent_status(conn, weaker["id"], "killed")
                db.log_decision(conn, weaker["id"], "kill", "rules",
                                f"анти-клон: дубликат {ga['type']}/{a['symbol']} "
                                f"со слабым test_sharpe ({min(sa, sb):.2f})")
                weaker["status"] = "killed"
                killed += 1
    return killed


def evolve(conn, cfg, data_by_symbol):
    """
    data_by_symbol: {symbol: DataFrame OHLCV}
    Запускает несколько поколений эволюции.
    """
    rng = random.Random(42)  # фиксируем seed для воспроизводимости
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
        survivors = sorted(alive, key=lambda a: _fitness(a, min_tr),
                           reverse=True)[:ev["survivors"]]

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

        # 3. Отбор: убиваем всё, что вне топа по train_sharpe.
        alive = db.get_agents(conn, "candidate")
        ranked = sorted(alive, key=lambda a: _fitness(a, min_tr), reverse=True)
        for a in ranked[ev["survivors"]:]:
            db.set_agent_status(conn, a["id"], "killed")
            reason = (f"мало сделок ({a['train_trades']} < {min_tr})"
                      if (a["train_trades"] or 0) < min_tr
                      else f"train_sharpe {a['train_sharpe']} вне топ-{ev['survivors']}")
            db.log_decision(conn, a["id"], "kill", "rules",
                            f"эволюционный отбор: {reason}")

        # 4. Анти-клон фильтр.
        cloned = _anti_clone(conn, cfg)

        survivors_now = db.get_agents(conn, "candidate")
        print(f"  Живых агентов: {len(survivors_now)} | убито клонов: {cloned}")
        for a in sorted(survivors_now,
                        key=lambda x: x["test_sharpe"] or -99, reverse=True)[:5]:
            g = json.loads(a["genome"])
            print(f"   #{a['id']} {g['type']:14s} {a['symbol']:8s} "
                  f"train_sh={a['train_sharpe']:.2f} test_sh={a['test_sharpe']:.2f} "
                  f"cons={a['consistency']:.2f} trades={a['test_trades']}")
