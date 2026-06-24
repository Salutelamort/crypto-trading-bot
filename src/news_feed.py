"""
Новостной фон — СТРАЖ, а не генератор сделок.

Принцип из треда r/algotrading: новости и сентимент полезны как ОГРАЖДЕНИЕ
(блокировать входы в плохой момент), но НЕ как источник альфы. LLM к этому не
подпускаем — он галлюцинирует логику, которая «звучит правильно». Поэтому здесь
всё детерминированно: числовой индекс сентимента + поиск опасных слов в заголовках.

Бесплатно, без API-ключей:
  - Индекс страха и жадности крипторынка (alternative.me) — число 0..100.
  - Заголовки CoinDesk + Cointelegraph (RSS) — ищем явные негативные катализаторы.

Используется как ещё один фильтр входа в live_trade / paper_trade:
  - сильный негативный катализатор (взлом/бан/иск/крах) → блок новых входов;
  - "Extreme Greed" (рынок перегрет) → блок новых входов (профиль = низкий риск).
"""
import re
import requests

_H = {"User-Agent": "Mozilla/5.0 (compatible; trading-bot/1.0)"}

_RSS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
]

# Слова СИЛЬНЫХ системных негативных катализаторов (англ. заголовки).
# Намеренно узкий список — крипто-новости всегда слегка негативны, и широкий
# набор слов блокировал бы бота постоянно. Здесь только тяжёлые события.
_NEGATIVE = [
    "hacked", "is hacked", "exploited", "stolen", "drained", "depeg",
    "collapse", "bankrupt", "insolven", "halts withdrawal", "halt withdrawal",
    "rug pull", "ponzi", "market crash", "liquidation cascade",
    "massive sell", "plummet",
]


def fear_greed():
    """Индекс страха и жадности крипторынка: {value:0..100, label:str}."""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=12)
        d = r.json()["data"][0]
        return {"value": int(d["value"]), "label": d["value_classification"]}
    except Exception:  # noqa
        return {"value": None, "label": "n/a"}


def _headlines():
    titles = []
    for url in _RSS:
        try:
            r = requests.get(url, timeout=12, headers=_H)
            for t in re.findall(r"<title>(.*?)</title>", r.text, re.S):
                t = re.sub(r"<!\[CDATA\[|\]\]>", "", t).strip()
                if t and "rss" not in t.lower():
                    titles.append(t)
        except Exception:  # noqa
            continue
    return titles


def news_risk():
    """Ищет негативные катализаторы в свежих заголовках.
    Возвращает {hits:int, matched:[...]} — заголовки с опасными словами."""
    matched = []
    for t in _headlines():
        low = t.lower()
        if any(w in low for w in _NEGATIVE):
            matched.append(t)
    return {"hits": len(matched), "matched": matched[:6]}


def news_gate(cfg):
    """
    Главная точка: блокировать ли НОВЫЕ входы из-за новостного фона.
    Выходы из позиций этот страж не трогает (риск-менеджмент всегда работает).
    """
    nc = cfg.get("news", {})
    fng = fear_greed()
    risk = news_risk()

    reasons = []
    block = False

    # 1. сильный негативный катализатор
    if risk["hits"] >= nc.get("min_negative_headlines", 2):
        block = True
        reasons.append(f"негативные новости ({risk['hits']} заголовков)")

    # 2. перегрев рынка (экстремальная жадность) — для низкорискового профиля
    greed_limit = nc.get("block_above_greed", 80)
    if fng["value"] is not None and fng["value"] >= greed_limit:
        block = True
        reasons.append(f"перегрев рынка (F&G {fng['value']} {fng['label']})")

    return {
        "block": block,
        "fng": fng,
        "news_hits": risk["hits"],
        "matched": risk["matched"],
        "reason": "; ".join(reasons) if reasons else
                  f"фон спокойный (F&G {fng['value']} {fng['label']}, негатива нет)",
    }


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    print(news_gate(cfg))
