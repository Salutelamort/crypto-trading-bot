"""
Макро-фильтр на основе потоков в спотовые крипто-ETF (farside.co.uk).

Зачем это нужно (логика из треда):
  - HelloEarthSpaceWorld / Droppingdubs: "используй ИИ и внешние данные для
    создания ОГРАЖДЕНИЙ (guardrails), а не для поиска преимущества".
  - veritas7411: собирает макро-новости как контекст режима рынка.
  farside.co.uk публикует ежедневные чистые потоки денег в спотовые BTC/ETH ETF
  США. Это сильный прокси risk-on / risk-off для всего крипторынка:
    приток денег в ETF  → институционалы покупают  → risk-on
    отток денег из ETF  → институционалы продают    → risk-off

Как используется: НЕ для генерации сделок (математику и сигналы считает
детерминированный Python). Это макро-СТРАЖ — при сильном оттоке из ETF
блокирует открытие НОВЫХ позиций в бумажной торговле. Выходы из позиций
он не трогает (риск-менеджмент всегда работает).

Деградирует мягко: если farside недоступен/заблокирован — возвращает "neutral"
и ничего не блокирует.
"""
import io
import requests
import pandas as pd

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

_URLS = {"BTC": "https://farside.co.uk/btc/", "ETH": "https://farside.co.uk/eth/"}


def _parse_total_flows(html: str) -> pd.Series:
    """Достаёт ежедневные значения чистого потока (колонка Total, млн $)."""
    tables = pd.read_html(io.StringIO(html))
    for t in tables:
        cols = ["".join(str(c) for c in col) if isinstance(col, tuple) else str(col)
                for col in t.columns]
        total_idx = next((i for i, c in enumerate(cols) if "Total" in c), None)
        date_idx = 0
        if total_idx is None:
            continue
        flows = {}
        for _, row in t.iterrows():
            raw_date = str(row.iloc[date_idx]).strip()
            dt = pd.to_datetime(raw_date, errors="coerce", dayfirst=True)
            if pd.isna(dt):          # пропускаем строки Total/Average/Min/Max
                continue
            val = str(row.iloc[total_idx]).replace(",", "").replace("(", "-").replace(")", "")
            try:
                flows[dt] = float(val)
            except ValueError:
                continue
        if len(flows) >= 3:
            return pd.Series(flows).sort_index()
    return pd.Series(dtype=float)


def etf_flow_bias(asset: str = "BTC", lookback_days: int = 5,
                  threshold_musd: float = 0.0) -> dict:
    """
    Возвращает {'bias': risk_on|risk_off|neutral, 'net_flow': сумма за N дней, ...}.
    threshold_musd — порог (млн $) для срабатывания. 0 = по знаку суммы.
    """
    url = _URLS.get(asset.upper())
    if not url:
        return {"bias": "neutral", "net_flow": 0.0, "note": f"нет данных для {asset}"}
    try:
        r = requests.get(url, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        flows = _parse_total_flows(r.text)
    except Exception as e:  # noqa
        return {"bias": "neutral", "net_flow": 0.0,
                "note": f"farside недоступен ({type(e).__name__}) — фильтр выключен"}

    if flows.empty:
        return {"bias": "neutral", "net_flow": 0.0, "note": "не удалось распарсить таблицу"}

    recent = flows.tail(lookback_days)
    net = float(recent.sum())
    if net > threshold_musd:
        bias = "risk_on"
    elif net < -abs(threshold_musd):
        bias = "risk_off"
    else:
        bias = "neutral"
    return {"bias": bias, "net_flow": round(net, 1),
            "days": int(len(recent)), "last_date": str(flows.index[-1].date()),
            "note": f"{asset} ETF: чистый поток {net:+.0f}M$ за {len(recent)}д"}


if __name__ == "__main__":
    for a in ("BTC", "ETH"):
        print(a, etf_flow_bias(a))
