"""Public spot entry constraints. Current rules are never applied to old candles."""
import copy
import math
import time
from decimal import ROUND_DOWN, Decimal

import requests

from .data_feed import BINANCE_FALLBACKS

_cache = {}


def _public(path, symbol):
    for base in BINANCE_FALLBACKS:
        try:
            response = requests.get(base + path, params={"symbol": symbol}, timeout=5)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            continue
    raise RuntimeError("exchange_rules_unavailable")


def entry_rules(symbol):
    """Cache metadata for one hour; refresh market notional reference every call."""
    cached = _cache.get(symbol)
    if cached is None or not 0 <= time.monotonic() - cached[0] < 3600:
        payload = _public("/api/v3/exchangeInfo", symbol)
        rows = [row for row in payload["symbols"] if row["symbol"] == symbol]
        if len(rows) != 1:
            raise ValueError("exchange_symbol_unavailable")
        cached = (time.monotonic(), rows[0])
        _cache[symbol] = cached
    rules = copy.deepcopy(cached[1])
    windows = set()
    for item in rules["filters"]:
        if ((item["filterType"] == "MIN_NOTIONAL" and item.get("applyToMarket", False))
                or (item["filterType"] == "NOTIONAL"
                    and (item.get("applyMinToMarket", False) or item.get("applyMaxToMarket", False)))):
            windows.add(int(item["avgPriceMins"]))
    references = {}
    if 0 in windows:
        references[0] = _public("/api/v3/ticker/price", symbol)["price"]
    if windows - {0}:
        average = _public("/api/v3/avgPrice", symbol)
        references[int(average["mins"])] = average["price"]
    if not windows.issubset(references):
        raise ValueError("notional_reference_window_unavailable")
    rules["references"] = references
    rules["received_at"] = time.time()
    return rules


def market_entry_quantity(quantity, rules):
    """Round DOWN to the common lot step; reject a below-minimum entry.

    Average-price checks approximate Binance market-notional validation; exchange
    reference prices and account-dependent filters are not available here.
    """
    if (rules.get("status") != "TRADING" or not rules.get("isSpotTradingAllowed", False)
            or "MARKET" not in rules.get("orderTypes", [])):
        raise ValueError("symbol_not_tradable")
    qty = Decimal(str(quantity))
    if not qty.is_finite() or qty <= 0:
        raise ValueError("invalid_quantity")
    lots = [f for f in rules["filters"] if f["filterType"] in ("LOT_SIZE", "MARKET_LOT_SIZE")]
    if not any(f["filterType"] == "LOT_SIZE" for f in lots):
        raise ValueError("lot_rules_missing")
    steps = []
    for lot in lots:
        values = [Decimal(lot[key]) for key in ("minQty", "maxQty", "stepSize")]
        if any(not value.is_finite() or value < 0 for value in values):
            raise ValueError("invalid_lot_rule")
        _, maximum, step = values
        if maximum > 0:
            qty = min(qty, maximum)
        if step > 0:
            steps.append(step)
    if steps:
        places = max(max(0, -step.as_tuple().exponent) for step in steps)
        scale = 10 ** places
        step = Decimal(math.lcm(*(int(s * scale) for s in steps))) / scale
        qty = (qty / step).to_integral_value(rounding=ROUND_DOWN) * step
    if qty <= 0 or any(qty < Decimal(lot["minQty"]) for lot in lots):
        raise ValueError("below_min_quantity")
    for item in rules["filters"]:
        kind = item["filterType"]
        minimum = kind == "MIN_NOTIONAL" and item.get("applyToMarket", False)
        minimum |= kind == "NOTIONAL" and item.get("applyMinToMarket", False)
        maximum = kind == "NOTIONAL" and item.get("applyMaxToMarket", False)
        if not (minimum or maximum):
            continue
        reference = Decimal(str(rules["references"][int(item["avgPriceMins"])]))
        if not reference.is_finite() or reference <= 0:
            raise ValueError("invalid_notional_reference")
        value = qty * reference
        if minimum and value < Decimal(item["minNotional"]):
            raise ValueError("below_min_notional")
        if maximum and value > Decimal(item["maxNotional"]):
            raise ValueError("above_max_notional")
    return float(qty)


def market_exit_quantity(quantity, rules):
    """Split inventory above a market-order notional maximum into legal orders."""
    qty = Decimal(str(quantity))
    for item in rules["filters"]:
        if item["filterType"] == "NOTIONAL" and item.get("applyMaxToMarket", False):
            reference = Decimal(str(rules["references"][int(item["avgPriceMins"])]))
            maximum = Decimal(item["maxNotional"])
            if not reference.is_finite() or reference <= 0 or not maximum.is_finite() or maximum <= 0:
                raise ValueError("invalid_notional_reference")
            qty = min(qty, maximum / reference)
    return market_entry_quantity(qty, rules)
