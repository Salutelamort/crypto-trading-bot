"""Shared deterministic paper execution rules. No network or persistence."""
import math

MODEL_VERSION = "paper-v4"


def effective_risk(genome, risk):
    result = dict(risk)
    for gene, setting in (("stop_atr", "atr_stop_mult"), ("trail_atr", "atr_trail_mult")):
        if genome.get(gene):
            result[setting] = genome[gene]
    rr = risk.get("fixed_rr", genome.get("rr"))
    if genome.get("stop_atr") and rr:
        result["atr_take_mult"] = round(genome["stop_atr"] * rr, 3)
    return result


def exit_levels(direction, entry, extreme, atr, risk):
    if risk.get("atr_stop") and atr is not None and math.isfinite(atr) and atr > 0:
        stop = risk.get("atr_stop_mult", 2.0) * atr
        take = risk.get("atr_take_mult", 6.0) * atr
        trail = risk.get("atr_trail_mult", 2.5) * atr
    else:
        stop, take = entry * risk["stop_loss_pct"], entry * risk["take_profit_pct"]
        trail = extreme * risk["trailing_stop_pct"]
    if direction == 1:
        return max(entry - stop, extreme - trail), entry + take
    return min(entry + stop, extreme + trail), entry - take


def protective_exit(direction, stop, take, high, low, opened=None):
    """Opening gaps precede intrabar extremes; otherwise stop wins OHLC ambiguity."""
    if opened is not None:
        if (direction == 1 and opened <= stop) or (direction == -1 and opened >= stop):
            return "stop", opened
        if (direction == 1 and opened >= take) or (direction == -1 and opened <= take):
            return "take_profit", take
    if (direction == 1 and low <= stop) or (direction == -1 and high >= stop):
        return "stop", stop
    if (direction == 1 and high >= take) or (direction == -1 and low <= take):
        return "take_profit", take
    return None


def cooldown_bars(genome):
    # Exit candle is always excluded, including a configured cooldown of zero.
    return max(1, int(genome.get("cooldown", 0)))


def fill_price(reference, side, slippage, quote=None, quantity=None, max_spread=None):
    """side +1 buys at ask, -1 sells at bid; deterministic adverse slippage follows.

    Top-of-book quantities constrain admission, not a claim of complete order-book depth.
    Protective exits may use quantity=None: liquidity uncertainty is separately reported.
    """
    if side not in (-1, 1) or not math.isfinite(reference) or reference <= 0:
        raise ValueError("invalid fill reference or side")
    if not math.isfinite(slippage) or not 0 <= slippage < 1:
        raise ValueError("invalid slippage")
    base = reference
    if quote is not None:
        bid, ask = quote["bid"], quote["ask"]
        if not (0 < bid <= ask and math.isfinite(bid) and math.isfinite(ask)):
            raise ValueError("invalid book")
        spread = (ask - bid) / ((ask + bid) / 2)
        if max_spread is not None and spread > max_spread:
            raise ValueError("spread_limit")
        if quantity is not None and quantity > quote["ask_qty" if side == 1 else "bid_qty"]:
            raise ValueError("insufficient_book_depth")
        base = ask if side == 1 else bid
    return base * (1 + slippage * side)


def depth_entry(budget, side, slippage, quote, participation, consumed=0, quantity=None):
    """Market IOC entry: walk visible levels, cancel unfilled remainder.

    Participation applies per level, and consumed is this account's prior quantity
    against this snapshot. No queue-position or market-impact claim is made.
    """
    if not 0 < participation <= 1 or not math.isfinite(budget) or budget <= 0:
        raise ValueError("invalid_depth_budget")
    levels = quote["asks" if side == 1 else "bids"]
    left = float("inf") if quantity is None else quantity
    cost, filled = 0., 0.
    for price, size in levels:
        available = size * participation
        skipped = min(available, consumed)
        consumed -= skipped
        available -= skipped
        price *= 1 + side * slippage
        take = min(available, left, max(0, budget - cost) / price)
        cost += take * price
        filled += take
        left -= take
        if left <= 0:
            break
    if filled <= 0:
        raise ValueError("insufficient_book_depth")
    if quantity is not None and not math.isclose(filled, quantity, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("insufficient_book_depth")
    return filled, cost / filled


def depth_exit(quantity, side, slippage, quote, participation, consumed=0):
    """Fill up to the requested quantity; never invent liquidity for the remainder."""
    left, filled, value = quantity, 0., 0.
    for price, size in quote["asks" if side == 1 else "bids"]:
        available = size * participation
        skipped = min(available, consumed)
        consumed -= skipped
        take = min(left, available - skipped)
        filled += take
        value += take * price * (1 + side * slippage)
        left -= take
        if left <= 0:
            break
    if filled <= 0:
        raise ValueError("insufficient_exit_depth")
    return filled, value / filled
