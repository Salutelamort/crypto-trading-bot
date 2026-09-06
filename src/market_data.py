"""Public Binance bid/ask stream with validated, fresh REST fallback. No API keys."""
import copy
import json
import math
import threading
import time

import requests
import websocket

from .data_feed import BINANCE_FALLBACKS


def parse_depth(payload, source, received_at=None):
    levels = {}
    for side in ("bids", "asks"):
        rows = [(float(p), float(q)) for p, q in payload[side]]
        if not rows or len(rows) > 20 or any(not math.isfinite(v) or v <= 0 for row in rows for v in row):
            raise ValueError("invalid_depth")
        prices = [row[0] for row in rows]
        if prices != sorted(set(prices), reverse=side == "bids"):
            raise ValueError("unordered_depth")
        levels[side] = rows
    quote = parse_book({"b": levels["bids"][0][0], "B": levels["bids"][0][1],
                        "a": levels["asks"][0][0], "A": levels["asks"][0][1]}, source, received_at)
    quote.update(levels, update_id=payload["lastUpdateId"])
    return quote


def parse_book(payload, source, received_at=None, latency_seconds=0):
    if not isinstance(payload, dict):
        raise TypeError("invalid order book payload")
    values = {"bid": float(payload.get("bidPrice", payload.get("b", 0))),
              "ask": float(payload.get("askPrice", payload.get("a", 0))),
              "bid_qty": float(payload.get("bidQty", payload.get("B", 0))),
              "ask_qty": float(payload.get("askQty", payload.get("A", 0)))}
    if (not all(math.isfinite(x) and x > 0 for x in values.values())
            or values["bid"] > values["ask"]):
        raise ValueError("invalid order book")
    values.update(source=source, received_at=received_at if received_at is not None else time.time(),
                  latency_seconds=latency_seconds)
    return values


def rest_book(symbol):
    errors = []
    for base in BINANCE_FALLBACKS:
        try:
            started = time.monotonic()
            response = requests.get(base + "/api/v3/ticker/bookTicker",
                                    params={"symbol": symbol}, timeout=5)
            response.raise_for_status()
            return parse_book(response.json(), "binance_rest", latency_seconds=time.monotonic() - started)
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            errors.append(type(exc).__name__)
    raise RuntimeError("book unavailable: " + ",".join(errors))


def rest_depth(symbol):
    for base in BINANCE_FALLBACKS:
        try:
            started = time.monotonic()
            response = requests.get(base + "/api/v3/depth", params={"symbol": symbol, "limit": 20}, timeout=5)
            response.raise_for_status()
            quote = parse_depth(response.json(), "binance_depth20_rest")
            quote["latency_seconds"] = time.monotonic() - started
            return quote
        except (requests.RequestException, ValueError, KeyError, TypeError):
            continue
    raise RuntimeError("depth unavailable")


class BookStream:
    """Reconnects after disconnects; never reuses stale quotes as if fresh."""
    def __init__(self, symbols, max_age_seconds=3, require_depth=False):
        self.symbols = set(symbols)
        self.require_depth = require_depth
        self.max_age_seconds = max_age_seconds
        self.lock = threading.Lock()
        self.quotes = {}
        self.depths = {}
        self.stop_event = threading.Event()
        self.ws = None
        self.thread = None

    def _message(self, _ws, message):
        try:
            payload = json.loads(message)
            if not isinstance(payload, dict):
                return
            stream = payload.get("stream", "")
            payload = payload.get("data", payload)
            if not isinstance(payload, dict):
                return
            if "@depth20" in stream:
                symbol = stream.split("@", 1)[0].upper()
                if symbol in self.symbols:
                    quote = parse_depth(payload, "binance_depth20_websocket")
                    with self.lock:
                        self.depths[symbol] = quote
                return
            symbol = payload["s"]
            if symbol not in self.symbols:
                return
            quote = parse_book(payload, "binance_websocket")
            with self.lock:
                self.quotes[symbol] = quote
        except (KeyError, ValueError, TypeError):
            return

    def _run(self):
        streams = "/".join(s.lower() + suffix for s in sorted(self.symbols)
                           for suffix in ("@bookTicker", "@depth20@100ms"))
        while not self.stop_event.is_set():
            self.ws = websocket.WebSocketApp(
                "wss://data-stream.binance.vision/stream?streams=" + streams,
                on_message=self._message)
            try:
                if self.stop_event.is_set():
                    break
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
            except (websocket.WebSocketException, OSError):
                pass  # Retry connection; callers use REST while disconnected.
            finally:
                with self.lock:
                    self.quotes.clear()
                    self.depths.clear()
            self.stop_event.wait(3)

    def start(self):
        if self.thread and self.thread.is_alive():
            return self
        if self.stop_event.is_set():
            raise RuntimeError("closed stream cannot be restarted")
        if not self.symbols:
            return self
        self.thread = threading.Thread(target=self._run, daemon=True, name="public-book-stream")
        self.thread.start()
        return self

    def book(self, symbol):
        with self.lock:
            depth = self.depths.get(symbol)
            if depth and 0 <= time.time() - depth["received_at"] <= self.max_age_seconds:
                return copy.deepcopy(depth)
            quote = self.quotes.get(symbol)
            if not self.require_depth and quote and 0 <= time.time() - quote["received_at"] <= self.max_age_seconds:
                return dict(quote)
        return rest_depth(symbol) if self.require_depth else rest_book(symbol)

    def close(self):
        self.stop_event.set()
        if self.ws:
            self.ws.close()
        if self.thread:
            self.thread.join(timeout=5)
