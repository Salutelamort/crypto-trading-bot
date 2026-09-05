import json
import unittest
from unittest import mock

import requests

from src import market_data


def payload(**changes):
    return {"s": "BTCUSDT", "b": "100", "a": "101", "B": "2", "A": "3", **changes}


class MarketDataTests(unittest.TestCase):
    def test_rejects_invalid_books(self):
        for data in [None, [], payload(b="nan"), payload(a="inf"),
                     payload(b="102"), payload(A="0"), payload(B="-1")]:
            with self.subTest(data=data), self.assertRaises((ValueError, TypeError)):
                market_data.parse_book(data, "test")

    def test_websocket_uses_fresh_book_and_returns_copy(self):
        stream = market_data.BookStream(["BTCUSDT"])
        with mock.patch.object(market_data.time, "time", return_value=100), \
                mock.patch.object(market_data, "rest_book") as rest:
            stream._message(None, json.dumps({"data": payload()}))
            quote = stream.book("BTCUSDT")
            self.assertEqual((quote["bid"], quote["ask"]), (100, 101))
            self.assertEqual(quote["source"], "binance_websocket")
            quote["bid"] = 0
            self.assertEqual(stream.book("BTCUSDT")["bid"], 100)
            rest.assert_not_called()

    def test_old_or_future_quote_uses_rest(self):
        stream = market_data.BookStream(["BTCUSDT"], max_age_seconds=3)
        stream.quotes["BTCUSDT"] = market_data.parse_book(payload(), "test", received_at=100)
        for now in (104, 99):
            with self.subTest(now=now), \
                    mock.patch.object(market_data.time, "time", return_value=now), \
                    mock.patch.object(market_data, "rest_book", return_value={"fallback": True}) as rest:
                self.assertEqual(stream.book("BTCUSDT"), {"fallback": True})
                rest.assert_called_once_with("BTCUSDT")

    def test_malformed_or_unrequested_messages_are_ignored(self):
        stream = market_data.BookStream(["BTCUSDT"])
        for data in ("broken", "[]", "null", '{"data": []}', "{}",
                     json.dumps(payload(s="ETHUSDT")), json.dumps(payload(b="nan"))):
            stream._message(None, data)
        self.assertEqual(stream.quotes, {})

    def test_rest_fails_over_and_validates_response(self):
        response = mock.Mock()
        response.json.return_value = {"bidPrice": "100", "askPrice": "101",
                                      "bidQty": "2", "askQty": "3"}
        with mock.patch.object(market_data.requests, "get", side_effect=[requests.Timeout(), response]) as get:
            quote = market_data.rest_book("BTCUSDT")
        self.assertEqual(quote["source"], "binance_rest")
        self.assertEqual(get.call_count, 2)

    def test_rest_failure_never_returns_old_price(self):
        with mock.patch.object(market_data.requests, "get", side_effect=requests.Timeout()), \
                self.assertRaises(RuntimeError):
            market_data.rest_book("BTCUSDT")

    def test_disconnect_clears_cache_and_reconnects(self):
        stream = market_data.BookStream(["BTCUSDT"])
        stream.quotes["BTCUSDT"] = market_data.parse_book(payload(), "test")
        connection = mock.Mock()
        connection.run_forever.side_effect = OSError("connection lost")

        def stop_after_retry(_seconds):
            self.assertEqual(stream.quotes, {})
            stream.stop_event.set()

        with mock.patch.object(market_data.websocket, "WebSocketApp", return_value=connection), \
                mock.patch.object(stream.stop_event, "wait", side_effect=stop_after_retry):
            stream._run()
        connection.run_forever.assert_called_once()


if __name__ == "__main__":
    unittest.main()
