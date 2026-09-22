"""Tests for parsing Gamma's payloads and filtering them down to tradeable rows.

Gamma hands back several fields as JSON *strings* and mixes types between rows,
so the parsing here is checked against the real shapes recorded in the fixtures.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmjev.config import DiscoveryConfig
from pmjev.models import Market
from pmjev.polymarket import fetch_pages, filter_markets, refresh_prices

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pmjev", "fixtures")
NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def recorded_rows():
    with open(os.path.join(FIXTURES, "markets.json"), encoding="utf-8") as handle:
        return json.load(handle)


def test_parses_the_recorded_gamma_rows():
    markets = [Market.from_gamma(row) for row in recorded_rows()]
    assert len(markets) == 4
    for market in markets:
        assert market.id
        assert market.outcomes == ["Yes", "No"]
        assert len(market.clob_token_ids) == 2
        assert market.description
        assert market.end_date is not None
        assert market.end_date.tzinfo is not None


def test_json_string_fields_are_decoded():
    market = Market.from_gamma({
        "id": 42,
        "slug": "x",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.25", "0.75"]',
        "clobTokenIds": '["111", "222"]',
    })
    assert market.id == "42"                    # coerced to str
    assert market.outcomes == ["Yes", "No"]
    assert market.outcome_prices == [0.25, 0.75]
    assert market.clob_token_ids == ["111", "222"]


def test_malformed_json_fields_do_not_raise():
    market = Market.from_gamma({"id": "1", "outcomes": "not json",
                                "outcomePrices": None, "clobTokenIds": "["})
    assert market.outcomes == []
    assert market.outcome_prices == []
    assert market.clob_token_ids == []


def test_yes_price_prefers_the_book_midpoint():
    market = Market.from_gamma({
        "id": "1", "bestBid": 0.40, "bestAsk": 0.44,
        "outcomePrices": '["0.10", "0.90"]',
    })
    assert market.yes_price == pytest.approx(0.42)


def test_yes_price_falls_back_to_the_listed_price():
    market = Market.from_gamma({"id": "1", "outcomePrices": '["0.10", "0.90"]'})
    assert market.yes_price == pytest.approx(0.10)


def test_trailing_z_timestamps_parse_as_utc():
    market = Market.from_gamma({"id": "1", "endDate": "2027-01-01T04:59:00Z"})
    assert market.end_date == datetime(2027, 1, 1, 4, 59, tzinfo=timezone.utc)


def test_days_to_resolution_is_measured_in_code_not_by_the_model():
    market = Market.from_gamma({"id": "1", "endDate": "2026-09-29T12:00:00Z"})
    assert market.days_to_resolution(NOW) == pytest.approx(10.0)


# --- filtering ------------------------------------------------------------

def base_row(**overrides):
    row = {
        "id": "1", "slug": "s", "question": "Will X happen?",
        "description": "Resolves YES if X happens by the deadline.",
        "outcomes": '["Yes", "No"]', "outcomePrices": '["0.40", "0.60"]',
        "clobTokenIds": '["1", "2"]', "endDate": "2026-10-19T12:00:00Z",
        "volumeNum": 100000.0, "liquidityNum": 50000.0,
        "bestBid": 0.39, "bestAsk": 0.41, "spread": 0.02,
        "acceptingOrders": True, "closed": False,
    }
    row.update(overrides)
    return row


def test_filter_keeps_a_healthy_market():
    cfg = DiscoveryConfig()
    kept = filter_markets([Market.from_gamma(base_row())], cfg, now=NOW)
    assert len(kept) == 1


@pytest.mark.parametrize("override,reason", [
    ({"closed": True}, "closed"),
    ({"acceptingOrders": False}, "not accepting orders"),
    ({"liquidityNum": 10.0}, "illiquid"),
    ({"volumeNum": 10.0}, "no volume"),
    ({"outcomePrices": '["0.99", "0.01"]', "bestBid": 0.985, "bestAsk": 0.995}, "already settled"),
    ({"spread": 0.40}, "spread too wide"),
    ({"endDate": "2029-01-01T00:00:00Z"}, "resolves too far out"),
    ({"endDate": "2026-09-18T00:00:00Z"}, "already past its end date"),
    ({"description": ""}, "no rules text for Jev to read"),
    ({"outcomes": '["Trump", "Biden"]'}, "not a Yes/No market"),
    ({"question": "Lakers vs. Celtics"}, "sports"),
    ({"question": "What price will BTC hit above $200,000?"}, "resolves off a price feed"),
])
def test_filter_drops_untradeable_markets(override, reason):
    cfg = DiscoveryConfig()
    kept = filter_markets([Market.from_gamma(base_row(**override))], cfg, now=NOW)
    assert kept == [], "should have dropped: {0}".format(reason)


def test_include_slugs_bypasses_every_filter():
    # Pinning a market by slug is an explicit instruction, so it overrides the
    # heuristics rather than being silently dropped by them.
    cfg = DiscoveryConfig(include_slugs=["s"])
    row = base_row(liquidityNum=0.0, volumeNum=0.0, closed=True)
    kept = filter_markets([Market.from_gamma(row)], cfg, now=NOW)
    assert len(kept) == 1


def test_refresh_prices_overwrites_the_stale_book():
    class FakeClob:
        def prices(self, token_ids, side="BUY"):
            return {"1": 0.55} if side == "BUY" else {"1": 0.53}

    market = Market.from_gamma(base_row())
    refresh_prices([market], FakeClob())
    assert market.best_ask == pytest.approx(0.55)
    assert market.best_bid == pytest.approx(0.53)
    assert market.spread == pytest.approx(0.02)


def test_refresh_prices_ignores_a_crossed_book():
    class CrossedClob:
        def prices(self, token_ids, side="BUY"):
            return {"1": 0.40} if side == "BUY" else {"1": 0.60}

    market = Market.from_gamma(base_row())
    refresh_prices([market], CrossedClob())
    assert market.best_ask == pytest.approx(0.41)   # unchanged
    assert market.best_bid == pytest.approx(0.39)


def test_discover_pages_past_the_first_batch():
    """Regression: a single 250-row page yielded 3 eligible markets, 1000 yielded 56.

    Gamma returns rows volume-descending and the top of that ordering is
    almost entirely sports and crypto-price markets, which this strategy
    excludes. Without pagination the scanner starves.
    """
    from pmjev.polymarket import PAGE_SIZE, fetch_pages

    class PagedGamma:
        def __init__(self):
            self.calls = []

        def list_markets(self, limit=250, offset=0, **kwargs):
            self.calls.append((limit, offset))
            if offset >= 1000:
                return []
            return [Market.from_gamma({"id": str(offset + i), "slug": "s{0}".format(offset + i)})
                    for i in range(min(limit, PAGE_SIZE))]

    gamma = PagedGamma()
    markets = fetch_pages(gamma, fetch_limit=1000)
    assert len(markets) == 1000
    assert len({m.id for m in markets}) == 1000, "offsets must not overlap"
    assert gamma.calls[0] == (PAGE_SIZE, 0)
    assert gamma.calls[1] == (PAGE_SIZE, PAGE_SIZE)
    assert len(gamma.calls) == 10


def test_a_short_page_does_not_end_pagination():
    """Gamma caps a page at 100 rows however many you ask for, and says nothing.

    Treating a short page as "listing exhausted" is what limited the scanner
    to its first 100 rows, so a short page must keep paging.
    """
    class CappingGamma:
        """Honours offset but silently truncates every page to 100 rows."""

        def __init__(self, total):
            self.total = total

        def list_markets(self, limit=250, offset=0, **kwargs):
            end = min(offset + 100, self.total)
            return [Market.from_gamma({"id": str(i), "slug": "s{0}".format(i)})
                    for i in range(offset, end)]

    markets = fetch_pages(CappingGamma(total=430), fetch_limit=1000)
    assert len(markets) == 430
    assert len({m.id for m in markets}) == 430


def test_fetch_pages_stops_when_the_listing_runs_out():
    class EmptyGamma:
        def list_markets(self, limit=250, offset=0, **kwargs):
            return []

    assert fetch_pages(EmptyGamma(), fetch_limit=1000) == []


def test_rss_summaries_have_entities_decoded():
    """RSS descriptions arrive entity-encoded; Jev should not have to read &quot;."""
    from pmjev.evidence import strip_html

    raw = '<p>The IRGC says the US &quot;must accept the region&#039;s freedom&quot;</p>'
    assert strip_html(raw) == 'The IRGC says the US "must accept the region\'s freedom"'
    assert strip_html("<b>a</b>   &amp;   <i>b</i>") == "a & b"
    assert strip_html("") == ""


def test_markets_without_an_end_date_can_be_required_to_have_one():
    """A "resolving within 10 days" scan must not return undated markets.

    Gamma leaves endDate unset on some rows. They used to bypass the date
    window entirely, so Senate races with no listed date turned up in a
    short-dated scan alongside genuine 8-day markets.
    """
    undated = Market.from_gamma(base_row(endDate=None))

    lenient = DiscoveryConfig(max_days_to_resolution=10.0)
    assert len(filter_markets([undated], lenient, now=NOW)) == 1

    strict = DiscoveryConfig(max_days_to_resolution=10.0, require_end_date=True)
    assert filter_markets([undated], strict, now=NOW) == []


def test_requiring_an_end_date_still_keeps_dated_markets_in_the_window():
    strict = DiscoveryConfig(max_days_to_resolution=30.0, require_end_date=True)
    inside = Market.from_gamma(base_row(endDate="2026-09-29T12:00:00Z"))
    assert len(filter_markets([inside], strict, now=NOW)) == 1
