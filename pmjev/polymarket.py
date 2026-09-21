"""Polymarket read-only clients: Gamma (markets) and CLOB (live prices).

Both are public and unauthenticated. Nothing in this module can place an order --
there is no signing code here and no wallet, by design.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests

from .config import DiscoveryConfig
from .models import Market, utcnow

log = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


class GammaClient:
    """Market discovery and metadata."""

    def __init__(self, base_url: str = GAMMA_BASE, timeout: float = 20.0,
                 session: Optional[requests.Session] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = "{0}/{1}".format(self.base_url, path.lstrip("/"))
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def list_markets(self, limit: int = 250, offset: int = 0,
                     order: str = "volumeNum", ascending: bool = False,
                     tag_id: Optional[int] = None,
                     closed: bool = False) -> List[Market]:
        """One page of open markets, richest first by default."""
        params: Dict[str, Any] = {
            "limit": min(limit, 500),
            "offset": offset,
            "order": order,
            "ascending": str(ascending).lower(),
            "closed": str(closed).lower(),
            "active": "true",
        }
        if tag_id is not None:
            params["tag_id"] = tag_id
        payload = self._get("/markets", params)
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        return [Market.from_gamma(row) for row in rows]

    def get_market(self, market_id: str) -> Optional[Market]:
        try:
            payload = self._get("/markets/{0}".format(market_id))
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise
        if isinstance(payload, list):
            payload = payload[0] if payload else None
        return Market.from_gamma(payload) if payload else None

    def get_market_by_slug(self, slug: str) -> Optional[Market]:
        payload = self._get("/markets", {"slug": slug})
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        return Market.from_gamma(rows[0]) if rows else None


class ClobClient:
    """Live book prices. Gamma's bestBid/bestAsk lag; these do not."""

    def __init__(self, base_url: str = CLOB_BASE, timeout: float = 20.0,
                 session: Optional[requests.Session] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def midpoints(self, token_ids: Sequence[str]) -> Dict[str, float]:
        """Batch midpoints. The endpoint accepts up to 500 ids per request."""
        out: Dict[str, float] = {}
        ids = [t for t in token_ids if t]
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            body = [{"token_id": t} for t in chunk]
            try:
                response = self.session.post(
                    "{0}/midpoints".format(self.base_url), json=body,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                for token_id, value in (response.json() or {}).items():
                    try:
                        out[str(token_id)] = float(value)
                    except (TypeError, ValueError):
                        continue
            except (requests.RequestException, ValueError) as exc:
                log.warning("CLOB midpoints failed for %d ids: %s", len(chunk), exc)
        return out

    def prices(self, token_ids: Sequence[str], side: str = "BUY") -> Dict[str, float]:
        """Batch best prices for one side. BUY = the ask you would pay."""
        out: Dict[str, float] = {}
        ids = [t for t in token_ids if t]
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            body = [{"token_id": t, "side": side} for t in chunk]
            try:
                response = self.session.post(
                    "{0}/prices".format(self.base_url), json=body,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json() or {}
                for token_id, value in payload.items():
                    # /prices returns {token_id: {"BUY": "0.42"}} or a bare value.
                    if isinstance(value, dict):
                        value = value.get(side) or value.get(side.upper())
                    try:
                        out[str(token_id)] = float(value)
                    except (TypeError, ValueError):
                        continue
            except (requests.RequestException, ValueError) as exc:
                log.warning("CLOB prices failed for %d ids: %s", len(chunk), exc)
        return out


def refresh_prices(markets: Sequence[Market], clob: ClobClient) -> None:
    """Overwrite each market's book with live CLOB quotes, in place.

    Gamma's bestBid/bestAsk can be minutes stale. For a strategy whose entire
    premise is "the price has not caught up yet", using a stale price would
    manufacture edge that is not there.
    """
    token_ids: List[str] = []
    for market in markets:
        if market.yes_token_id:
            token_ids.append(market.yes_token_id)

    asks = clob.prices(token_ids, side="BUY")
    bids = clob.prices(token_ids, side="SELL")
    for market in markets:
        token = market.yes_token_id
        if not token:
            continue
        ask, bid = asks.get(token), bids.get(token)
        if ask is not None and bid is not None and bid <= ask:
            market.best_ask = ask
            market.best_bid = bid
            market.spread = round(ask - bid, 6)


def _excluded_by_keyword(market: Market, keywords: Iterable[str]) -> Optional[str]:
    haystack = "{0} {1}".format(market.question, market.event_title).lower()
    for word in keywords:
        if word.lower() in haystack:
            return word
    return None


def filter_markets(markets: Sequence[Market], cfg: DiscoveryConfig,
                   now: Optional[datetime] = None) -> List[Market]:
    """Narrow a raw Gamma page to markets this strategy can actually trade.

    Returns markets in Gamma's order (volume-descending by default); the caller
    truncates to max_markets.
    """
    now = now or utcnow()
    include = {s.lower() for s in cfg.include_slugs}
    kept: List[Market] = []

    for market in markets:
        if include and market.slug.lower() not in include:
            continue
        if not include:
            if market.closed or not market.accepting_orders:
                continue
            if not market.is_binary_yes_no():
                continue
            if market.liquidity_num < cfg.min_liquidity:
                continue
            if market.volume_num < cfg.min_volume:
                continue
            if _excluded_by_keyword(market, cfg.exclude_keywords):
                continue

            price = market.yes_price
            if price is None or not (cfg.price_band_low <= price <= cfg.price_band_high):
                continue
            if market.spread is not None and market.spread > cfg.max_spread:
                continue

            days = market.days_to_resolution(now)
            if days is not None:
                if days < cfg.min_days_to_resolution or days > cfg.max_days_to_resolution:
                    continue
            if not market.description.strip():
                # No rules text means nothing for Jev to check the evidence against.
                continue

        kept.append(market)

    return kept


def discover(cfg: DiscoveryConfig, gamma: Optional[GammaClient] = None,
             clob: Optional[ClobClient] = None,
             now: Optional[datetime] = None) -> List[Market]:
    """Fetch, price-refresh and filter, returning at most cfg.max_markets."""
    gamma = gamma or GammaClient()
    raw: List[Market] = []
    tag_ids: List[Optional[int]] = [t for t in cfg.tag_ids] or [None]
    for tag_id in tag_ids:
        raw.extend(gamma.list_markets(limit=cfg.fetch_limit, tag_id=tag_id))

    seen = set()
    unique: List[Market] = []
    for market in raw:
        if market.id in seen:
            continue
        seen.add(market.id)
        unique.append(market)

    if clob is not None:
        refresh_prices(unique, clob)

    return filter_markets(unique, cfg, now=now)[: cfg.max_markets]
