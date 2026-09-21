"""Plain data records passed between the stages of the pipeline.

Everything here is a dumb container. No network, no judgment, no math beyond
trivial derived properties -- that keeps the pipeline easy to test with fixtures.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso(value: Optional[str]) -> Optional[datetime]:
    """Parse the ISO-8601 shapes Gamma actually emits (trailing Z, or +00:00)."""
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def json_field(raw: Any, default: Any) -> Any:
    """Gamma returns several fields as JSON *strings*. Decode defensively."""
    if raw is None:
        return default
    if isinstance(raw, (list, dict)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return default
    return default


@dataclass
class Market:
    """One binary Polymarket market (a single Yes/No row from the Gamma API)."""

    id: str
    slug: str
    question: str
    description: str                  # the resolution rules -- what Jev reads
    outcomes: List[str]
    outcome_prices: List[float]
    clob_token_ids: List[str]
    end_date: Optional[datetime]
    volume_num: float
    liquidity_num: float
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]
    last_trade_price: Optional[float]
    updated_at: Optional[datetime]
    accepting_orders: bool
    closed: bool
    neg_risk: bool
    resolution_source: str = ""
    event_title: str = ""
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ---- derived ---------------------------------------------------------

    @property
    def url(self) -> str:
        return "https://polymarket.com/market/{0}".format(self.slug)

    @property
    def yes_token_id(self) -> Optional[str]:
        return self.clob_token_ids[0] if self.clob_token_ids else None

    @property
    def no_token_id(self) -> Optional[str]:
        return self.clob_token_ids[1] if len(self.clob_token_ids) > 1 else None

    @property
    def yes_price(self) -> Optional[float]:
        """Best read of P(Yes): mid of the book, else the listed outcome price."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        if self.outcome_prices:
            return self.outcome_prices[0]
        return self.last_trade_price

    def days_to_resolution(self, now: Optional[datetime] = None) -> Optional[float]:
        if self.end_date is None:
            return None
        return (self.end_date - (now or utcnow())).total_seconds() / 86400.0

    def is_binary_yes_no(self) -> bool:
        return [o.strip().lower() for o in self.outcomes] == ["yes", "no"]

    def keywords(self) -> str:
        """A news query for this market: the event title plus the group item.

        Deliberately crude. The screen stage is what separates signal from the
        junk a keyword search returns, so over-tuning this would be wasted work.
        """
        item = (self.raw.get("groupItemTitle") or "").strip()
        base = (self.event_title or self.question).strip()
        if item and item.lower() not in base.lower():
            return "{0} {1}".format(base, item)
        return base

    # ---- construction ----------------------------------------------------

    @classmethod
    def from_gamma(cls, data: Dict[str, Any]) -> "Market":
        def as_float(key: str) -> Optional[float]:
            value = data.get(key)
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        prices: List[float] = []
        for raw_price in json_field(data.get("outcomePrices"), []):
            try:
                prices.append(float(raw_price))
            except (TypeError, ValueError):
                pass

        events = data.get("events") or []
        event_title = ""
        if isinstance(events, list) and events and isinstance(events[0], dict):
            event_title = events[0].get("title", "") or ""

        return cls(
            id=str(data.get("id", "")),
            slug=data.get("slug", "") or "",
            question=data.get("question", "") or "",
            description=data.get("description", "") or "",
            outcomes=[str(o) for o in json_field(data.get("outcomes"), [])],
            outcome_prices=prices,
            clob_token_ids=[str(t) for t in json_field(data.get("clobTokenIds"), [])],
            end_date=parse_iso(data.get("endDate")),
            volume_num=float(data.get("volumeNum") or 0.0),
            liquidity_num=float(data.get("liquidityNum") or 0.0),
            best_bid=as_float("bestBid"),
            best_ask=as_float("bestAsk"),
            spread=as_float("spread"),
            last_trade_price=as_float("lastTradePrice"),
            updated_at=parse_iso(data.get("updatedAt")),
            accepting_orders=bool(data.get("acceptingOrders", False)),
            closed=bool(data.get("closed", False)),
            neg_risk=bool(data.get("negRisk", False)),
            resolution_source=data.get("resolutionSource", "") or "",
            event_title=event_title,
            raw=data,
        )


@dataclass
class Headline:
    """A candidate piece of evidence, before we decide it is worth a deep read."""

    title: str
    source: str
    url: str
    published: Optional[datetime]
    snippet: str = ""

    def age_hours(self, now: Optional[datetime] = None) -> Optional[float]:
        if self.published is None:
            return None
        return ((now or utcnow()) - self.published).total_seconds() / 3600.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "published": self.published.isoformat() if self.published else None,
            "snippet": self.snippet,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Headline":
        return cls(
            title=data.get("title", ""),
            source=data.get("source", ""),
            url=data.get("url", ""),
            published=parse_iso(data.get("published")),
            snippet=data.get("snippet", "") or "",
        )


# Level counts for the Score questions defined in questions.py. Kept here so
# scoring.py can normalise a raw score without importing the question text.
DIRECTNESS_LEVELS = 4
AMBIGUITY_LEVELS = 4
AUTHORITY_LEVELS = 4


@dataclass
class Judgment:
    """The typed answers Jev returned for one (market, headline) pair.

    Field names mirror the question ids in questions.py, so any number here can
    be traced straight back to the text that produced it.
    """

    market_id: str
    headline: Headline
    condition_met: float           # noul: rules already satisfied by this evidence
    condition_precluded: float     # noul: a Yes resolution is already impossible
    stance: str                    # choice: supports_yes|supports_no|mixed|not_relevant
    stance_probs: Dict[str, float]
    stance_confidence: float
    directness: float              # score 0..3
    directness_confidence: float
    rule_ambiguity: float          # score 0..3
    rule_ambiguity_confidence: float
    source_authority: float        # score 0..3
    source_authority_confidence: float
    screen_probability: float = 0.0
    input_tokens: int = 0
    model: str = ""
    judged_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["headline"] = self.headline.to_dict()
        out["judged_at"] = self.judged_at.isoformat()
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Judgment":
        payload = dict(data)
        payload["headline"] = Headline.from_dict(payload.get("headline") or {})
        judged = parse_iso(payload.pop("judged_at", None))
        payload["judged_at"] = judged or utcnow()
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in allowed})


@dataclass
class Signal:
    """A scored result for one market -- actionable or explicitly not."""

    market: Market
    judgment: Optional[Judgment]
    side: str                      # "YES" | "NO" | "NONE"
    p_model: Optional[float]
    market_price: Optional[float]  # price of the side we would buy
    edge: Optional[float]
    kelly_stake: Optional[float]   # fraction of bankroll, after caps
    dollars: Optional[float]
    gate: str                      # "TRADE" | "WATCH" | "AVOID" | "NO_VIEW"
    reasons: List[str] = field(default_factory=list)
    scanned_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scanned_at": self.scanned_at.isoformat(),
            "market": {
                "id": self.market.id,
                "slug": self.market.slug,
                "question": self.market.question,
                "url": self.market.url,
                "end_date": self.market.end_date.isoformat() if self.market.end_date else None,
                "yes_price": self.market.yes_price,
                "best_bid": self.market.best_bid,
                "best_ask": self.market.best_ask,
                "liquidity": self.market.liquidity_num,
                "volume": self.market.volume_num,
            },
            "judgment": self.judgment.to_dict() if self.judgment else None,
            "side": self.side,
            "p_model": self.p_model,
            "market_price": self.market_price,
            "edge": self.edge,
            "kelly_stake": self.kelly_stake,
            "dollars": self.dollars,
            "gate": self.gate,
            "reasons": list(self.reasons),
        }


@dataclass
class RunStats:
    """Bookkeeping for one scan: what it cost, and where markets dropped out."""

    markets_fetched: int = 0
    markets_after_filter: int = 0
    headlines_found: int = 0
    headlines_screened_in: int = 0
    deep_judgments: int = 0
    requests_made: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: List[str] = field(default_factory=list)

    def cost_usd(self, price_per_billion: float) -> float:
        # Output tokens are free on Jev; only input tokens are billed.
        return self.input_tokens * price_per_billion / 1e9

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
