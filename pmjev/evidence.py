"""Evidence providers: where the text Jev reads comes from.

A note on what is actually available for free. Google News RSS gives a headline,
a publisher and a timestamp, but its article links are JavaScript redirects, so
full body text cannot be fetched from it without a scraping stack. For the
resolution-lag question that is largely fine -- "X was sworn in", "the deal
collapsed" is exactly what a headline says -- but it is a real limit, and the
honest fix is a provider that returns body text. `RssProvider` (publisher feeds,
which do carry summaries) and `FileProvider` (evidence you supply) are both here
for that, and a keyed news API drops in behind the same interface.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree

import requests

from .config import EvidenceConfig
from .models import Headline, Market, utcnow

log = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    if not text:
        return ""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", text)).strip()


def parse_rfc822(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def split_title_source(title: str) -> Tuple[str, str]:
    """Google News titles end with ' - Publisher'. Split it off when present."""
    if " - " in title:
        head, _, tail = title.rpartition(" - ")
        if head and len(tail) < 60:
            return head.strip(), tail.strip()
    return title.strip(), ""


class EvidenceProvider:
    """Interface: given a market, return candidate headlines, newest first."""

    name = "base"

    def search(self, market: Market, cfg: EvidenceConfig) -> List[Headline]:
        raise NotImplementedError


class NullProvider(EvidenceProvider):
    name = "none"

    def search(self, market: Market, cfg: EvidenceConfig) -> List[Headline]:
        return []


class GoogleNewsProvider(EvidenceProvider):
    """Free, keyless, worldwide coverage. Headline + publisher + date only."""

    name = "google_news"
    BASE = "https://news.google.com/rss/search"

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()

    def search(self, market: Market, cfg: EvidenceConfig) -> List[Headline]:
        days = max(1, int(round(cfg.lookback_hours / 24.0)))
        query = "{0} when:{1}d".format(market.keywords(), days)
        params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
        try:
            response = self.session.get(
                self.BASE, params=params,
                headers={"User-Agent": cfg.user_agent}, timeout=20,
            )
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
        except (requests.RequestException, ElementTree.ParseError) as exc:
            log.warning("news search failed for %s: %s", market.slug, exc)
            return []

        cutoff = utcnow() - timedelta(hours=cfg.lookback_hours)
        found: List[Headline] = []
        for item in root.findall("./channel/item"):
            raw_title = (item.findtext("title") or "").strip()
            if not raw_title:
                continue
            title, inline_source = split_title_source(raw_title)
            source_el = item.find("source")
            source = (source_el.text if source_el is not None else "") or inline_source
            published = parse_rfc822(item.findtext("pubDate"))
            if published is not None and published < cutoff:
                continue
            found.append(Headline(
                title=title,
                source=(source or "").strip(),
                url=(item.findtext("link") or "").strip(),
                published=published,
                # Google's description is only a link wrapper; the headline is
                # the evidence. Recorded as-is so the log shows what Jev saw.
                snippet=title,
            ))
        log.debug("google_news: %d hits for %r", len(found), query)
        return _dedupe(found)[: cfg.max_headlines]


class RssProvider(EvidenceProvider):
    """Publisher feeds listed in config. Slower to match, but carries real text."""

    name = "rss"

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()
        self._cache: Dict[str, List[Headline]] = {}

    def _fetch_feed(self, url: str, cfg: EvidenceConfig) -> List[Headline]:
        if url in self._cache:
            return self._cache[url]
        items: List[Headline] = []
        try:
            response = self.session.get(
                url, headers={"User-Agent": cfg.user_agent}, timeout=20)
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
        except (requests.RequestException, ElementTree.ParseError) as exc:
            log.warning("rss feed failed %s: %s", url, exc)
            self._cache[url] = []
            return []

        channel_title = root.findtext("./channel/title") or url
        for item in root.findall("./channel/item"):
            title = (item.findtext("title") or "").strip()
            if not title:
                continue
            summary = strip_html(item.findtext("description") or "")
            items.append(Headline(
                title=title,
                source=channel_title,
                url=(item.findtext("link") or "").strip(),
                published=parse_rfc822(item.findtext("pubDate")),
                snippet=summary or title,
            ))
        self._cache[url] = items
        return items

    def search(self, market: Market, cfg: EvidenceConfig) -> List[Headline]:
        terms = _query_terms(market.keywords())
        if not terms:
            return []
        cutoff = utcnow() - timedelta(hours=cfg.lookback_hours)
        scored: List[Any] = []
        for feed_url in cfg.rss_feeds:
            for headline in self._fetch_feed(feed_url, cfg):
                if headline.published is not None and headline.published < cutoff:
                    continue
                haystack = "{0} {1}".format(headline.title, headline.snippet).lower()
                hits = sum(1 for term in terms if term in haystack)
                if hits:
                    scored.append((hits, headline))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return _dedupe([h for _, h in scored])[: cfg.max_headlines]


class FileProvider(EvidenceProvider):
    """Evidence you supply yourself, keyed by market slug or market id.

    Shape:
        {"<slug-or-id>": [{"title": ..., "source": ..., "url": ...,
                           "published": "2026-09-18T12:00:00Z", "snippet": ...}]}
    """

    name = "file"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._data: Dict[str, List[Dict[str, Any]]] = {}
        if self.path.is_file():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            log.warning("evidence file not found: %s", self.path)

    def search(self, market: Market, cfg: EvidenceConfig) -> List[Headline]:
        rows = self._data.get(market.slug) or self._data.get(market.id) or []
        return [Headline.from_dict(row) for row in rows][: cfg.max_headlines]


def _query_terms(text: str) -> List[str]:
    stop = {
        "will", "the", "a", "an", "be", "in", "on", "of", "to", "by", "for",
        "and", "or", "is", "are", "next", "before", "after", "any", "who",
        "what", "when", "which", "this", "that", "at", "from", "with",
    }
    words = re.findall(r"[a-z0-9']+", text.lower())
    return [w for w in words if len(w) > 2 and w not in stop]


def _dedupe(headlines: Iterable[Headline]) -> List[Headline]:
    """Drop near-duplicate syndicated copies, keeping the earliest seen."""
    seen = set()
    out: List[Headline] = []
    for headline in headlines:
        key = _WS_RE.sub(" ", headline.title.lower()).strip()[:90]
        if key in seen:
            continue
        seen.add(key)
        out.append(headline)
    out.sort(key=lambda h: h.published or datetime.min.replace(tzinfo=timezone.utc),
             reverse=True)
    return out


def build_provider(cfg: EvidenceConfig, root: Optional[Path] = None,
                   session: Optional[requests.Session] = None) -> EvidenceProvider:
    name = (cfg.provider or "none").lower()
    if name in ("google_news", "google", "news"):
        return GoogleNewsProvider(session=session)
    if name == "rss":
        return RssProvider(session=session)
    if name == "file":
        path = Path(cfg.evidence_file)
        if root is not None and not path.is_absolute():
            path = root / path
        return FileProvider(path)
    if name in ("none", "null", ""):
        return NullProvider()
    raise ValueError("unknown evidence provider {0!r}".format(cfg.provider))
