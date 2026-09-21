"""The two-stage cascade: screen many headlines cheaply, read a few deeply.

Stage 1 sends every candidate headline for a market in one request, as one Noul
each. Stage 2 sends the survivors one at a time with the full question set. That
split exists for two documented reasons: batching independent questions is far
cheaper than asking them separately, and Jev's accuracy degrades when state is
padded with irrelevant context -- so the deep read sees exactly one article.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Protocol, Sequence, Tuple

from .config import Config
from .models import Headline, Judgment, Market, RunStats
from .questions import (deep_questions, deep_state, screen_index,
                        screen_questions, screen_state)
from .typesafe import Answers, TypeSafeError

log = logging.getLogger(__name__)


class SystemOne(Protocol):
    """Anything that can answer a System One request (real client or stub)."""

    def system_one(self, state, questions) -> Answers: ...  # noqa: E704


def screen(client: SystemOne, market: Market, headlines: Sequence[Headline],
           cfg: Config, stats: Optional[RunStats] = None
           ) -> List[Tuple[Headline, float]]:
    """Score every candidate headline for relevance in a single request.

    Returns the survivors as (headline, probability), best first.
    """
    if not headlines:
        return []

    state = screen_state(
        market.question, market.description, headlines,
        rules_chars=cfg.evidence.rules_chars,
    )
    questions = screen_questions(headlines)

    if hasattr(client, "next_tag"):
        client.next_tag = "screen:{0}".format(market.id)  # type: ignore[attr-defined]

    answers = client.system_one(state, questions)
    if stats is not None:
        stats.requests_made += 1
        stats.input_tokens += answers.input_tokens
        stats.output_tokens += answers.output_tokens

    scored: List[Tuple[Headline, float]] = []
    for key in questions:
        probability = answers.noul(key)
        if probability >= cfg.evidence.screen_threshold:
            scored.append((headlines[screen_index(key)], probability))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[: cfg.evidence.max_deep_reads]


def deep_read(client: SystemOne, market: Market, headline: Headline,
              cfg: Config, screen_probability: float = 0.0,
              stats: Optional[RunStats] = None) -> Optional[Judgment]:
    """Ask the six judgments about one article, in one request."""
    state = deep_state(
        market.question, market.description, headline,
        rules_chars=cfg.evidence.rules_chars,
        snippet_chars=cfg.evidence.snippet_chars,
    )
    questions = deep_questions()

    if hasattr(client, "next_tag"):
        client.next_tag = "deep:{0}:{1}".format(  # type: ignore[attr-defined]
            market.id, _headline_key(headline))

    answers = client.system_one(state, questions)
    if stats is not None:
        stats.requests_made += 1
        stats.input_tokens += answers.input_tokens
        stats.output_tokens += answers.output_tokens

    missing = answers.missing(questions)
    if missing:
        log.warning("market %s: missing answers %s", market.slug, missing)
        return None

    return Judgment(
        market_id=market.id,
        headline=headline,
        condition_met=answers.noul("condition_met"),
        condition_precluded=answers.noul("condition_precluded"),
        stance=answers.choice("stance"),
        stance_probs=answers.choice_probs("stance"),
        stance_confidence=answers.confidence("stance"),
        directness=answers.score("directness"),
        directness_confidence=answers.confidence("directness"),
        rule_ambiguity=answers.score("rule_ambiguity"),
        rule_ambiguity_confidence=answers.confidence("rule_ambiguity"),
        source_authority=answers.score("source_authority"),
        source_authority_confidence=answers.confidence("source_authority"),
        screen_probability=screen_probability,
        input_tokens=answers.input_tokens,
        model=answers.model,
    )


def best_judgment(client: SystemOne, market: Market,
                  headlines: Sequence[Headline], cfg: Config,
                  stats: Optional[RunStats] = None) -> Optional[Judgment]:
    """Screen, deep-read the survivors, and keep the most decisive judgment.

    "Most decisive" means furthest from the 0.5 no-information point on either
    Noul -- an article that settles the question in either direction is worth
    more than three that hedge.
    """
    if not headlines:
        return None

    try:
        survivors = screen(client, market, headlines, cfg, stats)
    except TypeSafeError:
        raise
    if stats is not None:
        stats.headlines_screened_in += len(survivors)

    best: Optional[Judgment] = None
    best_decisiveness = -1.0
    for headline, probability in survivors:
        judgment = deep_read(client, market, headline, cfg, probability, stats)
        if judgment is None:
            continue
        if stats is not None:
            stats.deep_judgments += 1
        decisiveness = max(judgment.condition_met, judgment.condition_precluded)
        if decisiveness > best_decisiveness:
            best_decisiveness = decisiveness
            best = judgment

    return best


def _headline_key(headline: Headline) -> str:
    """Stable short key for fixtures and logs."""
    import hashlib
    return hashlib.sha1(headline.title.encode("utf-8")).hexdigest()[:10]
