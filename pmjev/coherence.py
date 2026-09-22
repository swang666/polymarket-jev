"""Cross-market coherence checks for forecast mode.

Jev's own jaggedness page says the model does not guarantee intuitive
relationships between logically related questions -- explicitly, that
`P(noul) != 1 - P(not noul)`. Each question is answered on its own.

For resolution-lag mode that barely matters: every question is about one
article and one rule, and the two Nouls that could conflict are already
cross-checked against each other. Forecast mode is far more exposed, because
Polymarket groups markets into events whose outcomes are mutually exclusive,
and nothing in a per-market request makes the answers add up.

The first live forecast scan showed exactly that:

    Democratic Party control the House   model 0.35   market 0.93
    Republican Party control the House   model 0.34   market 0.07

Those two are mutually exclusive and exhaustive. The market sums to 1.00; the
model sums to 0.69. Worse, the conjunction "R Senate AND R House" scored 0.36 --
higher than "R House" alone, which is impossible.

This module detects that automatically so it is caught by the tool rather than
by someone reading a table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .models import Market, Signal

log = logging.getLogger(__name__)

# A mutually exclusive group's probabilities may never sum far past 1.0.
SUM_CEILING = 1.10

# The low end cannot be a fixed floor, because Polymarket groups are often not
# exhaustive -- many carry an implicit "someone else" that is not listed, so a
# legitimate group can total well under 1. The book's own sum says how
# exhaustive the group is, which makes it the right reference: what matters is
# how far the model's total drifts from it.
#
# The case this is calibrated on: D-House 0.35 + R-House 0.34 = 0.69 on the
# model against 0.99 on the book. Two mutually exclusive, exhaustive outcomes,
# and the model lost 0.30 of probability mass between them.
MAX_GROUP_DIVERGENCE = 0.25


def event_key(market: Market) -> Optional[str]:
    """The Polymarket event a market belongs to, if any.

    Markets sharing an event are alternative answers to one question ("who wins
    the election"), so their YES probabilities are linked whether or not the
    model was told so.
    """
    events = market.raw.get("events") or []
    if isinstance(events, list) and events and isinstance(events[0], dict):
        key = events[0].get("id") or events[0].get("slug")
        return str(key) if key else None
    return None


@dataclass
class GroupCheck:
    """One event group's worth of coherence evidence."""

    key: str
    title: str
    signals: List[Signal] = field(default_factory=list)
    model_sum: float = 0.0
    market_sum: float = 0.0
    mutually_exclusive: bool = False

    @property
    def size(self) -> int:
        return len(self.signals)

    @property
    def divergence(self) -> float:
        """How far the model's total drifts from the book's for the same group."""
        return abs(self.model_sum - self.market_sum)

    @property
    def incoherent(self) -> bool:
        """True when the model's probabilities cannot all be right at once."""
        if not self.mutually_exclusive or self.size < 2:
            return False
        return (self.model_sum > SUM_CEILING
                or self.divergence > MAX_GROUP_DIVERGENCE)

    def describe(self) -> str:
        return (
            "{0} sibling markets sum to {1:.2f} on the model but {2:.2f} on the "
            "book (drift {3:.2f})".format(
                self.size, self.model_sum, self.market_sum, self.divergence)
        )


def group_signals(signals: Sequence[Signal]) -> Dict[str, GroupCheck]:
    """Bucket signals by Polymarket event and total their probabilities."""
    groups: Dict[str, GroupCheck] = {}
    for signal in signals:
        if signal.p_model is None:
            continue
        key = event_key(signal.market)
        if key is None:
            continue
        group = groups.get(key)
        if group is None:
            group = GroupCheck(key=key, title=signal.market.event_title or key)
            groups[key] = group
        group.signals.append(signal)
        group.model_sum += signal.p_model
        group.market_sum += signal.market.yes_price or 0.0
        # negRisk is Polymarket's own flag for "exactly one of these wins".
        if signal.market.neg_risk:
            group.mutually_exclusive = True
    return groups


def apply_coherence_guard(signals: Sequence[Signal]) -> List[str]:
    """Downgrade trades whose sibling markets contradict each other.

    Returns human-readable warnings. A market whose own event group does not
    add up has not been forecast -- it has been guessed at independently, and
    the sum proves it.
    """
    warnings: List[str] = []
    for group in group_signals(signals).values():
        if not group.incoherent:
            continue
        warnings.append("{0}: {1}".format(group.title[:60], group.describe()))
        for signal in group.signals:
            if signal.gate in ("TRADE", "WATCH"):
                signal.gate = "AVOID"
            signal.reasons.append(
                "sibling markets in this event sum to {0:.2f} on the model vs "
                "{1:.2f} on the book; these forecasts contradict each other"
                .format(group.model_sum, group.market_sum)
            )
    return warnings


def spread_diagnostic(signals: Sequence[Signal]) -> Optional[str]:
    """Warn when the model's probabilities are far flatter than the market's.

    Base-rate collapse looks exactly like edge: every forecast lands near its
    class prior, so every market priced away from that prior appears mispriced.
    The tell is variance -- a model that is genuinely reading each situation
    should spread out at least roughly as much as the book does.
    """
    usable = [s for s in signals
              if s.p_model is not None and s.market.yes_price is not None]
    if len(usable) < 5:
        return None

    model = [s.p_model for s in usable]
    market = [s.market.yes_price for s in usable]
    model_sd = _stdev(model)
    market_sd = _stdev(market)
    if market_sd <= 0:
        return None

    ratio = model_sd / market_sd
    if ratio >= 0.6:
        return None
    return (
        "forecast spread is {0:.0%} of the market's ({1:.3f} vs {2:.3f}) across "
        "{3} markets -- the probabilities are clustering near their base rates, "
        "so most of the apparent edge is the gap between the base rate and the "
        "price, not a read on the situation".format(
            ratio, model_sd, market_sd, len(usable))
    )


def _stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
