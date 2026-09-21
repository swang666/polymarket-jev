"""Turning typed judgments into a position. Pure functions, no I/O, no model.

Everything the model is bad at lives here: arithmetic, prices, deadlines, Kelly,
thresholds. Jev supplies the reading of the text; this module supplies the money
maths. Keeping the split clean is also what makes the strategy testable -- every
function below can be checked against a hand-worked example.

The central judgment call, stated plainly: a high `condition_met` does not mean
"this will probably happen". It means "the supplied article says it already did".
That is the only kind of claim this project trades on.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

from .config import JudgmentConfig, SizingConfig
from .models import (AMBIGUITY_LEVELS, AUTHORITY_LEVELS, DIRECTNESS_LEVELS,
                     Judgment, Market, Signal, utcnow)

# Gate values, ordered by how much they should draw your attention.
TRADE = "TRADE"
WATCH = "WATCH"
AVOID = "AVOID"
NO_VIEW = "NO_VIEW"


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalise_score(raw: float, levels: int) -> float:
    """Map a Score answer onto 0..1, where 1 is the top level.

    Score answers are probability-weighted positions on the level scale, so a
    4-level question returns roughly 0..3 and may land between levels.
    """
    if levels <= 1:
        return 0.0
    return clamp(raw / float(levels - 1))


def penalise_low_confidence(badness: float, confidence: float, floor: float) -> float:
    """Treat a rating the model is unsure of as closer to the worst case.

    Confidence measures how concentrated the distribution is, not whether the
    answer is right -- so it is not evidence of quality, only a reason to stop
    leaning on the number. At confidence >= floor nothing changes; at confidence
    0 the input is treated as fully bad.
    """
    if floor <= 0:
        return clamp(badness)
    shortfall = clamp((floor - confidence) / floor)
    return clamp(badness + (1.0 - badness) * shortfall)


def haircut(judgment: Judgment, cfg: JudgmentConfig) -> Tuple[float, List[str]]:
    """How much to shrink a decisive reading toward 50/50, and why.

    Returns (haircut in 0..max_haircut, human-readable reasons).
    """
    ambiguity_bad = normalise_score(judgment.rule_ambiguity, AMBIGUITY_LEVELS)
    authority_bad = 1.0 - normalise_score(judgment.source_authority, AUTHORITY_LEVELS)
    directness_bad = 1.0 - normalise_score(judgment.directness, DIRECTNESS_LEVELS)

    floor = cfg.low_confidence_floor
    ambiguity_bad = penalise_low_confidence(
        ambiguity_bad, judgment.rule_ambiguity_confidence, floor)
    authority_bad = penalise_low_confidence(
        authority_bad, judgment.source_authority_confidence, floor)
    directness_bad = penalise_low_confidence(
        directness_bad, judgment.directness_confidence, floor)

    weighted = (
        cfg.weight_ambiguity * ambiguity_bad
        + cfg.weight_authority * authority_bad
        + cfg.weight_directness * directness_bad
    )
    total_weight = cfg.weight_ambiguity + cfg.weight_authority + cfg.weight_directness
    if total_weight > 0:
        weighted /= total_weight

    reasons: List[str] = []
    if ambiguity_bad >= 0.5:
        reasons.append("rules are disputable (ambiguity {0:.2f})".format(ambiguity_bad))
    if authority_bad >= 0.5:
        reasons.append("weak source (authority {0:.2f})".format(1.0 - authority_bad))
    if directness_bad >= 0.5:
        reasons.append("indirect evidence (directness {0:.2f})".format(1.0 - directness_bad))

    return clamp(weighted, 0.0, 1.0) * cfg.max_haircut, reasons


def shrink(p: float, amount: float) -> float:
    """Pull a probability toward 0.5 by `amount` (0 = keep, 1 = no information)."""
    return 0.5 + (p - 0.5) * (1.0 - clamp(amount))


def model_probability(judgment: Judgment,
                      cfg: JudgmentConfig) -> Tuple[Optional[float], str, str, List[str]]:
    """Convert one judgment into P(Yes), a side, a gate and its reasons.

    Returns (p_yes, side, gate, reasons). p_yes is None when we have no view.

    The gate is the honest part of this function. Most markets are genuine
    forecasting questions, and on those the right answer is NO_VIEW: no article
    settles them, so the model has nothing to say that the price does not
    already know.
    """
    reasons: List[str] = []
    met = clamp(judgment.condition_met)
    precluded = clamp(judgment.condition_precluded)

    if judgment.stance == "not_relevant":
        return None, "NONE", NO_VIEW, ["evidence judged not relevant"]

    # Both directions asserted at once. Nouls are answered independently, so
    # P(met) and P(precluded) need not sum to 1 -- when both are high it means
    # the reading is incoherent, which is a reason to stay out, not to average.
    if min(met, precluded) >= cfg.contradiction_threshold:
        return None, "NONE", AVOID, [
            "contradictory reading (met {0:.2f} / precluded {1:.2f})".format(met, precluded)
        ]

    decisive = max(met, precluded)
    if decisive < cfg.decisive_threshold:
        return None, "NONE", NO_VIEW, [
            "evidence settles nothing (met {0:.2f}, precluded {1:.2f}); "
            "this is a forecasting question, not a resolution-lag one".format(met, precluded)
        ]

    cut, cut_reasons = haircut(judgment, cfg)
    reasons.extend(cut_reasons)

    ambiguity_bad = normalise_score(judgment.rule_ambiguity, AMBIGUITY_LEVELS)
    if ambiguity_bad >= cfg.avoid_ambiguity_norm:
        return None, "NONE", AVOID, reasons + [
            "resolution rules read as genuinely disputable; dispute risk dominates"
        ]

    if met >= precluded:
        side = "YES"
        p_yes = shrink(met, cut)
        reasons.insert(0, "evidence reports the YES condition already met ({0:.2f})".format(met))
    else:
        side = "NO"
        p_no = shrink(precluded, cut)
        p_yes = 1.0 - p_no
        reasons.insert(0, "evidence reports YES already precluded ({0:.2f})".format(precluded))

    if cut > 0:
        reasons.append("shrunk toward 0.50 by {0:.1%}".format(cut))

    return clamp(p_yes), side, WATCH, reasons


def effective_cost(market: Market, side: str, cfg: SizingConfig) -> Optional[float]:
    """What one contract actually costs us, including spread and slippage.

    Buying YES lifts the ask. Buying NO lifts the NO ask, which is 1 minus the
    YES bid -- that is where the spread is really paid, so it must not be
    approximated with the midpoint.
    """
    if side == "YES":
        base = market.best_ask
        if base is None:
            base = market.yes_price
    elif side == "NO":
        base = None if market.best_bid is None else 1.0 - market.best_bid
        if base is None and market.yes_price is not None:
            base = 1.0 - market.yes_price
    else:
        return None

    if base is None:
        return None
    cost = base * (1.0 + cfg.fee_rate) + cfg.slippage
    if cost <= 0.0 or cost >= 1.0:
        return None
    return cost


def edge(p_win: float, cost: float) -> float:
    """Expected profit per contract, in dollars. Contracts settle at 1 or 0."""
    return p_win - cost


def kelly_stake(p_win: float, cost: float) -> float:
    """Full-Kelly fraction of bankroll for a binary contract bought at `cost`.

    For a contract that pays 1 with probability p and costs q, net odds are
    (1-q)/q, and the Kelly criterion reduces to (p - q) / (1 - q).
    """
    if cost <= 0.0 or cost >= 1.0:
        return 0.0
    return clamp((p_win - cost) / (1.0 - cost))


def size_position(p_win: float, cost: float, cfg: SizingConfig) -> Tuple[float, float]:
    """Return (fraction of bankroll, dollars) after fractional Kelly and caps."""
    full = kelly_stake(p_win, cost)
    fraction = min(full * cfg.kelly_fraction, cfg.per_trade_cap_pct)
    fraction = max(fraction, 0.0)
    return fraction, fraction * cfg.bankroll


def evaluate(market: Market, judgment: Optional[Judgment],
             judgment_cfg: JudgmentConfig, sizing_cfg: SizingConfig,
             now: Optional[datetime] = None) -> Signal:
    """Score one market against one judgment, producing a gated Signal."""
    now = now or utcnow()

    if judgment is None:
        return Signal(
            market=market, judgment=None, side="NONE", p_model=None,
            market_price=None, edge=None, kelly_stake=None, dollars=None,
            gate=NO_VIEW, reasons=["no evidence survived the screen"], scanned_at=now,
        )

    p_yes, side, gate, reasons = model_probability(judgment, judgment_cfg)

    if p_yes is None or side == "NONE":
        return Signal(
            market=market, judgment=judgment, side="NONE", p_model=None,
            market_price=None, edge=None, kelly_stake=None, dollars=None,
            gate=gate, reasons=reasons, scanned_at=now,
        )

    p_win = p_yes if side == "YES" else 1.0 - p_yes
    cost = effective_cost(market, side, sizing_cfg)
    if cost is None:
        reasons.append("no usable book price")
        return Signal(
            market=market, judgment=judgment, side=side, p_model=p_yes,
            market_price=None, edge=None, kelly_stake=None, dollars=None,
            gate=WATCH, reasons=reasons, scanned_at=now,
        )

    signal_edge = edge(p_win, cost)
    fraction, dollars = size_position(p_win, cost, sizing_cfg)

    # Date arithmetic belongs here, not in a question: Jev treats dates as text.
    age = judgment.headline.age_hours(now)
    stale = age is not None and age > judgment_cfg.stale_evidence_hours

    if signal_edge >= sizing_cfg.min_edge and fraction > 0:
        if stale:
            gate = WATCH
            reasons.append(
                "evidence is {0:.0f}h old; the market has had time to price it".format(age)
            )
        else:
            gate = TRADE
    else:
        gate = WATCH
        reasons.append(
            "edge {0:+.3f} below the {1:.3f} minimum".format(signal_edge, sizing_cfg.min_edge)
        )

    days = market.days_to_resolution(now)
    if days is not None and days < 0:
        gate = AVOID
        reasons.append("listed end date has passed; resolution may be under way")

    return Signal(
        market=market, judgment=judgment, side=side, p_model=p_yes,
        market_price=cost, edge=signal_edge, kelly_stake=fraction,
        dollars=dollars, gate=gate, reasons=reasons, scanned_at=now,
    )


GATE_ORDER = {TRADE: 0, WATCH: 1, AVOID: 2, NO_VIEW: 3}


def rank(signals: List[Signal]) -> List[Signal]:
    """Best first: gate, then edge descending."""
    return sorted(
        signals,
        key=lambda s: (GATE_ORDER.get(s.gate, 9), -(s.edge if s.edge is not None else -9)),
    )
