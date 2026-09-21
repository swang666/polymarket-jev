"""Tests for the money maths. These are the numbers that decide a position size,
so each one is checked against a value worked out by hand rather than recorded
from a run."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmjev.config import JudgmentConfig, SizingConfig
from pmjev.models import Headline, Judgment, Market
from pmjev.scoring import (AVOID, NO_VIEW, TRADE, WATCH, edge, effective_cost,
                           evaluate, haircut, kelly_stake, model_probability,
                           normalise_score, penalise_low_confidence, rank,
                           shrink, size_position)

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def make_market(**overrides) -> Market:
    defaults = dict(
        id="1", slug="a-market", question="Will X happen?",
        description="Resolves YES if X happens.", outcomes=["Yes", "No"],
        outcome_prices=[0.40, 0.60], clob_token_ids=["tokyes", "tokno"],
        end_date=NOW + timedelta(days=30), volume_num=100000.0,
        liquidity_num=50000.0, best_bid=0.39, best_ask=0.41, spread=0.02,
        last_trade_price=0.40, updated_at=NOW, accepting_orders=True,
        closed=False, neg_risk=False,
    )
    defaults.update(overrides)
    return Market(**defaults)


def make_judgment(**overrides) -> Judgment:
    defaults = dict(
        market_id="1",
        headline=Headline(title="X happened", source="Reuters",
                          url="https://example.invalid/x",
                          published=NOW - timedelta(hours=6)),
        condition_met=0.90, condition_precluded=0.02,
        stance="supports_yes", stance_probs={"supports_yes": 0.9},
        stance_confidence=0.9,
        directness=3.0, directness_confidence=0.9,
        rule_ambiguity=0.0, rule_ambiguity_confidence=0.9,
        source_authority=3.0, source_authority_confidence=0.9,
    )
    defaults.update(overrides)
    return Judgment(**defaults)


# --- primitives -----------------------------------------------------------

def test_normalise_score_maps_top_level_to_one():
    assert normalise_score(0.0, 4) == 0.0
    assert normalise_score(3.0, 4) == 1.0
    assert normalise_score(1.5, 4) == pytest.approx(0.5)


def test_normalise_score_clamps_out_of_range():
    assert normalise_score(9.0, 4) == 1.0
    assert normalise_score(-1.0, 4) == 0.0


def test_low_confidence_pushes_toward_worst_case():
    # At or above the floor the rating is taken as given.
    assert penalise_low_confidence(0.2, 0.9, 0.5) == pytest.approx(0.2)
    assert penalise_low_confidence(0.2, 0.5, 0.5) == pytest.approx(0.2)
    # At zero confidence the input is treated as fully bad.
    assert penalise_low_confidence(0.2, 0.0, 0.5) == pytest.approx(1.0)
    # Halfway to the floor, half the remaining distance is closed.
    assert penalise_low_confidence(0.2, 0.25, 0.5) == pytest.approx(0.6)


def test_shrink_pulls_toward_a_half():
    assert shrink(0.9, 0.0) == pytest.approx(0.9)
    assert shrink(0.9, 1.0) == pytest.approx(0.5)
    assert shrink(0.9, 0.5) == pytest.approx(0.7)
    assert shrink(0.1, 0.5) == pytest.approx(0.3)


# --- Kelly ----------------------------------------------------------------

def test_kelly_matches_the_closed_form():
    # For a contract paying 1 at cost q with win probability p, Kelly is
    # (p - q) / (1 - q).
    assert kelly_stake(0.60, 0.40) == pytest.approx(0.20 / 0.60)
    assert kelly_stake(0.94, 0.83) == pytest.approx(0.11 / 0.17)


def test_kelly_is_zero_without_an_edge():
    assert kelly_stake(0.40, 0.40) == 0.0
    assert kelly_stake(0.30, 0.40) == 0.0


def test_kelly_handles_degenerate_prices():
    assert kelly_stake(0.9, 0.0) == 0.0
    assert kelly_stake(0.9, 1.0) == 0.0


def test_size_position_applies_fraction_then_cap():
    cfg = SizingConfig(bankroll=1000.0, kelly_fraction=0.25, per_trade_cap_pct=0.05)
    # Full Kelly 0.3333 -> quarter Kelly 0.0833 -> capped at 0.05.
    fraction, dollars = size_position(0.60, 0.40, cfg)
    assert fraction == pytest.approx(0.05)
    assert dollars == pytest.approx(50.0)

    # A small edge stays below the cap and is not inflated by it.
    fraction, dollars = size_position(0.42, 0.40, cfg)
    assert fraction == pytest.approx((0.02 / 0.60) * 0.25)
    assert dollars == pytest.approx(fraction * 1000.0)


def test_edge_is_probability_minus_cost():
    assert edge(0.7, 0.5) == pytest.approx(0.2)
    assert edge(0.3, 0.5) == pytest.approx(-0.2)


# --- cost -----------------------------------------------------------------

def test_yes_cost_lifts_the_ask_and_adds_slippage():
    cfg = SizingConfig(slippage=0.01, fee_rate=0.0)
    market = make_market(best_bid=0.39, best_ask=0.41)
    assert effective_cost(market, "YES", cfg) == pytest.approx(0.42)


def test_no_cost_is_one_minus_the_bid_not_the_midpoint():
    # Buying NO crosses the spread on the other side. Using the midpoint here
    # would invent about half a spread of edge that does not exist.
    cfg = SizingConfig(slippage=0.01, fee_rate=0.0)
    market = make_market(best_bid=0.39, best_ask=0.41)
    assert effective_cost(market, "NO", cfg) == pytest.approx(0.62)


def test_fee_rate_is_applied_to_the_price():
    cfg = SizingConfig(slippage=0.0, fee_rate=0.02)
    market = make_market(best_bid=0.39, best_ask=0.50)
    assert effective_cost(market, "YES", cfg) == pytest.approx(0.51)


def test_cost_falls_back_to_the_listed_price_without_a_book():
    cfg = SizingConfig(slippage=0.01, fee_rate=0.0)
    market = make_market(best_bid=None, best_ask=None, outcome_prices=[0.30, 0.70])
    assert effective_cost(market, "YES", cfg) == pytest.approx(0.31)


def test_cost_is_none_when_it_would_exceed_a_dollar():
    cfg = SizingConfig(slippage=0.05, fee_rate=0.0)
    market = make_market(best_bid=0.97, best_ask=0.99)
    assert effective_cost(market, "YES", cfg) is None


# --- the gate -------------------------------------------------------------

def test_indecisive_evidence_takes_no_view():
    cfg = JudgmentConfig()
    judgment = make_judgment(condition_met=0.55, condition_precluded=0.10)
    p, side, gate, reasons = model_probability(judgment, cfg)
    assert p is None and side == "NONE" and gate == NO_VIEW
    assert "settles nothing" in reasons[0]


def test_contradictory_nouls_are_avoided_not_averaged():
    # Nouls are answered independently, so P(met) and P(precluded) need not sum
    # to one. Both high means the reading is incoherent.
    cfg = JudgmentConfig()
    judgment = make_judgment(condition_met=0.85, condition_precluded=0.60)
    p, side, gate, reasons = model_probability(judgment, cfg)
    assert p is None and gate == AVOID
    assert "contradictory" in reasons[0]


def test_irrelevant_evidence_takes_no_view():
    cfg = JudgmentConfig()
    judgment = make_judgment(stance="not_relevant", condition_met=0.95)
    p, side, gate, _ = model_probability(judgment, cfg)
    assert p is None and gate == NO_VIEW


def test_disputable_rules_are_avoided_even_when_decisive():
    cfg = JudgmentConfig()
    judgment = make_judgment(condition_met=0.95, rule_ambiguity=2.6)
    p, side, gate, reasons = model_probability(judgment, cfg)
    assert p is None and gate == AVOID
    assert any("disputable" in r for r in reasons)


def test_clean_yes_evidence_keeps_its_probability():
    cfg = JudgmentConfig()
    judgment = make_judgment(condition_met=0.94, directness=3.0,
                             rule_ambiguity=0.0, source_authority=3.0)
    p, side, gate, _ = model_probability(judgment, cfg)
    assert side == "YES"
    assert p == pytest.approx(0.94)   # perfect inputs mean no haircut
    assert gate == WATCH              # pricing has not happened yet


def test_precluded_evidence_flips_to_the_no_side():
    cfg = JudgmentConfig()
    judgment = make_judgment(condition_met=0.03, condition_precluded=0.92,
                             stance="supports_no", directness=3.0,
                             rule_ambiguity=0.0, source_authority=3.0)
    p, side, gate, _ = model_probability(judgment, cfg)
    assert side == "NO"
    assert p == pytest.approx(0.08)   # p_yes = 1 - p_no


def test_haircut_is_worst_at_the_worst_inputs():
    cfg = JudgmentConfig()
    worst = make_judgment(rule_ambiguity=3.0, source_authority=0.0, directness=0.0)
    amount, reasons = haircut(worst, cfg)
    assert amount == pytest.approx(cfg.max_haircut)
    assert len(reasons) == 3

    best = make_judgment(rule_ambiguity=0.0, source_authority=3.0, directness=3.0)
    amount, reasons = haircut(best, cfg)
    assert amount == pytest.approx(0.0)
    assert reasons == []


# --- end to end -----------------------------------------------------------

def test_evaluate_produces_a_trade_with_a_real_edge():
    judgment_cfg, sizing_cfg = JudgmentConfig(), SizingConfig()
    market = make_market(best_bid=0.81, best_ask=0.82)
    judgment = make_judgment(condition_met=0.94, directness=3.0,
                             rule_ambiguity=0.0, source_authority=3.0)
    signal = evaluate(market, judgment, judgment_cfg, sizing_cfg, now=NOW)

    assert signal.gate == TRADE
    assert signal.side == "YES"
    assert signal.market_price == pytest.approx(0.83)
    assert signal.edge == pytest.approx(0.11)
    assert signal.dollars == pytest.approx(50.0)   # quarter Kelly, hits the cap


def test_evaluate_downgrades_a_thin_edge_to_watch():
    judgment_cfg, sizing_cfg = JudgmentConfig(), SizingConfig()
    market = make_market(best_bid=0.90, best_ask=0.91)
    judgment = make_judgment(condition_met=0.94, directness=3.0,
                             rule_ambiguity=0.0, source_authority=3.0)
    signal = evaluate(market, judgment, judgment_cfg, sizing_cfg, now=NOW)
    assert signal.gate == WATCH
    assert signal.edge == pytest.approx(0.02)
    assert any("below the" in r for r in signal.reasons)


def test_stale_evidence_never_becomes_a_trade():
    # An old headline has had time to reach the order book, so the premise of
    # the whole strategy -- that the price has not caught up -- no longer holds.
    judgment_cfg, sizing_cfg = JudgmentConfig(), SizingConfig()
    market = make_market(best_bid=0.81, best_ask=0.82)
    old = Headline(title="X happened", source="Reuters", url="",
                   published=NOW - timedelta(hours=200))
    judgment = make_judgment(condition_met=0.94, headline=old, directness=3.0,
                             rule_ambiguity=0.0, source_authority=3.0)
    signal = evaluate(market, judgment, judgment_cfg, sizing_cfg, now=NOW)
    assert signal.gate == WATCH
    assert any("old" in r for r in signal.reasons)


def test_evaluate_without_a_judgment_is_no_view():
    signal = evaluate(make_market(), None, JudgmentConfig(), SizingConfig(), now=NOW)
    assert signal.gate == NO_VIEW
    assert signal.p_model is None


def test_expired_market_is_avoided():
    judgment_cfg, sizing_cfg = JudgmentConfig(), SizingConfig()
    market = make_market(best_bid=0.81, best_ask=0.82,
                         end_date=NOW - timedelta(days=1))
    judgment = make_judgment(condition_met=0.94, directness=3.0,
                             rule_ambiguity=0.0, source_authority=3.0)
    signal = evaluate(market, judgment, judgment_cfg, sizing_cfg, now=NOW)
    assert signal.gate == AVOID


def test_rank_puts_trades_first_then_biggest_edge():
    judgment_cfg, sizing_cfg = JudgmentConfig(), SizingConfig()
    strong = evaluate(make_market(id="strong", best_bid=0.60, best_ask=0.61),
                      make_judgment(condition_met=0.95, directness=3.0,
                                    rule_ambiguity=0.0, source_authority=3.0),
                      judgment_cfg, sizing_cfg, now=NOW)
    weak = evaluate(make_market(id="weak", best_bid=0.81, best_ask=0.82),
                    make_judgment(condition_met=0.94, directness=3.0,
                                  rule_ambiguity=0.0, source_authority=3.0),
                    judgment_cfg, sizing_cfg, now=NOW)
    none = evaluate(make_market(id="none"), None, judgment_cfg, sizing_cfg, now=NOW)

    ordered = rank([none, weak, strong])
    assert [s.market.id for s in ordered] == ["strong", "weak", "none"]
