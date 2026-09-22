"""Tests for forecast mode.

Forecast mode asks the harder question -- how will this resolve -- against a
price that is already an aggregate answer to it. These tests pin the three
decisions that keep it from being a dressed-up prior: pooling in log-odds,
shrinking toward the base rate, and treating a big disagreement with the book
as a warning rather than a jackpot.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmjev.config import Config, ForecastConfig, SizingConfig
from pmjev.models import Forecast, Headline, Market
from pmjev.questions import (EVENT_CLASSES, FORECAST_KEYS, forecast_questions,
                             forecast_state)
from pmjev.scoring import (AVOID, NO_VIEW, TRADE, WATCH, evaluate_forecast,
                           forecast_probability, logit, sigmoid,
                           time_remaining_phrase)
from pmjev.typesafe import StubClient

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)


def make_market(**overrides) -> Market:
    row = {
        "id": "1", "slug": "s", "question": "Will X happen?",
        "description": "Resolves YES if X happens by the deadline.",
        "outcomes": '["Yes", "No"]', "outcomePrices": '["0.40", "0.60"]',
        "clobTokenIds": '["1", "2"]', "endDate": "2026-10-21T12:00:00Z",
        "volumeNum": 100000.0, "liquidityNum": 50000.0,
        "bestBid": 0.39, "bestAsk": 0.41, "spread": 0.02,
        "acceptingOrders": True, "closed": False,
    }
    row.update(overrides)
    return Market.from_gamma(row)


def make_forecast(**overrides) -> Forecast:
    defaults = dict(
        market_id="1",
        headlines=[Headline(title="Something happened", source="Reuters",
                            url="", published=NOW - timedelta(hours=6))],
        resolves_yes=0.60,
        event_class="contested_competitive",
        event_class_probs={"contested_competitive": 0.9},
        event_class_confidence=0.9,
        evidence_tilt=2.0,              # neutral on a 0..4 scale
        evidence_tilt_confidence=0.9,
        rule_ambiguity=0.0,
        rule_ambiguity_confidence=0.9,
        evidence_sufficiency=3.0,       # top of a 0..3 scale
        evidence_sufficiency_confidence=0.9,
        time_remaining="about 4 weeks left",
    )
    defaults.update(overrides)
    return Forecast(**defaults)


# --- question design ------------------------------------------------------

def test_forecast_state_never_contains_the_market_price():
    """The single most important property of this mode.

    If the model sees the price it anchors to it, and a forecast that echoes
    the price tells you nothing -- while silently invalidating the backtest,
    which exists to compare the two.
    """
    market = make_market()
    state = forecast_state(market.question, market.description,
                           [Headline("h", "Reuters", "", NOW)], "about 4 weeks left")
    blob = json.dumps(state).lower()
    for banned in ("price", "0.40", "0.41", "bestbid", "bestask", "odds", "cents"):
        assert banned not in blob, "price leaked into forecast state: {0}".format(banned)


def test_forecast_state_passes_time_as_a_phrase_not_dates():
    # Jev treats dates as text, not ordered quantities, so it never sees two.
    state = forecast_state("q", "rules", [], "about 3 weeks left")
    assert state["time_remaining"] == "about 3 weeks left"
    assert "2026" not in json.dumps(state)


def test_every_event_class_has_a_base_rate():
    cfg = ForecastConfig()
    for option in EVENT_CLASSES:
        assert option in cfg.base_rates, option


def test_base_rates_decrease_with_how_unusual_the_event_is():
    rates = ForecastConfig().base_rates
    ordered = [rates[c] for c in EVENT_CLASSES]
    assert ordered == sorted(ordered, reverse=True), ordered


def test_forecast_questions_match_their_keys():
    questions = forecast_questions()
    assert sorted(questions) == sorted(FORECAST_KEYS)
    assert questions["resolves_yes"]["type"] == "noul"
    assert questions["event_class"]["type"] == "choice"


def test_the_class_question_asks_for_a_kind_not_a_likelihood():
    # Code owns the number. The model only says what sort of event it is.
    text = json.dumps(forecast_questions()["event_class"]).lower()
    assert "classify the requirement itself" in text
    assert "not how likely" in text


# --- pooling --------------------------------------------------------------

def test_pooling_happens_in_log_odds_not_by_averaging():
    """Pooling must be the geometric mean of ODDS, not the mean of probabilities.

    Averaging probabilities treats 0.95-and-0.05 the same as 0.51-and-0.49,
    throwing away that the first pair is two strong opposite signals.
    """
    import math

    assert sigmoid(logit(0.42)) == pytest.approx(0.42)
    cfg = ForecastConfig(weight_direct=0.5, weight_base_rate=0.5, weight_tilt=0.0)
    p, base, _ = forecast_probability(
        make_forecast(resolves_yes=0.90, event_class="contested_competitive"), cfg)
    assert base == pytest.approx(0.35)

    odds = math.sqrt((0.90 / 0.10) * (0.35 / 0.65))
    assert p == pytest.approx(odds / (1.0 + odds), abs=1e-6)
    assert p != pytest.approx((0.90 + 0.35) / 2.0, abs=1e-3)


def test_a_confident_model_still_gets_pulled_by_a_low_base_rate():
    cfg = ForecastConfig()
    high = make_forecast(resolves_yes=0.90, event_class="requires_extraordinary_change")
    p, base, _ = forecast_probability(high, cfg)
    assert base == pytest.approx(0.03)
    assert p < high.resolves_yes, "base rate must drag a confident answer down"


def test_a_routine_event_is_lifted_by_its_base_rate():
    cfg = ForecastConfig()
    low = make_forecast(resolves_yes=0.60, event_class="scheduled_routine")
    p, _, _ = forecast_probability(low, cfg)
    assert p > low.resolves_yes


def test_evidence_tilt_moves_the_answer_in_the_right_direction():
    cfg = ForecastConfig()
    against, _, _ = forecast_probability(make_forecast(evidence_tilt=0.0), cfg)
    neutral, _, _ = forecast_probability(make_forecast(evidence_tilt=2.0), cfg)
    toward, _, _ = forecast_probability(make_forecast(evidence_tilt=4.0), cfg)
    assert against < neutral < toward


def test_thin_coverage_shrinks_toward_the_base_rate_not_toward_a_half():
    """With no information the honest answer is the base rate, not a coin flip."""
    cfg = ForecastConfig()
    rich = make_forecast(resolves_yes=0.90, event_class="requires_unusual_departure",
                         evidence_sufficiency=3.0)
    thin = make_forecast(resolves_yes=0.90, event_class="requires_unusual_departure",
                         evidence_sufficiency=0.0)
    p_rich, base, _ = forecast_probability(rich, cfg)
    p_thin, _, _ = forecast_probability(thin, cfg)
    assert abs(p_thin - base) < abs(p_rich - base)


def test_an_unknown_event_class_falls_back_without_crashing():
    p, base, reasons = forecast_probability(
        make_forecast(event_class="something_new"), ForecastConfig())
    assert base == pytest.approx(0.5)
    assert any("unrecognised" in r for r in reasons)
    assert 0.0 <= p <= 1.0


def test_extreme_inputs_stay_in_range():
    cfg = ForecastConfig()
    for direct in (0.0, 1.0):
        for tilt in (0.0, 4.0):
            p, _, _ = forecast_probability(
                make_forecast(resolves_yes=direct, evidence_tilt=tilt), cfg)
            assert 0.0 < p < 1.0


# --- gating ---------------------------------------------------------------

def test_no_evidence_is_refused_as_a_bare_prior():
    cfg = ForecastConfig()
    signal = evaluate_forecast(make_market(),
                               make_forecast(headlines=[], evidence_sufficiency=0.0),
                               cfg, SizingConfig(), now=NOW)
    assert signal.gate == NO_VIEW
    assert any("bare prior" in r for r in signal.reasons)


def test_bare_priors_can_be_enabled_deliberately_for_measurement():
    cfg = ForecastConfig(allow_without_evidence=True, min_edge=0.0)
    signal = evaluate_forecast(make_market(),
                               make_forecast(headlines=[], evidence_sufficiency=0.0),
                               cfg, SizingConfig(), now=NOW)
    assert signal.gate != NO_VIEW


def test_a_huge_disagreement_with_the_book_is_avoided_not_traded():
    """A 50-point gap on a liquid market is usually a missing fact, not a 10x."""
    cfg = ForecastConfig(max_disagreement=0.45)
    market = make_market(bestBid=0.04, bestAsk=0.05, outcomePrices='["0.045", "0.955"]')
    signal = evaluate_forecast(
        market, make_forecast(resolves_yes=0.97, event_class="scheduled_routine",
                              evidence_tilt=4.0),
        cfg, SizingConfig(), now=NOW)
    assert signal.gate == AVOID
    assert any("missing fact" in r for r in signal.reasons)


def test_disputable_rules_are_avoided():
    signal = evaluate_forecast(make_market(), make_forecast(rule_ambiguity=3.0),
                               ForecastConfig(), SizingConfig(), now=NOW)
    assert signal.gate == AVOID


def test_a_small_edge_is_watched_not_traded():
    cfg = ForecastConfig(min_edge=0.10)
    signal = evaluate_forecast(
        make_market(), make_forecast(resolves_yes=0.50, evidence_tilt=2.0),
        cfg, SizingConfig(), now=NOW)
    assert signal.gate in (WATCH, NO_VIEW)


def test_a_real_edge_within_the_disagreement_limit_trades():
    cfg = ForecastConfig(min_edge=0.10, max_disagreement=0.45)
    # Market at 0.55 against a forecast of ~0.80: a 0.25 gap, inside the limit.
    # Moving this market to 0.30 makes the gap 0.50 and the guard correctly
    # refuses it -- see test_a_huge_disagreement_with_the_book_is_avoided.
    market = make_market(bestBid=0.54, bestAsk=0.56, outcomePrices='["0.55", "0.45"]')
    signal = evaluate_forecast(
        market, make_forecast(resolves_yes=0.75, event_class="scheduled_routine",
                              evidence_tilt=4.0),
        cfg, SizingConfig(), now=NOW)
    assert signal.gate == TRADE
    assert signal.side == "YES"
    assert signal.edge >= 0.10
    assert signal.dollars > 0


def test_the_no_side_is_taken_when_the_forecast_favours_it():
    cfg = ForecastConfig(min_edge=0.10, max_disagreement=0.60)
    market = make_market(bestBid=0.69, bestAsk=0.71, outcomePrices='["0.70", "0.30"]')
    signal = evaluate_forecast(
        market, make_forecast(resolves_yes=0.10,
                              event_class="requires_extraordinary_change",
                              evidence_tilt=0.0),
        cfg, SizingConfig(), now=NOW)
    assert signal.side == "NO"
    assert signal.edge > 0


def test_forecast_signals_are_tagged_for_the_backtest():
    from pmjev.evidence import build_provider
    from pmjev.scanner import FIXTURES, scan

    cfg = Config()
    cfg.evidence.provider = "file"
    cfg.evidence.evidence_file = str(FIXTURES / "evidence.json")
    markets = [Market.from_gamma(r) for r in
               json.loads((FIXTURES / "markets.json").read_text(encoding="utf-8"))]

    signals, stats = scan(cfg, StubClient({}), build_provider(cfg.evidence),
                          markets, mode="forecast", now=NOW)
    assert all(s.mode == "forecast" for s in signals)
    # One request per market -- forecast mode has no screen stage.
    assert stats.requests_made == len(markets)
    assert all(s.to_dict()["mode"] == "forecast" for s in signals)


def test_the_two_modes_are_scored_separately_in_the_backtest(tmp_path):
    from pmjev.backtest import run_backtest
    from pmjev.store import JsonlStore

    store = JsonlStore(tmp_path / "log.jsonl")
    for mode, p in (("resolution_lag", 0.95), ("forecast", 0.55)):
        store.append({
            "kind": "signal", "mode": mode, "gate": "TRADE", "side": "YES",
            "p_model": p, "market_price": 0.60, "dollars": 50.0,
            "market": {"id": "111", "slug": "s", "question": "q", "yes_price": 0.60},
        })

    class FakeGamma:
        def get_market(self, market_id):
            return type("M", (), {"raw": {"id": "111", "closed": True,
                                          "outcomePrices": '["1", "0"]'}})()

    report = run_backtest(store, gamma=FakeGamma())
    assert len(report.resolved) == 2, "one row per (market, mode), not per market"
    assert report.modes() == ["forecast", "resolution_lag"]
    # The sharper correct call must score better.
    assert report.by_mode("resolution_lag").brier_model() < \
        report.by_mode("forecast").brier_model()
    assert "Mode comparison" in report.render()


def test_time_phrases_never_leak_a_raw_number_of_days_as_arithmetic():
    for days in (0.2, 3, 20, 100, 900, None, -5):
        phrase = time_remaining_phrase(days)
        assert isinstance(phrase, str) and phrase


# --- coherence ------------------------------------------------------------

def _grouped_signal(market_id, event_id, question, p_model, market_price,
                    neg_risk=True, gate="TRADE"):
    from pmjev.models import Signal
    market = Market.from_gamma({
        "id": market_id, "slug": "s" + market_id, "question": question,
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["{0}", "{1}"]'.format(market_price, 1 - market_price),
        "negRisk": neg_risk,
        "events": [{"id": event_id, "title": "2026 Midterms"}],
    })
    return Signal(market=market, judgment=None, side="YES", p_model=p_model,
                  market_price=market_price, edge=0.2, kelly_stake=0.05,
                  dollars=50.0, gate=gate, mode="forecast")


def test_mutually_exclusive_siblings_that_do_not_add_up_are_refused():
    """The real failure from the first live forecast scan.

    D-House scored 0.35 and R-House 0.34. Those are mutually exclusive and
    exhaustive: the book summed to 1.00, the model to 0.69. Jev's own docs say
    logically related questions are not guaranteed to cohere, so this has to be
    caught by code.
    """
    from pmjev.coherence import apply_coherence_guard

    signals = [
        _grouped_signal("1", "e1", "Will the Democratic Party control the House?", 0.35, 0.93),
        _grouped_signal("2", "e1", "Will the Republican Party control the House?", 0.34, 0.07),
    ]
    warnings = apply_coherence_guard(signals)

    assert warnings, "an incoherent group must be reported"
    assert all(s.gate == "AVOID" for s in signals)
    assert any("contradict each other" in r for s in signals for r in s.reasons)


def test_a_coherent_group_is_left_alone():
    from pmjev.coherence import apply_coherence_guard

    signals = [
        _grouped_signal("1", "e1", "Candidate A wins?", 0.60, 0.62),
        _grouped_signal("2", "e1", "Candidate B wins?", 0.35, 0.33),
    ]
    assert apply_coherence_guard(signals) == []
    assert all(s.gate == "TRADE" for s in signals)


def test_probabilities_summing_far_above_one_are_refused():
    from pmjev.coherence import apply_coherence_guard

    signals = [
        _grouped_signal(str(i), "e1", "Outcome {0}?".format(i), 0.50, 0.20)
        for i in range(4)
    ]  # sums to 2.00 on a mutually exclusive group
    assert apply_coherence_guard(signals)
    assert all(s.gate == "AVOID" for s in signals)


def test_unrelated_markets_are_not_grouped():
    from pmjev.coherence import apply_coherence_guard

    signals = [
        _grouped_signal("1", "e1", "Thing A?", 0.90, 0.10),
        _grouped_signal("2", "e2", "Thing B?", 0.90, 0.10),
    ]
    assert apply_coherence_guard(signals) == []


def test_non_exclusive_groups_may_sum_past_one():
    # Without negRisk these are not alternatives, so a high sum proves nothing.
    from pmjev.coherence import apply_coherence_guard

    signals = [
        _grouped_signal("1", "e1", "A?", 0.80, 0.80, neg_risk=False),
        _grouped_signal("2", "e1", "B?", 0.80, 0.80, neg_risk=False),
    ]
    assert apply_coherence_guard(signals) == []


def test_flat_forecasts_are_diagnosed_as_base_rate_collapse():
    """Every forecast landing on its class prior looks exactly like edge."""
    from pmjev.coherence import spread_diagnostic

    prices = [0.05, 0.20, 0.50, 0.80, 0.93, 0.07, 0.65]
    flat = [_grouped_signal(str(i), "e{0}".format(i), "q", 0.34, price)
            for i, price in enumerate(prices)]
    warning = spread_diagnostic(flat)
    assert warning is not None
    assert "base rates" in warning


def test_a_model_that_tracks_the_market_is_not_flagged():
    from pmjev.coherence import spread_diagnostic

    prices = [0.05, 0.20, 0.50, 0.80, 0.93, 0.07, 0.65]
    tracking = [_grouped_signal(str(i), "e{0}".format(i), "q", price + 0.03, price)
                for i, price in enumerate(prices)]
    assert spread_diagnostic(tracking) is None


def test_the_diagnostic_stays_quiet_on_a_small_sample():
    from pmjev.coherence import spread_diagnostic

    assert spread_diagnostic([_grouped_signal("1", "e1", "q", 0.3, 0.9)]) is None
