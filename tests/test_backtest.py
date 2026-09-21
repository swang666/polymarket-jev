"""Tests for the calibration harness.

This is the part that decides whether the strategy is real, so a bug here is
worse than a bug in the scanner: it would tell you to size up on nothing.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmjev.backtest import (BacktestReport, Resolved, _brier, _calibration,
                            resolved_outcome, run_backtest)
from pmjev.store import JsonlStore


# --- reading resolutions --------------------------------------------------

def test_resolved_outcome_reads_a_settled_yes():
    assert resolved_outcome({"closed": True, "outcomePrices": '["1", "0"]'}) == 1


def test_resolved_outcome_reads_a_settled_no():
    assert resolved_outcome({"closed": True, "outcomePrices": '["0", "1"]'}) == 0


def test_open_markets_have_no_outcome():
    assert resolved_outcome({"closed": False, "outcomePrices": '["0.4", "0.6"]'}) is None


def test_a_half_settled_market_is_excluded_rather_than_guessed():
    # Polymarket can resolve 50/50 on a disputed market. That is not a label.
    assert resolved_outcome({"closed": True, "outcomePrices": '["0.5", "0.5"]'}) is None


def test_missing_prices_are_excluded():
    assert resolved_outcome({"closed": True}) is None
    assert resolved_outcome({"closed": True, "outcomePrices": "[]"}) is None


# --- scoring --------------------------------------------------------------

def test_brier_is_mean_squared_error():
    assert _brier([(1.0, 1), (0.0, 0)]) == pytest.approx(0.0)
    assert _brier([(0.0, 1)]) == pytest.approx(1.0)
    assert _brier([(0.5, 1), (0.5, 0)]) == pytest.approx(0.25)


def test_brier_of_nothing_is_none():
    assert _brier([]) is None


def test_calibration_buckets_by_predicted_probability():
    pairs = [(0.05, 0), (0.15, 0), (0.85, 1), (0.95, 1), (1.0, 1)]
    buckets = _calibration(pairs, bins=5)
    assert len(buckets) == 5
    assert buckets[0][1] == 2          # 0.0-0.2 holds two predictions
    assert buckets[0][3] == pytest.approx(0.0)
    assert buckets[-1][1] == 3         # 0.8-1.0 holds three, including p == 1.0
    assert buckets[-1][3] == pytest.approx(1.0)


def make_resolved(**overrides) -> Resolved:
    defaults = dict(
        market_id="1", slug="s", question="Will X?", gate="TRADE", side="YES",
        p_model_yes=0.90, market_yes_at_scan=0.60, entry_cost=0.62,
        stake_dollars=50.0, outcome_yes=1,
    )
    defaults.update(overrides)
    return Resolved(**defaults)


def test_a_winning_yes_pays_out_the_rest_of_the_dollar():
    won = make_resolved(entry_cost=0.62, stake_dollars=62.0, outcome_yes=1)
    assert won.side_won is True
    assert won.profit_per_contract == pytest.approx(0.38)
    assert won.pnl_dollars == pytest.approx(38.0)   # 100 contracts


def test_a_losing_yes_loses_the_stake():
    lost = make_resolved(entry_cost=0.62, stake_dollars=62.0, outcome_yes=0)
    assert lost.side_won is False
    assert lost.pnl_dollars == pytest.approx(-62.0)


def test_the_no_side_wins_when_the_market_resolves_no():
    no_win = make_resolved(side="NO", entry_cost=0.40, stake_dollars=40.0,
                           outcome_yes=0)
    assert no_win.side_won is True
    assert no_win.pnl_dollars == pytest.approx(60.0)


def test_brier_gap_is_negative_when_the_model_beat_the_price():
    report = BacktestReport(resolved=[
        make_resolved(p_model_yes=0.95, market_yes_at_scan=0.60, outcome_yes=1),
        make_resolved(p_model_yes=0.05, market_yes_at_scan=0.40, outcome_yes=0),
    ])
    assert report.brier_model() < report.brier_market()
    assert report.brier_gap() < 0


def test_brier_gap_is_positive_when_the_price_already_knew():
    report = BacktestReport(resolved=[
        make_resolved(p_model_yes=0.55, market_yes_at_scan=0.95, outcome_yes=1),
        make_resolved(p_model_yes=0.45, market_yes_at_scan=0.05, outcome_yes=0),
    ])
    assert report.brier_gap() > 0


def test_hit_rate_counts_only_gated_trades():
    report = BacktestReport(resolved=[
        make_resolved(gate="TRADE", outcome_yes=1),
        make_resolved(gate="TRADE", outcome_yes=0),
        make_resolved(gate="WATCH", outcome_yes=0),   # never entered
    ])
    assert report.hit_rate() == pytest.approx(0.5)


def test_an_empty_report_renders_without_crashing():
    text = BacktestReport().render()
    assert "Nothing has resolved yet" in text


# --- end to end -----------------------------------------------------------

def test_run_backtest_joins_the_log_to_resolutions(tmp_path):
    store = JsonlStore(tmp_path / "judgments.jsonl")
    store.append({
        "kind": "signal", "gate": "TRADE", "side": "YES", "p_model": 0.92,
        "market_price": 0.70, "dollars": 50.0,
        "market": {"id": "111", "slug": "won", "question": "q", "yes_price": 0.68},
    })
    store.append({
        "kind": "signal", "gate": "TRADE", "side": "YES", "p_model": 0.88,
        "market_price": 0.50, "dollars": 50.0,
        "market": {"id": "222", "slug": "open", "question": "q", "yes_price": 0.50},
    })
    store.append({
        "kind": "signal", "gate": "NO_VIEW", "side": "NONE", "p_model": None,
        "market": {"id": "333", "slug": "skip", "question": "q", "yes_price": 0.5},
    })

    class FakeGamma:
        def get_market(self, market_id):
            rows = {
                "111": {"id": "111", "closed": True, "outcomePrices": '["1", "0"]'},
                "222": {"id": "222", "closed": False, "outcomePrices": '["0.5", "0.5"]'},
            }
            row = rows.get(market_id)
            if row is None:
                return None
            return type("M", (), {"raw": row})()

    report = run_backtest(store, gamma=FakeGamma())
    assert len(report.resolved) == 1
    assert report.pending == 1
    assert report.resolved[0].outcome_yes == 1
    assert report.resolved[0].side_won is True
    assert "Brier" in report.render()


def test_only_the_first_view_of_a_market_is_scored(tmp_path):
    # Re-scanning the same market every hour must not turn one call into
    # twenty data points and make the sample look stronger than it is.
    store = JsonlStore(tmp_path / "judgments.jsonl")
    for price in (0.70, 0.75, 0.80):
        store.append({
            "kind": "signal", "gate": "TRADE", "side": "YES", "p_model": 0.92,
            "market_price": price, "dollars": 50.0,
            "market": {"id": "111", "slug": "s", "question": "q", "yes_price": price},
        })

    class FakeGamma:
        def get_market(self, market_id):
            return type("M", (), {"raw": {"id": "111", "closed": True,
                                          "outcomePrices": '["1", "0"]'}})()

    report = run_backtest(store, gamma=FakeGamma())
    assert len(report.resolved) == 1
    assert report.resolved[0].entry_cost == pytest.approx(0.70)
