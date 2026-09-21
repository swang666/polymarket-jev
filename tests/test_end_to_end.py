"""End-to-end tests over the recorded fixtures, plus the guards around spend.

The demo is the project's executable specification: four markets, four different
reasons to act or not act. If these assertions change, the strategy changed.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmjev.config import Config, load_config
from pmjev.scanner import run_demo
from pmjev.store import JsonlStore
from pmjev.typesafe import (BudgetExceeded, StubClient, TypeSafeClient,
                            TypeSafeError, UsageMeter)


def demo_signals():
    signals, stats = run_demo(Config())
    return {s.market.slug: s for s in signals}, stats


def test_demo_runs_offline_and_judges_every_market():
    signals, stats = demo_signals()
    assert len(signals) == 4
    assert stats.requests_made == 9        # 4 screens + 5 deep reads
    assert stats.errors == []


def test_demo_produces_one_trade_on_the_settled_election():
    signals, _ = demo_signals()
    trade = signals["will-united-russia-er-gain-the-most-seats-in-the-next-russian-parliamentary-election"]
    assert trade.gate == "TRADE"
    assert trade.side == "YES"
    assert trade.edge == pytest.approx(0.101, abs=0.002)
    assert trade.dollars == pytest.approx(50.0)


def test_demo_declines_to_trade_a_market_that_already_agrees():
    # The model reads the evidence as precluding YES, and so does the price.
    # Being right about the direction is not an edge.
    signals, _ = demo_signals()
    watch = signals["clarity-act-signed-into-law-in-2026"]
    assert watch.gate == "WATCH"
    assert watch.side == "NO"
    assert watch.edge < 0


def test_demo_avoids_the_market_with_disputable_rules():
    signals, _ = demo_signals()
    avoid = signals["will-the-iranian-regime-fall-by-the-end-of-2026"]
    assert avoid.gate == "AVOID"
    assert avoid.p_model is None
    assert any("disputable" in r for r in avoid.reasons)


def test_demo_takes_no_view_on_a_pure_forecasting_market():
    signals, _ = demo_signals()
    no_view = signals["will-jd-vance-win-the-2028-us-presidential-election"]
    assert no_view.gate == "NO_VIEW"
    assert no_view.p_model is None
    assert any("forecasting question" in r for r in no_view.reasons)


def test_demo_is_deterministic():
    first, _ = demo_signals()
    second, _ = demo_signals()
    assert {k: v.gate for k, v in first.items()} == {k: v.gate for k, v in second.items()}
    assert first["will-united-russia-er-gain-the-most-seats-in-the-next-russian-parliamentary-election"].edge == \
        second["will-united-russia-er-gain-the-most-seats-in-the-next-russian-parliamentary-election"].edge


def test_signals_serialise_to_json():
    signals, _ = demo_signals()
    blob = json.dumps([s.to_dict() for s in signals.values()])
    assert "condition_met" in blob


# --- spend and credential guards ------------------------------------------

def test_the_client_refuses_to_start_without_a_key():
    cfg = Config()
    cfg.typesafe.api_key = ""
    with pytest.raises(TypeSafeError) as excinfo:
        TypeSafeClient(cfg.typesafe)
    assert "TYPESAFE_API_KEY" in str(excinfo.value)


def test_the_budget_stops_a_runaway_scan():
    meter = UsageMeter(price_per_billion=42.0, budget_usd=0.001)
    meter.check()                       # nothing spent yet
    meter.record(30_000_000, 0)         # $1.26 of input tokens
    with pytest.raises(BudgetExceeded):
        meter.check()


def test_spend_is_computed_from_input_tokens_only():
    meter = UsageMeter(price_per_billion=42.0)
    meter.record(1_000_000_000, 500_000_000)
    assert meter.spent_usd == pytest.approx(42.0)


# --- config ---------------------------------------------------------------

def test_a_typo_in_config_is_rejected_rather_than_ignored(tmp_path):
    # A silently dropped key is a threshold that never took effect.
    path = tmp_path / "config.yaml"
    path.write_text("sizing:\n  kelly_fractoin: 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_config(str(path))
    assert "kelly_fractoin" in str(excinfo.value)


def test_an_unknown_section_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("trading:\n  live: true\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(path))


def test_the_example_config_is_valid(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(root, "config.example.yaml"))
    assert cfg.sizing.kelly_fraction == 0.25
    assert cfg.judgment.decisive_threshold == 0.80
    assert cfg.evidence.provider == "google_news"


def test_the_environment_key_wins_over_the_file(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-from-env")
    path = tmp_path / "config.yaml"
    path.write_text("typesafe:\n  api_key: sk-from-file\n", encoding="utf-8")
    assert load_config(str(path)).typesafe.api_key == "sk-from-env"


# --- store ----------------------------------------------------------------

def test_the_store_round_trips_and_survives_a_corrupt_line(tmp_path):
    store = JsonlStore(tmp_path / "log.jsonl")
    store.append({"kind": "signal", "gate": "TRADE"})
    with store.path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    store.append({"kind": "signal", "gate": "WATCH"})

    assert [r["gate"] for r in store.signals()] == ["TRADE", "WATCH"]


def test_scanning_logs_the_exact_state_the_model_saw(tmp_path):
    from pmjev.evidence import build_provider
    from pmjev.models import Market
    from pmjev.scanner import FIXTURES, scan

    cfg = Config()
    cfg.evidence.provider = "file"
    cfg.evidence.evidence_file = str(FIXTURES / "evidence.json")
    markets = [Market.from_gamma(row) for row in
               json.loads((FIXTURES / "markets.json").read_text(encoding="utf-8"))]
    script = json.loads((FIXTURES / "jev_script.json").read_text(encoding="utf-8"))

    store = JsonlStore(tmp_path / "log.jsonl")
    scan(cfg, StubClient(script), build_provider(cfg.evidence), markets,
         store=store, run_id="test-run")

    logged = store.signals()
    assert len(logged) == 4
    judged = [r for r in logged if r.get("state")]
    assert judged, "records with a judgment must carry the state that produced it"
    for record in judged:
        assert "resolution_rules" in record["state"]
        assert "evidence" in record["state"]
        assert "condition_met" in record["questions"]
        assert record["run_id"] == "test-run"
