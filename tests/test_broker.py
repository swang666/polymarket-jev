"""Tests for position keeping and the portfolio limits.

The limits are the part that matters once this runs on a timer. Per-trade
sizing is already capped, but a scheduled scanner can open twenty correlated
positions in one morning, each individually within its own cap.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmjev.broker import (OPEN, SETTLED, PaperBroker, PortfolioLimits,
                          Position, position_id)
from pmjev.live import LiveBroker, LiveExecutionNotEnabled
from pmjev.models import Market, Signal

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)


def make_signal(market_id="1", side="YES", price=0.40, dollars=50.0,
                gate="TRADE", mode="resolution_lag", edge=0.15,
                accepting=True, question=None):
    market = Market.from_gamma({
        "id": market_id, "slug": "s" + market_id,
        "question": question or "Will market {0} resolve YES?".format(market_id),
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.40", "0.60"]',
        "clobTokenIds": '["t1", "t2"]',
        "acceptingOrders": accepting,
    })
    return Signal(market=market, judgment=None, side=side, p_model=0.60,
                  market_price=price, edge=edge, kelly_stake=0.05,
                  dollars=dollars, gate=gate, mode=mode)


def broker(tmp_path, bankroll=1000.0, **limit_kwargs):
    limits = PortfolioLimits(**limit_kwargs) if limit_kwargs else PortfolioLimits()
    return PaperBroker(tmp_path / "positions.jsonl", bankroll, limits)


# --- entry ----------------------------------------------------------------

def test_opening_records_contracts_from_stake_and_price(tmp_path):
    b = broker(tmp_path)
    opened, skipped = b.execute([make_signal(price=0.40, dollars=50.0)],
                                run_id="r1", now=NOW)
    assert len(opened) == 1
    position = opened[0]
    assert position.entry_price == pytest.approx(0.40)
    assert position.stake == pytest.approx(50.0)
    assert position.contracts == pytest.approx(125.0)   # 50 / 0.40
    assert position.status == OPEN
    assert position.venue == "paper"


def test_only_trade_gated_signals_are_executed(tmp_path):
    b = broker(tmp_path)
    for gate in ("WATCH", "AVOID", "NO_VIEW"):
        opened, _ = b.execute([make_signal(gate=gate)], run_id="r", now=NOW)
        assert opened == []
    assert b.open_positions() == []


def test_forecast_mode_is_not_executable_by_default(tmp_path):
    """Its signals failed the coherence check the book passed."""
    b = PaperBroker(tmp_path / "p.jsonl", 1000.0,
                    PortfolioLimits(allow_modes=("resolution_lag",)))
    opened, skipped = b.execute([make_signal(mode="forecast")], run_id="r", now=NOW)
    assert opened == []
    assert "not enabled for execution" in skipped[0][1]


def test_the_same_market_is_never_doubled_up(tmp_path):
    b = broker(tmp_path)
    b.execute([make_signal(market_id="1")], run_id="r1", now=NOW)
    opened, skipped = b.execute([make_signal(market_id="1")], run_id="r2", now=NOW)
    assert opened == []
    assert "already holding" in skipped[0][1]


def test_the_open_position_cap_is_enforced_across_runs(tmp_path):
    b = broker(tmp_path, max_open_positions=2)
    b.execute([make_signal(market_id=str(i)) for i in range(2)],
              run_id="r1", now=NOW)
    opened, skipped = b.execute([make_signal(market_id="99")], run_id="r2", now=NOW)
    assert opened == []
    assert "position cap" in skipped[0][1]


def test_total_exposure_is_capped_even_when_each_trade_is_within_its_own_cap(tmp_path):
    # The reason this class exists: ten $50 positions are each fine and
    # together are half the bankroll.
    b = broker(tmp_path, bankroll=1000.0, max_open_positions=20,
               max_total_exposure_pct=0.15, max_per_market_pct=0.05)
    signals = [make_signal(market_id=str(i), dollars=50.0) for i in range(10)]
    opened, skipped = b.execute(signals, run_id="r1", now=NOW)
    assert len(opened) == 3                    # 3 x $50 = $150 = 15%
    assert b.exposure() == pytest.approx(150.0)
    assert any("ceiling" in reason for _, reason in skipped)


def test_an_oversized_signal_is_refused_not_silently_clipped(tmp_path):
    b = broker(tmp_path, bankroll=1000.0, max_per_market_pct=0.05)
    opened, skipped = b.execute([make_signal(dollars=200.0)], run_id="r", now=NOW)
    assert opened == []
    assert "per-market cap" in skipped[0][1]


def test_strongest_edge_gets_the_capacity(tmp_path):
    b = broker(tmp_path, max_open_positions=1)
    signals = [make_signal(market_id="weak", edge=0.06),
               make_signal(market_id="strong", edge=0.30)]
    opened, _ = b.execute(signals, run_id="r", now=NOW)
    assert [p.market_id for p in opened] == ["strong"]


def test_a_market_that_stopped_accepting_orders_is_skipped(tmp_path):
    b = broker(tmp_path)
    opened, skipped = b.execute([make_signal(accepting=False)], run_id="r", now=NOW)
    assert opened == []
    assert "not accepting orders" in skipped[0][1]


def test_the_daily_loss_cap_halts_new_entries(tmp_path):
    b = broker(tmp_path, bankroll=1000.0, daily_loss_cap_pct=0.10)
    # A loss of $150 today, against a $100 cap.
    lost = Position(position_id="old", market_id="9", slug="s9", question="q",
                    side="YES", entry_price=0.5, contracts=300.0, stake=150.0,
                    opened_at=NOW - timedelta(hours=3), status=SETTLED,
                    outcome_yes=0, settled_at=NOW - timedelta(hours=1), pnl=-150.0)
    b._append(lost, SETTLED)

    opened, skipped = b.execute([make_signal()], run_id="r", now=NOW)
    assert opened == []
    assert "daily loss cap" in skipped[0][1]


def test_yesterdays_losses_do_not_halt_today(tmp_path):
    b = broker(tmp_path, bankroll=1000.0, daily_loss_cap_pct=0.10)
    lost = Position(position_id="old", market_id="9", slug="s9", question="q",
                    side="YES", entry_price=0.5, contracts=300.0, stake=150.0,
                    opened_at=NOW - timedelta(days=3), status=SETTLED,
                    outcome_yes=0, settled_at=NOW - timedelta(days=2), pnl=-150.0)
    b._append(lost, SETTLED)
    opened, _ = b.execute([make_signal()], run_id="r", now=NOW)
    assert len(opened) == 1


# --- settlement -----------------------------------------------------------

class FakeGamma:
    def __init__(self, outcomes):
        self.outcomes = outcomes            # market_id -> "1" / "0" / None

    def get_market(self, market_id):
        prices = self.outcomes.get(market_id)
        if prices is None:
            return None
        row = {"id": market_id, "closed": prices is not False,
               "outcomePrices": prices if prices is not False else '["0.4","0.6"]'}
        return Market.from_gamma(row)


def test_a_winning_position_settles_for_the_rest_of_the_dollar(tmp_path):
    b = broker(tmp_path)
    b.execute([make_signal(market_id="1", side="YES", price=0.40, dollars=40.0)],
              run_id="r", now=NOW)
    settled = b.settle(FakeGamma({"1": '["1", "0"]'}), now=NOW)

    assert len(settled) == 1
    position = settled[0]
    assert position.won is True
    assert position.status == SETTLED
    # 100 contracts at $1 = $100, less the $40 staked.
    assert position.pnl == pytest.approx(60.0)
    assert b.open_positions() == []


def test_a_losing_position_settles_for_the_whole_stake(tmp_path):
    b = broker(tmp_path)
    b.execute([make_signal(market_id="1", side="YES", price=0.40, dollars=40.0)],
              run_id="r", now=NOW)
    settled = b.settle(FakeGamma({"1": '["0", "1"]'}), now=NOW)
    assert settled[0].won is False
    assert settled[0].pnl == pytest.approx(-40.0)


def test_the_no_side_wins_when_the_market_resolves_no(tmp_path):
    b = broker(tmp_path)
    # $50 is the default per-market cap on a $1,000 bankroll; asking for more
    # is refused rather than clipped, so the fixture has to stay inside it.
    b.execute([make_signal(market_id="1", side="NO", price=0.50, dollars=50.0)],
              run_id="r", now=NOW)
    settled = b.settle(FakeGamma({"1": '["0", "1"]'}), now=NOW)
    assert settled[0].won is True
    assert settled[0].pnl == pytest.approx(50.0)     # 100 contracts, $50 staked


def test_unresolved_markets_are_left_open(tmp_path):
    b = broker(tmp_path)
    b.execute([make_signal(market_id="1")], run_id="r", now=NOW)
    assert b.settle(FakeGamma({"1": False}), now=NOW) == []
    assert len(b.open_positions()) == 1


def test_settlement_survives_a_market_that_cannot_be_fetched(tmp_path):
    b = broker(tmp_path)
    b.execute([make_signal(market_id="1")], run_id="r", now=NOW)
    assert b.settle(FakeGamma({}), now=NOW) == []
    assert len(b.open_positions()) == 1


# --- state ----------------------------------------------------------------

def test_the_book_is_rebuilt_by_replaying_the_event_log(tmp_path):
    path = tmp_path / "positions.jsonl"
    first = PaperBroker(path, 1000.0)
    first.execute([make_signal(market_id="1")], run_id="r", now=NOW)
    first.settle(FakeGamma({"1": '["1", "0"]'}), now=NOW)

    # A fresh broker over the same log, as another device would see it.
    second = PaperBroker(path, 1000.0)
    assert len(second.positions()) == 1
    assert second.positions()[0].status == SETTLED
    assert second.open_positions() == []
    assert second.realised_pnl() == pytest.approx(75.0)   # 125 contracts, $50


def test_a_corrupt_line_does_not_destroy_the_book(tmp_path):
    b = broker(tmp_path)
    b.execute([make_signal(market_id="1")], run_id="r", now=NOW)
    with b.path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    assert len(b.open_positions()) == 1


def test_position_ids_are_stable_and_distinct():
    assert position_id("1", "YES", "r") == position_id("1", "YES", "r")
    assert position_id("1", "YES", "r") != position_id("1", "NO", "r")
    assert position_id("1", "YES", "r1") != position_id("1", "YES", "r2")


def test_mark_to_market_values_the_book_without_closing_it(tmp_path):
    b = broker(tmp_path)
    b.execute([make_signal(market_id="1", side="YES", price=0.40, dollars=40.0)],
              run_id="r", now=NOW)

    class Quoting:
        def get_market(self, market_id):
            return Market.from_gamma({"id": market_id, "bestBid": 0.59,
                                      "bestAsk": 0.61, "outcomes": '["Yes","No"]'})

    marks = b.mark_to_market(Quoting())
    assert marks["cost"] == pytest.approx(40.0)
    assert marks["value"] == pytest.approx(60.0)      # 100 contracts at 0.60
    assert marks["unrealised"] == pytest.approx(20.0)
    assert len(b.open_positions()) == 1                # nothing was closed


def test_summary_reports_the_state_a_scheduled_run_would_print(tmp_path):
    b = broker(tmp_path)
    b.execute([make_signal(market_id="1"), make_signal(market_id="2")],
              run_id="r", now=NOW)
    b.settle(FakeGamma({"1": '["1", "0"]'}), now=NOW)

    summary = b.summary(now=NOW)
    assert summary["open"] == 1
    assert summary["settled"] == 1
    assert summary["wins"] == 1
    assert summary["hit_rate"] == pytest.approx(1.0)
    assert summary["realised_pnl"] > 0


# --- the live gate --------------------------------------------------------

def test_live_execution_refuses_and_explains_why(tmp_path):
    live = LiveBroker(tmp_path / "positions.jsonl", 1000.0)
    with pytest.raises(LiveExecutionNotEnabled) as excinfo:
        live.execute([make_signal()], run_id="r", now=NOW)

    message = str(excinfo.value)
    assert "backtest" in message
    assert "--execute" in message


def test_the_live_broker_inherits_the_paper_limits(tmp_path):
    # When this is implemented, only the fill source should change.
    live = LiveBroker(tmp_path / "p.jsonl", 1000.0)
    assert isinstance(live.limits, PortfolioLimits)
    assert live.venue == "live"
    assert hasattr(live, "settle") and hasattr(live, "mark_to_market")
