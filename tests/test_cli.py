"""Tests for the command line surface.

Mostly guarding the contract that --json emits parseable JSON and nothing else,
since that is what any downstream automation would depend on.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from run import main


def test_demo_json_is_parseable(capsys):
    assert main(["--demo", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["signals"]) == 4
    assert payload["stats"]["requests_made"] == 9


def test_demo_json_carries_the_judgment_behind_each_signal(capsys):
    main(["--demo", "--json"])
    payload = json.loads(capsys.readouterr().out)
    traded = [s for s in payload["signals"] if s["gate"] == "TRADE"]
    assert len(traded) == 1
    judgment = traded[0]["judgment"]
    assert judgment["condition_met"] == pytest.approx(0.94)
    assert judgment["headline"]["source"] == "Reuters"


def test_demo_prints_a_table_by_default(capsys):
    assert main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert "GATE" in out and "TRADE" in out
    with pytest.raises(ValueError):
        json.loads(out)


def test_show_all_reveals_the_no_view_markets(capsys):
    main(["--demo"])
    without = capsys.readouterr().out
    main(["--demo", "--show-all"])
    with_all = capsys.readouterr().out
    assert "JD Vance" not in without
    assert "JD Vance" in with_all


def test_scan_without_a_key_fails_before_touching_the_network(capsys, monkeypatch,
                                                              tmp_path):
    # Discovery and the news fetch take about a minute. A missing key has to
    # fail in a second, not after all that work has already been done.
    # Set it empty rather than deleting it: a real .env sitting next to the
    # package would otherwise repopulate the variable and defeat the test.
    monkeypatch.setenv("TYPESAFE_API_KEY", "")
    config = tmp_path / "config.yaml"
    config.write_text("discovery:\n  max_markets: 5\n", encoding="utf-8")

    def refuse(*args, **kwargs):
        raise AssertionError("network was touched before the key was checked")

    monkeypatch.setattr("run.discover", refuse)
    monkeypatch.setattr("run.GammaClient", refuse)

    assert main(["--scan", "--config", str(config)]) == 2
    assert "TYPESAFE_API_KEY" in capsys.readouterr().err


def test_backtest_without_a_log_explains_itself(capsys, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "output:\n  jsonl_path: {0}\n".format(
            (tmp_path / "missing.jsonl").as_posix()),
        encoding="utf-8")
    assert main(["--backtest", "--config", str(config)]) == 1
    assert "run a scan first" in capsys.readouterr().out


def test_an_unknown_config_key_exits_two(capsys, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("sizing:\n  bankrol: 10\n", encoding="utf-8")
    assert main(["--demo", "--config", str(config)]) == 2
    assert "config error" in capsys.readouterr().err


def test_a_market_name_with_an_accent_does_not_kill_the_report(capsys, monkeypatch):
    """Regression: a live scan crashed on "Ballon d'Or" markets.

    Polymarket questions carry accents and em-dashes, and a Windows console on
    a legacy codepage raises UnicodeEncodeError when asked to print them --
    after every Jev request has already been billed.
    """
    import io

    from pmjev.models import RunStats, Signal
    from pmjev.polymarket import Market

    class LegacyConsole(io.StringIO):
        encoding = "cp1252"

        def write(self, text):                      # noqa: D102
            text.encode("cp1252")                   # raises on non-cp1252 input
            return super().write(text)

    market = Market.from_gamma({
        "id": "1", "slug": "ballon", "question": "Will Kylian Mbappé win the 2026 Ballon d'Or?",
        "outcomes": '["Yes", "No"]', "outcomePrices": '["0.08", "0.92"]',
    })
    signal = Signal(market=market, judgment=None, side="NONE", p_model=None,
                    market_price=None, edge=None, kelly_stake=None,
                    dollars=None, gate="NO_VIEW", reasons=["≥ nothing"])

    from pmjev.report import console_report
    text = console_report([signal], RunStats(), 42.0, show_no_view=True)

    import run as run_module
    monkeypatch.setattr(run_module.sys, "stdout", LegacyConsole())
    run_module.emit(text)                           # must not raise
    assert "Ballon" in run_module.sys.stdout.getvalue()
