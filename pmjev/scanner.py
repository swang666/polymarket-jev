"""The pipeline: discover -> gather evidence -> screen -> judge -> score -> log."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .config import Config
from .evidence import EvidenceProvider, build_provider
from .judge import best_judgment
from .models import Market, RunStats, Signal, utcnow
from .polymarket import ClobClient, GammaClient, discover
from .questions import deep_questions, deep_state
from .scoring import evaluate, rank
from .store import JsonlStore, new_run_id
from .typesafe import BudgetExceeded, StubClient, TypeSafeError

log = logging.getLogger(__name__)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def scan(cfg: Config, client, provider: EvidenceProvider,
         markets: Sequence[Market], store: Optional[JsonlStore] = None,
         now: Optional[datetime] = None,
         run_id: Optional[str] = None) -> Tuple[List[Signal], RunStats]:
    """Judge and score a set of markets that have already been discovered."""
    now = now or utcnow()
    run_id = run_id or new_run_id(now)
    stats = RunStats(markets_after_filter=len(markets))
    signals: List[Signal] = []

    for market in markets:
        try:
            headlines = provider.search(market, cfg.evidence)
        except Exception as exc:  # a bad feed must not kill the scan
            stats.errors.append("evidence {0}: {1}".format(market.slug, exc))
            headlines = []
        stats.headlines_found += len(headlines)

        judgment = None
        if headlines:
            try:
                judgment = best_judgment(client, market, headlines, cfg, stats)
            except BudgetExceeded as exc:
                stats.errors.append(str(exc))
                log.warning("stopping early: %s", exc)
                break
            except TypeSafeError as exc:
                stats.errors.append("jev {0}: {1}".format(market.slug, exc))

        signal = evaluate(market, judgment, cfg.judgment, cfg.sizing, now=now)
        signals.append(signal)

        if store is not None:
            state = questions = None
            if judgment is not None:
                # Record exactly what the model saw, so a wrong call can be
                # re-read later instead of reconstructed from memory.
                state = deep_state(
                    market.question, market.description, judgment.headline,
                    rules_chars=cfg.evidence.rules_chars,
                    snippet_chars=cfg.evidence.snippet_chars)
                questions = deep_questions()
            store.append_signal(signal, run_id, state=state, questions=questions)

    ranked = rank(signals)
    if cfg.sizing.max_signals > 0:
        # Cap concurrent exposure. Demoting rather than dropping keeps the
        # market visible in the report and in the log for the backtest.
        actionable = [s for s in ranked if s.gate == "TRADE"]
        for extra in actionable[cfg.sizing.max_signals:]:
            extra.gate = "WATCH"
            extra.reasons.append(
                "beyond the {0} concurrent-position cap".format(cfg.sizing.max_signals))
        ranked = rank(ranked)
    return ranked, stats


def run_dry(cfg: Config, now: Optional[datetime] = None
            ) -> Tuple[List[Market], List[dict]]:
    """Discover markets and build the requests, but send nothing to Jev.

    Use this to see what a scan would cost and read the exact prompts before
    spending anything.
    """
    now = now or utcnow()
    markets = discover(cfg.discovery, gamma=GammaClient(), clob=ClobClient(), now=now)
    provider = build_provider(cfg.evidence, root=cfg.root)

    previews: List[dict] = []
    for market in markets:
        headlines = provider.search(market, cfg.evidence)
        if not headlines:
            continue
        from .questions import screen_questions, screen_state
        state = screen_state(market.question, market.description, headlines,
                             rules_chars=cfg.evidence.rules_chars)
        questions = screen_questions(headlines)
        approx = len(json.dumps({"state": state, "questions": questions})) // 4
        previews.append({
            "market": market.question,
            "slug": market.slug,
            "headlines": [h.title for h in headlines],
            "screen_state": state,
            "screen_questions": questions,
            "approx_input_tokens": approx,
        })
    return markets, previews


def run_demo(cfg: Config, now: Optional[datetime] = None
             ) -> Tuple[List[Signal], RunStats]:
    """Fully offline walkthrough: recorded markets, recorded Jev answers.

    No API key, no network. The four fixture markets are chosen to show the
    four outcomes the gate can produce, including the two that decline to trade.
    """
    markets_path = FIXTURES / "markets.json"
    script_path = FIXTURES / "jev_script.json"
    evidence_path = FIXTURES / "evidence.json"

    rows = json.loads(markets_path.read_text(encoding="utf-8"))
    markets = [Market.from_gamma(row) for row in rows]

    script = json.loads(script_path.read_text(encoding="utf-8"))
    client = StubClient(script)

    demo_cfg = cfg
    demo_cfg.evidence.provider = "file"
    demo_cfg.evidence.evidence_file = str(evidence_path)
    provider = build_provider(demo_cfg.evidence, root=cfg.root)

    # The fixtures were recorded at a fixed instant; pin "now" to it so the
    # staleness rules behave the same way on every run.
    fixed_now = now or datetime.fromisoformat(
        json.loads((FIXTURES / "meta.json").read_text(encoding="utf-8"))["recorded_at"]
    )

    signals, stats = scan(demo_cfg, client, provider, markets, store=None,
                          now=fixed_now)
    stats.markets_fetched = len(markets)
    return signals, stats
