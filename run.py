#!/usr/bin/env python3
"""Entrypoint.

  python run.py --demo                 Offline walkthrough. No API key, no network.
  python run.py --dry-run              Real markets + news, but send nothing to Jev.
  python run.py --scan                 Live scan (needs TYPESAFE_API_KEY).
  python run.py --scan --market SLUG   Scan one market by its Polymarket slug.
  python run.py --backtest             Score the logged judgments against outcomes.

This tool is alert-only. There is no wallet, no signing code and no order path in
this repository -- every result is for you to act on yourself, or not.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from pmjev.config import load_config
from pmjev.evidence import build_provider
from pmjev.models import RunStats
from pmjev.polymarket import ClobClient, GammaClient, discover, refresh_prices
from pmjev.report import console_report, markdown_report, write_markdown
from pmjev.scanner import run_demo, run_dry, scan
from pmjev.store import JsonlStore, new_run_id
from pmjev.typesafe import TypeSafeClient, TypeSafeError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Polymarket resolution-lag scanner powered by TypeSafe Jev.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true",
                      help="offline walkthrough with recorded fixtures")
    mode.add_argument("--scan", action="store_true",
                      help="live scan against Polymarket, news and Jev")
    mode.add_argument("--dry-run", action="store_true",
                      help="build the requests and price them, but call nothing")
    mode.add_argument("--backtest", action="store_true",
                      help="score logged judgments against realised resolutions")

    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--market", action="append", default=[],
                        help="restrict to this market slug (repeatable)")
    parser.add_argument("--limit", type=int, default=None,
                        help="override discovery.max_markets")
    parser.add_argument("--show-all", action="store_true",
                        help="include NO_VIEW markets in the console report")
    parser.add_argument("--json", action="store_true",
                        help="print machine-readable JSON instead of a table")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _make_stdout_unicode_safe() -> None:
    """Stop a market name from killing a run that has already been paid for.

    Windows consoles still default to a legacy codepage, and Polymarket
    questions routinely carry accents and dashes ("Mbappe" with an acute,
    em-dashes in rules text). Printing one of those raises UnicodeEncodeError
    *after* every Jev request has been billed, which is the worst possible
    moment to lose the report.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass    # older Python, or a stream that cannot be reconfigured


def emit(text: str) -> None:
    """Print, and if the terminal cannot encode it, degrade instead of dying.

    `_make_stdout_unicode_safe` handles this on any stream that supports
    reconfigure. This is the backstop for the ones that do not.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(encoding, errors="replace").decode(encoding))


def main(argv: Optional[List[str]] = None) -> int:
    _make_stdout_unicode_safe()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        cfg = load_config(args.config)
    except (ValueError, RuntimeError) as exc:
        print("config error: {0}".format(exc), file=sys.stderr)
        return 2

    if args.limit is not None:
        cfg.discovery.max_markets = args.limit
    if args.market:
        cfg.discovery.include_slugs = list(args.market)
        cfg.discovery.max_markets = max(cfg.discovery.max_markets, len(args.market))
    if args.show_all:
        cfg.output.show_no_view = True

    if args.backtest:
        return _backtest(cfg, args)
    if args.dry_run:
        return _dry_run(cfg, args)
    if args.scan:
        return _scan(cfg, args)

    return _demo(cfg, args)


def _demo(cfg, args) -> int:
    signals, stats = run_demo(cfg)
    emit(_render(signals, stats, cfg, args))
    if args.json:
        # --json is for piping; anything else on stdout would break the parse.
        return 0
    print()
    print("This was the offline demo: recorded markets, recorded Jev answers.")
    print("Set TYPESAFE_API_KEY and run 'python run.py --dry-run' next -- it uses")
    print("live markets and news but sends nothing, so you can read the prompts")
    print("and see the token cost before spending anything.")
    return 0


def _dry_run(cfg, args) -> int:
    markets, previews = run_dry(cfg)
    total = sum(p["approx_input_tokens"] for p in previews)
    cost = total * cfg.typesafe.price_per_billion_tokens / 1e9

    if args.json:
        print(json.dumps(previews, indent=2, ensure_ascii=False))
        return 0

    print("=" * 90)
    print(" DRY RUN -- nothing was sent to TypeSafe")
    print("=" * 90)
    print(" {0} markets passed discovery; {1} have evidence to screen".format(
        len(markets), len(previews)))
    print(" screen stage would cost about {0:,} input tokens (${1:.4f})".format(
        total, cost))
    print(" deep reads are additional: up to {0} per market that survives".format(
        cfg.evidence.max_deep_reads))
    print()
    for preview in previews[:10]:
        print("-" * 90)
        emit(" {0}".format(preview["market"]))
        print("   {0} headlines, ~{1:,} tokens".format(
            len(preview["headlines"]), preview["approx_input_tokens"]))
        for title in preview["headlines"][:5]:
            emit("     - {0}".format(title[:84]))
    print("-" * 90)
    print(" Re-run with --json to read the exact state and questions.")
    return 0


def _scan(cfg, args) -> int:
    now = None

    # Build the client before touching the network: a missing key should fail
    # in a second, not after a minute of market discovery and news fetching.
    try:
        client = TypeSafeClient(cfg.typesafe)
    except TypeSafeError as exc:
        print("error: {0}".format(exc), file=sys.stderr)
        return 2

    gamma, clob = GammaClient(), ClobClient()

    if cfg.discovery.include_slugs:
        markets = []
        for slug in cfg.discovery.include_slugs:
            market = gamma.get_market_by_slug(slug)
            if market is None:
                print("no market found for slug {0!r}".format(slug), file=sys.stderr)
            else:
                markets.append(market)
        refresh_prices(markets, clob)
    else:
        markets = discover(cfg.discovery, gamma=gamma, clob=clob)

    if not markets:
        print("no markets matched the discovery filters; loosen config.discovery")
        return 1

    provider = build_provider(cfg.evidence, root=cfg.root)
    store = JsonlStore(cfg.path(cfg.output.jsonl_path))
    run_id = new_run_id()

    signals, stats = scan(cfg, client, provider, markets, store=store,
                          now=now, run_id=run_id)
    stats.markets_fetched = len(markets)

    emit(_render(signals, stats, cfg, args))

    md_path = cfg.path(cfg.output.markdown_path)
    write_markdown(md_path, markdown_report(
        signals, stats, cfg.typesafe.price_per_billion_tokens, cfg.output.top_n))
    if args.json:
        return 0
    print()
    print("run id   : {0}".format(run_id))
    print("judgments: {0}".format(cfg.path(cfg.output.jsonl_path)))
    print("report   : {0}".format(md_path))
    return 0


def _backtest(cfg, args) -> int:
    from pmjev.backtest import run_backtest

    store = JsonlStore(cfg.path(cfg.output.jsonl_path))
    if not store.path.is_file():
        print("no judgment log at {0}; run a scan first".format(store.path))
        return 1

    report = run_backtest(store)
    if args.json:
        print(json.dumps({
            "resolved": len(report.resolved),
            "pending": report.pending,
            "skipped": report.skipped,
            "brier_model": report.brier_model(),
            "brier_market": report.brier_market(),
            "brier_gap": report.brier_gap(),
            "hit_rate": report.hit_rate(),
            "pnl": report.total_pnl(),
            "staked": report.staked(),
            "calibration": report.calibration(),
        }, indent=2))
        return 0
    emit(report.render())
    return 0


def _render(signals, stats: RunStats, cfg, args) -> str:
    if args.json:
        return json.dumps({
            "signals": [s.to_dict() for s in signals],
            "stats": stats.to_dict(),
        }, indent=2, ensure_ascii=False)
    return console_report(
        signals, stats, cfg.typesafe.price_per_billion_tokens,
        top_n=cfg.output.top_n, show_no_view=cfg.output.show_no_view,
    )


if __name__ == "__main__":
    sys.exit(main())
