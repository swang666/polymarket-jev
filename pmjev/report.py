"""Console and Markdown output.

ASCII only: Windows consoles still default to a codepage that will raise
UnicodeEncodeError on box-drawing characters, and a crash in the reporting layer
would throw away a scan that already cost real tokens.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from .models import RunStats, Signal
from .scoring import AVOID, NO_VIEW, TRADE, WATCH

GATE_LABEL = {
    TRADE: "TRADE",
    WATCH: "watch",
    AVOID: "AVOID",
    NO_VIEW: "-",
}


def _fmt(value: Optional[float], spec: str = "{0:.3f}") -> str:
    return "-" if value is None else spec.format(value)


def _truncate(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "."


def console_report(signals: Sequence[Signal], stats: RunStats,
                   price_per_billion: float, top_n: int = 15,
                   show_no_view: bool = False) -> str:
    rows = [s for s in signals if show_no_view or s.gate != NO_VIEW][:top_n]
    lines: List[str] = []
    lines.append("=" * 100)
    lines.append(" Polymarket resolution-lag scan -- alert only, NOTHING here places an order")
    lines.append("=" * 100)

    if not rows:
        lines.append("")
        lines.append(" No market had evidence decisive enough to take a view.")
        lines.append(" That is the expected result most of the time: the strategy only")
        lines.append(" fires when a published fact has already settled a market's rules.")
    else:
        header = "{0:<6} {1:<4} {2:>6} {3:>6} {4:>7} {5:>7} {6}".format(
            "GATE", "SIDE", "P_MOD", "PRICE", "EDGE", "STAKE", "MARKET")
        lines.append("")
        lines.append(header)
        lines.append("-" * 100)
        for signal in rows:
            lines.append("{0:<6} {1:<4} {2:>6} {3:>6} {4:>7} {5:>7} {6}".format(
                GATE_LABEL.get(signal.gate, signal.gate),
                signal.side if signal.side != "NONE" else "-",
                _fmt(signal.p_model, "{0:.2f}"),
                _fmt(signal.market_price, "{0:.2f}"),
                _fmt(signal.edge, "{0:+.3f}"),
                "-" if signal.dollars is None else "${0:,.0f}".format(signal.dollars),
                _truncate(signal.market.question, 52),
            ))

        lines.append("")
        lines.append("-" * 100)
        for signal in rows:
            if signal.gate == NO_VIEW and not show_no_view:
                continue
            lines.append("")
            lines.append("[{0}]{1} {2}".format(
                GATE_LABEL.get(signal.gate, signal.gate),
                " (forecast)" if signal.mode == "forecast" else "",
                _truncate(signal.market.question, 76)))
            lines.append("    {0}".format(signal.market.url))
            if signal.judgment is not None:
                headline = signal.judgment.headline
                when = headline.published.strftime("%Y-%m-%d") if headline.published else "undated"
                lines.append("    evidence : {0}".format(_truncate(headline.title, 84)))
                lines.append("    source   : {0} ({1})  stance={2} @{3:.2f}".format(
                    headline.source or "unknown", when,
                    signal.judgment.stance or "?", signal.judgment.stance_confidence))
                lines.append(
                    "    judgment : met={0:.2f} precluded={1:.2f} "
                    "direct={2:.2f}/3 ambig={3:.2f}/3 auth={4:.2f}/3".format(
                        signal.judgment.condition_met,
                        signal.judgment.condition_precluded,
                        signal.judgment.directness,
                        signal.judgment.rule_ambiguity,
                        signal.judgment.source_authority))
            if signal.forecast is not None:
                f = signal.forecast
                lines.append("    forecast : P(yes)={0:.2f} class={1} @{2:.2f}".format(
                    f.resolves_yes, f.event_class or "?", f.event_class_confidence))
                lines.append("    inputs   : tilt={0:.2f}/4 sufficiency={1:.2f}/3 "
                             "ambig={2:.2f}/3 | {3}".format(
                                 f.evidence_tilt, f.evidence_sufficiency,
                                 f.rule_ambiguity, f.time_remaining))
                lines.append("    based on : {0} headline(s), top: {1}".format(
                    len(f.headlines),
                    _truncate(f.headlines[0].title, 60) if f.headlines else "none"))
            for reason in signal.reasons:
                lines.append("    why      : {0}".format(reason))

    lines.append("")
    lines.append("-" * 100)
    lines.append(
        " markets {0} fetched / {1} eligible | headlines {2} found / {3} screened in | "
        "deep reads {4}".format(
            stats.markets_fetched, stats.markets_after_filter,
            stats.headlines_found, stats.headlines_screened_in, stats.deep_judgments))
    lines.append(
        " jev requests {0} | input tokens {1:,} | cost ${2:.4f}".format(
            stats.requests_made, stats.input_tokens,
            stats.cost_usd(price_per_billion)))
    if stats.errors:
        lines.append(" errors:")
        for err in stats.errors[:10]:
            lines.append("   - {0}".format(err))
    lines.append("=" * 100)
    return "\n".join(lines)


def markdown_report(signals: Sequence[Signal], stats: RunStats,
                    price_per_billion: float, top_n: int = 15) -> str:
    rows = [s for s in signals if s.gate != NO_VIEW][:top_n]
    out: List[str] = []
    out.append("# Polymarket resolution-lag scan")
    out.append("")
    out.append("Alert only. No orders are placed by this tool.")
    out.append("")
    out.append("| Gate | Side | P(model) | Price | Edge | Stake | Market |")
    out.append("|---|---|---|---|---|---|---|")
    for signal in rows:
        out.append("| {0} | {1} | {2} | {3} | {4} | {5} | [{6}]({7}) |".format(
            signal.gate,
            signal.side if signal.side != "NONE" else "-",
            _fmt(signal.p_model, "{0:.2f}"),
            _fmt(signal.market_price, "{0:.2f}"),
            _fmt(signal.edge, "{0:+.3f}"),
            "-" if signal.dollars is None else "${0:,.0f}".format(signal.dollars),
            _truncate(signal.market.question.replace("|", "/"), 70),
            signal.market.url,
        ))
    out.append("")
    for signal in rows:
        out.append("## {0}".format(signal.market.question))
        out.append("")
        out.append("- Market: <{0}>".format(signal.market.url))
        if signal.judgment is not None:
            headline = signal.judgment.headline
            out.append("- Evidence: {0} ({1})".format(
                headline.title, headline.source or "unknown"))
            out.append("- Judgment: condition_met={0:.2f}, condition_precluded={1:.2f}, "
                       "directness={2:.2f}/3, rule_ambiguity={3:.2f}/3, "
                       "source_authority={4:.2f}/3".format(
                           signal.judgment.condition_met,
                           signal.judgment.condition_precluded,
                           signal.judgment.directness,
                           signal.judgment.rule_ambiguity,
                           signal.judgment.source_authority))
        for reason in signal.reasons:
            out.append("- {0}".format(reason))
        out.append("")
    out.append("---")
    out.append("")
    out.append("{0} Jev requests, {1:,} input tokens, ${2:.4f}.".format(
        stats.requests_made, stats.input_tokens, stats.cost_usd(price_per_billion)))
    return "\n".join(out)


def write_markdown(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
