"""Did the model actually beat the price? Replays the log against resolutions.

Run this before risking money, and keep running it. A typed answer guarantees
the shape of the output, not its truth, and Jev's calibration is measured on
judgment tasks -- not on this market, this evidence source, or these thresholds.
The only thing that settles it is logged judgments scored against what happened.

The number to watch is the Brier gap: the model's Brier score minus the market
price's Brier score on the same resolved markets. Negative means the model was
closer to the truth than the price was. Positive means the price already knew.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .polymarket import GammaClient
from .store import JsonlStore

log = logging.getLogger(__name__)


@dataclass
class Resolved:
    """One logged judgment whose market has since resolved."""

    market_id: str
    slug: str
    question: str
    gate: str
    side: str
    p_model_yes: float
    market_yes_at_scan: float
    entry_cost: Optional[float]
    stake_dollars: Optional[float]
    outcome_yes: int          # 1 if the market resolved YES
    mode: str = "resolution_lag"

    @property
    def side_won(self) -> bool:
        return (self.outcome_yes == 1) if self.side == "YES" else (self.outcome_yes == 0)

    @property
    def profit_per_contract(self) -> Optional[float]:
        if self.entry_cost is None:
            return None
        return (1.0 - self.entry_cost) if self.side_won else -self.entry_cost

    @property
    def pnl_dollars(self) -> Optional[float]:
        if self.entry_cost is None or self.stake_dollars is None or self.entry_cost <= 0:
            return None
        contracts = self.stake_dollars / self.entry_cost
        return contracts * (self.profit_per_contract or 0.0)


@dataclass
class BacktestReport:
    resolved: List[Resolved] = field(default_factory=list)
    pending: int = 0
    skipped: int = 0

    # ---- scoring ---------------------------------------------------------

    def brier_model(self) -> Optional[float]:
        return _brier([(r.p_model_yes, r.outcome_yes) for r in self.resolved])

    def brier_market(self) -> Optional[float]:
        return _brier([(r.market_yes_at_scan, r.outcome_yes) for r in self.resolved])

    def brier_gap(self) -> Optional[float]:
        model, market = self.brier_model(), self.brier_market()
        if model is None or market is None:
            return None
        return model - market

    def hit_rate(self) -> Optional[float]:
        traded = [r for r in self.resolved if r.gate == "TRADE"]
        if not traded:
            return None
        return sum(1 for r in traded if r.side_won) / float(len(traded))

    def total_pnl(self) -> float:
        return sum(r.pnl_dollars or 0.0 for r in self.resolved if r.gate == "TRADE")

    def staked(self) -> float:
        return sum(r.stake_dollars or 0.0 for r in self.resolved if r.gate == "TRADE")

    def calibration(self, bins: int = 5) -> List[Tuple[str, int, float, float]]:
        """(bucket label, n, mean predicted, observed frequency)."""
        return _calibration([(r.p_model_yes, r.outcome_yes) for r in self.resolved], bins)

    def by_mode(self, mode: str) -> "BacktestReport":
        """A sub-report covering only one mode, for head-to-head comparison."""
        return BacktestReport(resolved=[r for r in self.resolved if r.mode == mode])

    def modes(self) -> List[str]:
        return sorted({r.mode for r in self.resolved})

    def render_comparison(self) -> List[str]:
        """Score each mode against the price on the markets it actually saw.

        This is the only thing that settles which approach is worth running.
        Note the two modes are NOT necessarily scored on the same markets --
        resolution-lag mode declines most of them by design -- so compare each
        one against its own market benchmark, never against the other's.
        """
        modes = self.modes()
        if len(modes) < 2:
            return []
        out = ["", " Mode comparison (each vs the price on ITS OWN markets)",
               "   {0:<16} {1:>4} {2:>9} {3:>9} {4:>9}".format(
                   "mode", "n", "model", "market", "gap")]
        for mode in modes:
            sub = self.by_mode(mode)
            gap = sub.brier_gap()
            out.append("   {0:<16} {1:>4} {2:>9.4f} {3:>9.4f} {4:>+9.4f}".format(
                mode, len(sub.resolved), sub.brier_model() or 0.0,
                sub.brier_market() or 0.0, gap or 0.0))
        out.append("   Lower model Brier is better; a negative gap means that mode")
        out.append("   beat the price. Sample sizes differ -- read n before the gap.")
        return out

    def render(self) -> str:
        lines: List[str] = []
        lines.append("=" * 80)
        lines.append(" Backtest: logged judgments vs realised resolutions")
        lines.append("=" * 80)
        lines.append("")
        lines.append(" resolved {0} | still open {1} | unusable {2}".format(
            len(self.resolved), self.pending, self.skipped))

        if not self.resolved:
            lines.append("")
            lines.append(" Nothing has resolved yet. Keep scanning; come back when the")
            lines.append(" logged markets have settled. Do not size up before then.")
            lines.append("=" * 80)
            return "\n".join(lines)

        model, market, gap = self.brier_model(), self.brier_market(), self.brier_gap()
        lines.append("")
        lines.append(" Brier (lower is better)")
        lines.append("   model  : {0:.4f}".format(model or 0.0))
        lines.append("   market : {0:.4f}".format(market or 0.0))
        lines.append("   gap    : {0:+.4f}  ({1})".format(
            gap or 0.0,
            "model beat the price" if (gap or 0) < 0 else "the price already knew"))

        traded = [r for r in self.resolved if r.gate == "TRADE"]
        if traded:
            hit = self.hit_rate() or 0.0
            staked, pnl = self.staked(), self.total_pnl()
            lines.append("")
            lines.append(" Paper trades")
            lines.append("   count    : {0}".format(len(traded)))
            lines.append("   hit rate : {0:.1%}".format(hit))
            lines.append("   staked   : ${0:,.2f}".format(staked))
            lines.append("   pnl      : ${0:,.2f}{1}".format(
                pnl, "  ({0:+.1%})".format(pnl / staked) if staked else ""))

        lines.append("")
        lines.append(" Calibration of P(Yes)")
        lines.append("   {0:<12} {1:>4} {2:>10} {3:>10}".format(
            "bucket", "n", "predicted", "observed"))
        for label, count, predicted, observed in self.calibration():
            lines.append("   {0:<12} {1:>4} {2:>10.3f} {3:>10.3f}".format(
                label, count, predicted, observed))

        lines.extend(self.render_comparison())

        lines.append("")
        lines.append(" A gap that is not clearly negative across at least a few dozen")
        lines.append(" resolutions means this has not earned real money yet.")
        lines.append("=" * 80)
        return "\n".join(lines)


def _brier(pairs: Sequence[Tuple[float, int]]) -> Optional[float]:
    usable = [(p, o) for p, o in pairs if p is not None]
    if not usable:
        return None
    return sum((p - o) ** 2 for p, o in usable) / float(len(usable))


def _calibration(pairs: Sequence[Tuple[float, int]], bins: int
                 ) -> List[Tuple[str, int, float, float]]:
    usable = [(p, o) for p, o in pairs if p is not None]
    width = 1.0 / bins
    out: List[Tuple[str, int, float, float]] = []
    for index in range(bins):
        low, high = index * width, (index + 1) * width
        bucket = [(p, o) for p, o in usable
                  if (low <= p < high or (index == bins - 1 and p == 1.0))]
        label = "{0:.1f}-{1:.1f}".format(low, high)
        if not bucket:
            out.append((label, 0, 0.0, 0.0))
            continue
        predicted = sum(p for p, _ in bucket) / len(bucket)
        observed = sum(o for _, o in bucket) / float(len(bucket))
        out.append((label, len(bucket), predicted, observed))
    return out


def resolved_outcome(market_row: Dict) -> Optional[int]:
    """Read a settled market's outcome from Gamma. None if not settled cleanly.

    A resolved binary market prints its outcome in `outcomePrices` as 1/0. Any
    other shape (still trading, or resolved to something in between) is not a
    clean label and is excluded rather than guessed at.
    """
    if not market_row.get("closed"):
        return None
    from .models import json_field
    prices = json_field(market_row.get("outcomePrices"), [])
    try:
        yes = float(prices[0])
    except (IndexError, TypeError, ValueError):
        return None
    if yes >= 0.99:
        return 1
    if yes <= 0.01:
        return 0
    return None


def run_backtest(store: JsonlStore, gamma: Optional[GammaClient] = None,
                 include_gates: Sequence[str] = ("TRADE", "WATCH")) -> BacktestReport:
    """Fetch the resolution of every logged market and score the judgments."""
    gamma = gamma or GammaClient()
    report = BacktestReport()

    latest: Dict[str, Dict] = {}
    for record in store.signals():
        if record.get("p_model") is None:
            continue
        if record.get("gate") not in include_gates:
            continue
        market = record.get("market") or {}
        market_id = str(market.get("id", ""))
        if not market_id:
            continue
        # One row per (market, mode): the first time we took a view is the
        # honest entry, and the two modes must not overwrite each other.
        key = "{0}:{1}".format(record.get("mode", "resolution_lag"), market_id)
        if key not in latest:
            latest[key] = record

    cache: Dict[str, Optional[Dict]] = {}
    for key, record in latest.items():
        market_id = str((record.get("market") or {}).get("id", ""))
        if market_id not in cache:
            try:
                fetched = gamma.get_market(market_id)
                cache[market_id] = fetched.raw if fetched else None
            except Exception as exc:
                log.warning("could not fetch market %s: %s", market_id, exc)
                cache[market_id] = None

        row = cache[market_id]
        if row is None:
            report.skipped += 1
            continue

        outcome = resolved_outcome(row)
        if outcome is None:
            report.pending += 1
            continue

        market = record.get("market") or {}
        yes_at_scan = market.get("yes_price")
        if yes_at_scan is None:
            report.skipped += 1
            continue

        report.resolved.append(Resolved(
            mode=record.get("mode", "resolution_lag"),
            market_id=market_id,
            slug=market.get("slug", ""),
            question=market.get("question", ""),
            gate=record.get("gate", ""),
            side=record.get("side", "NONE"),
            p_model_yes=float(record["p_model"]),
            market_yes_at_scan=float(yes_at_scan),
            entry_cost=record.get("market_price"),
            stake_dollars=record.get("dollars"),
            outcome_yes=outcome,
        ))

    return report
