"""Position keeping and execution.

Until now signals evaporated the moment the report printed. This closes the
loop: a `TRADE` signal becomes a position at the real effective price, the book
is marked to market on every run, and positions settle automatically when their
market resolves. That turns "the model said 0.34" into "the model said 0.34 and
here is what it cost", which is the only version of the claim worth anything.

Two brokers share one interface:

* `PaperBroker` records fills at the price the book actually showed. No money
  moves, and it is the default.
* `LiveBroker` (live.py) signs real orders. It refuses to run until the
  backtest shows an edge and the operator has opted in twice, because a
  scheduled task with a funded private key and no validated edge is not a
  trading system.

Storage is an append-only event log -- OPEN and SETTLE rows, replayed to
rebuild state. Same reasoning as the judgment log: it survives being written
from two machines, and `.gitattributes` merges it with the union driver.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import SizingConfig
from .models import Signal, parse_iso, utcnow

log = logging.getLogger(__name__)

OPEN = "OPEN"
SETTLED = "SETTLED"


@dataclass
class PortfolioLimits:
    """Caps that apply across positions, not to any single one.

    Per-trade sizing already happens in scoring.py. These exist because a
    scheduled scanner can otherwise open twenty correlated positions in one
    morning, each individually within its cap.
    """

    max_open_positions: int = 8
    max_total_exposure_pct: float = 0.30   # of bankroll, across all open positions
    max_per_market_pct: float = 0.05
    daily_loss_cap_pct: float = 0.10       # realised losses today; halts new entries
    allow_modes: Tuple[str, ...] = ("resolution_lag", "forecast")


@dataclass
class Position:
    position_id: str
    market_id: str
    slug: str
    question: str
    side: str                              # YES | NO
    entry_price: float                     # effective cost per contract
    contracts: float
    stake: float
    opened_at: datetime
    run_id: str = ""
    mode: str = "resolution_lag"
    p_model: Optional[float] = None
    edge_at_entry: Optional[float] = None
    status: str = OPEN
    outcome_yes: Optional[int] = None
    settled_at: Optional[datetime] = None
    pnl: Optional[float] = None
    venue: str = "paper"

    @property
    def won(self) -> Optional[bool]:
        if self.outcome_yes is None:
            return None
        return (self.outcome_yes == 1) if self.side == "YES" else (self.outcome_yes == 0)

    def settle_value(self, outcome_yes: int) -> float:
        """Contracts pay $1 if the side won, $0 otherwise."""
        won = (outcome_yes == 1) if self.side == "YES" else (outcome_yes == 0)
        return self.contracts if won else 0.0

    def mark(self, yes_price: float) -> float:
        """Current value of the position at a given YES price."""
        price = yes_price if self.side == "YES" else (1.0 - yes_price)
        return self.contracts * price

    def to_event(self, kind: str) -> Dict[str, Any]:
        row = asdict(self)
        row["opened_at"] = self.opened_at.isoformat()
        row["settled_at"] = self.settled_at.isoformat() if self.settled_at else None
        row["event"] = kind
        return row

    @classmethod
    def from_event(cls, data: Dict[str, Any]) -> "Position":
        payload = {k: v for k, v in data.items() if k != "event"}
        payload["opened_at"] = parse_iso(payload.get("opened_at")) or utcnow()
        payload["settled_at"] = parse_iso(payload.get("settled_at"))
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in allowed})


def position_id(market_id: str, side: str, run_id: str) -> str:
    raw = "{0}|{1}|{2}".format(market_id, side, run_id)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


class PaperBroker:
    """Records fills at the book's real price. No money moves."""

    venue = "paper"

    def __init__(self, path: Path, bankroll: float,
                 limits: Optional[PortfolioLimits] = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.bankroll = bankroll
        self.limits = limits or PortfolioLimits()

    # ---- state ----------------------------------------------------------

    def _events(self) -> Iterable[Dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        return rows

    def positions(self) -> List[Position]:
        """Replay the log. Later events for an id supersede earlier ones."""
        by_id: Dict[str, Position] = {}
        for row in self._events():
            position = Position.from_event(row)
            by_id[position.position_id] = position
        return list(by_id.values())

    def open_positions(self) -> List[Position]:
        return [p for p in self.positions() if p.status == OPEN]

    def exposure(self) -> float:
        return sum(p.stake for p in self.open_positions())

    def realised_pnl(self, since: Optional[datetime] = None) -> float:
        total = 0.0
        for position in self.positions():
            if position.status != SETTLED or position.pnl is None:
                continue
            if since and position.settled_at and position.settled_at < since:
                continue
            total += position.pnl
        return total

    def _append(self, position: Position, kind: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(position.to_event(kind), ensure_ascii=False))
            handle.write("\n")

    # ---- entry ----------------------------------------------------------

    def _rejection(self, signal: Signal, open_now: List[Position],
                   exposure: float, halted: bool) -> Optional[str]:
        """Why this signal must not become a position, or None to proceed."""
        if signal.gate != "TRADE":
            return "not a TRADE signal"
        if signal.mode not in self.limits.allow_modes:
            return "mode {0} not enabled for execution".format(signal.mode)
        if signal.market_price is None or signal.dollars is None:
            return "no price or size"
        if signal.dollars <= 0:
            return "size rounded to zero"
        if halted:
            return "daily loss cap reached; entries halted"
        if any(p.market_id == signal.market.id for p in open_now):
            return "already holding this market"
        if len(open_now) >= self.limits.max_open_positions:
            return "at the {0}-position cap".format(self.limits.max_open_positions)
        if not signal.market.accepting_orders:
            return "market is not accepting orders"

        cap = self.limits.max_per_market_pct * self.bankroll
        if signal.dollars > cap + 1e-9:
            return "size ${0:,.0f} exceeds the per-market cap ${1:,.0f}".format(
                signal.dollars, cap)

        ceiling = self.limits.max_total_exposure_pct * self.bankroll
        if exposure + signal.dollars > ceiling + 1e-9:
            return "would take exposure to ${0:,.0f}, over the ${1:,.0f} ceiling".format(
                exposure + signal.dollars, ceiling)
        return None

    def execute(self, signals: Sequence[Signal], run_id: str = "",
                now: Optional[datetime] = None
                ) -> Tuple[List[Position], List[Tuple[Signal, str]]]:
        """Open positions for every signal that clears the portfolio limits."""
        now = now or utcnow()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        halted = self.realised_pnl(since=midnight) <= \
            -abs(self.limits.daily_loss_cap_pct * self.bankroll)

        open_now = self.open_positions()
        exposure = sum(p.stake for p in open_now)

        opened: List[Position] = []
        skipped: List[Tuple[Signal, str]] = []

        # Best edge first, so the caps are spent on the strongest signals.
        ordered = sorted(signals, key=lambda s: -(s.edge or -9))
        for signal in ordered:
            reason = self._rejection(signal, open_now, exposure, halted)
            if reason:
                if signal.gate == "TRADE":
                    skipped.append((signal, reason))
                continue

            price = signal.market_price
            stake = signal.dollars
            position = Position(
                position_id=position_id(signal.market.id, signal.side, run_id or str(now)),
                market_id=signal.market.id,
                slug=signal.market.slug,
                question=signal.market.question,
                side=signal.side,
                entry_price=price,
                contracts=stake / price,
                stake=stake,
                opened_at=now,
                run_id=run_id,
                mode=signal.mode,
                p_model=signal.p_model,
                edge_at_entry=signal.edge,
                venue=self.venue,
            )
            self._append(position, OPEN)
            opened.append(position)
            open_now.append(position)
            exposure += stake

        return opened, skipped

    # ---- settlement -----------------------------------------------------

    def settle(self, gamma, now: Optional[datetime] = None) -> List[Position]:
        """Close out every open position whose market has resolved."""
        from .backtest import resolved_outcome

        now = now or utcnow()
        settled: List[Position] = []
        for position in self.open_positions():
            try:
                market = gamma.get_market(position.market_id)
            except Exception as exc:
                log.warning("could not fetch %s: %s", position.market_id, exc)
                continue
            if market is None:
                continue
            outcome = resolved_outcome(market.raw)
            if outcome is None:
                continue

            position.outcome_yes = outcome
            position.status = SETTLED
            position.settled_at = now
            position.pnl = position.settle_value(outcome) - position.stake
            self._append(position, SETTLED)
            settled.append(position)
        return settled

    def mark_to_market(self, gamma) -> Dict[str, Any]:
        """Value the open book at current prices, without closing anything."""
        value = 0.0
        cost = 0.0
        marks: List[Tuple[Position, Optional[float]]] = []
        for position in self.open_positions():
            cost += position.stake
            try:
                market = gamma.get_market(position.market_id)
            except Exception:
                market = None
            price = market.yes_price if market else None
            if price is None:
                value += position.stake        # no quote: hold at cost
                marks.append((position, None))
            else:
                value += position.mark(price)
                marks.append((position, price))
        return {"cost": cost, "value": value, "unrealised": value - cost,
                "marks": marks}

    def summary(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        now = now or utcnow()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        all_positions = self.positions()
        closed = [p for p in all_positions if p.status == SETTLED]
        wins = [p for p in closed if p.won]
        return {
            "venue": self.venue,
            "bankroll": self.bankroll,
            "open": len(self.open_positions()),
            "exposure": self.exposure(),
            "settled": len(closed),
            "wins": len(wins),
            "hit_rate": (len(wins) / len(closed)) if closed else None,
            "realised_pnl": self.realised_pnl(),
            "realised_today": self.realised_pnl(since=midnight),
        }


def limits_from_config(cfg: SizingConfig,
                       overrides: Optional[Dict[str, Any]] = None) -> PortfolioLimits:
    limits = PortfolioLimits(max_per_market_pct=cfg.per_trade_cap_pct,
                             max_open_positions=max(1, cfg.max_signals))
    for key, value in (overrides or {}).items():
        if hasattr(limits, key):
            setattr(limits, key, value)
    return limits
