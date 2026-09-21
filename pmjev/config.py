"""Configuration: YAML file, overlaid with environment variables.

Every knob that decides whether money moves lives here rather than in code, so a
threshold can be re-tuned from backtest results without touching the pipeline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TypeSafeConfig:
    api_key: str = ""
    base_url: str = "https://api.typesafe.ai/v1/systemone"
    model: str = "jev-latest"
    timeout: float = 30.0
    max_retries: int = 3
    # Jev pricing: $42 per billion input tokens, output tokens free.
    price_per_billion_tokens: float = 42.0
    # Hard stop for one scan. Raises BudgetExceeded rather than silently spending.
    budget_usd_per_run: float = 0.50


@dataclass
class DiscoveryConfig:
    """Which markets are even worth looking at."""

    max_markets: int = 40
    fetch_limit: int = 250          # how many Gamma rows to pull before filtering
    min_liquidity: float = 5000.0
    min_volume: float = 20000.0
    # A resolution-lag trade needs a market that is still genuinely uncertain.
    # Outside this band there is no room left to be wrong about.
    price_band_low: float = 0.05
    price_band_high: float = 0.95
    max_days_to_resolution: float = 120.0
    min_days_to_resolution: float = 0.0
    max_spread: float = 0.10
    tag_ids: List[int] = field(default_factory=list)
    include_slugs: List[str] = field(default_factory=list)
    exclude_keywords: List[str] = field(
        # Sports and crypto-price markets resolve off a scoreboard or a price
        # feed, not off news text. Jev has no edge there and dates/arithmetic
        # are its documented weak spot, so they are excluded by default.
        default_factory=lambda: [
            "vs.", " vs ", "will win the game", "point spread",
            "what price will", "above $", "below $",
        ]
    )


@dataclass
class EvidenceConfig:
    provider: str = "google_news"   # google_news | rss | file | none
    max_headlines: int = 12         # candidates per market handed to the screen
    max_deep_reads: int = 3         # how many survive into the expensive stage
    lookback_hours: float = 168.0   # one week
    screen_threshold: float = 0.55  # noul cut for "worth a deep read"
    snippet_chars: int = 1200
    rules_chars: int = 4000         # truncate rules; guards against context rot
    user_agent: str = "Mozilla/5.0 (compatible; pmjev/0.1)"
    rss_feeds: List[str] = field(default_factory=list)
    evidence_file: str = ""


@dataclass
class JudgmentConfig:
    """How typed answers become a probability. Pure policy -- all applied in code."""

    # Below this, the evidence does not settle anything and we take no view.
    decisive_threshold: float = 0.80
    # Contradictory answers (both "met" and "precluded" high) are a red flag.
    contradiction_threshold: float = 0.35
    # Haircut weights. Each input is normalised to 0..1 (1 = worst) and the
    # weighted sum is capped at max_haircut, then multiplied into p_model.
    weight_ambiguity: float = 0.45
    weight_authority: float = 0.35
    weight_directness: float = 0.20
    max_haircut: float = 0.30
    # A Score answer the model is unsure about gets pulled toward the bad end.
    low_confidence_floor: float = 0.50
    # Markets whose rules Jev reads as genuinely disputable are flagged AVOID.
    avoid_ambiguity_norm: float = 0.75
    stale_evidence_hours: float = 72.0


@dataclass
class SizingConfig:
    bankroll: float = 1000.0
    kelly_fraction: float = 0.25    # quarter Kelly
    per_trade_cap_pct: float = 0.05
    min_edge: float = 0.05          # must clear fees, spread and model error
    slippage: float = 0.01          # added to the ask we assume we pay
    fee_rate: float = 0.0           # Polymarket charges no maker/taker fee today
    max_signals: int = 10


@dataclass
class OutputConfig:
    jsonl_path: str = "data/judgments.jsonl"
    markdown_path: str = "data/report.md"
    top_n: int = 15
    show_no_view: bool = False


@dataclass
class Config:
    typesafe: TypeSafeConfig = field(default_factory=TypeSafeConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    judgment: JudgmentConfig = field(default_factory=JudgmentConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    def path(self, relative: str) -> Path:
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else self.root / candidate


def _apply(target: Any, values: Dict[str, Any], where: str) -> None:
    """Copy known keys onto a dataclass, loudly rejecting unknown ones.

    A silently ignored typo in config.yaml is a threshold that never took
    effect, which is the kind of bug you only notice after it costs you.
    """
    known = {f.name for f in fields(target)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(
                "unknown config key {0!r} in section {1!r}; valid keys: {2}".format(
                    key, where, ", ".join(sorted(known))
                )
            )
        setattr(target, key, value)


def load_env_file(path: Path) -> None:
    """Minimal .env reader -- KEY=value lines, no dependency, no shell syntax."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: Optional[str] = None) -> Config:
    """Load config.yaml (or JSON), then overlay environment variables."""
    cfg = Config()
    root = cfg.root

    load_env_file(root / ".env")

    candidates = [Path(path)] if path else [root / "config.yaml", root / "config.json"]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        text = candidate.read_text(encoding="utf-8")
        if candidate.suffix in (".yaml", ".yml"):
            try:
                import yaml  # noqa: WPS433 - optional dependency
            except ImportError as exc:  # pragma: no cover - depends on env
                raise RuntimeError(
                    "config.yaml found but PyYAML is not installed "
                    "(pip install PyYAML), or use config.json instead"
                ) from exc
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)

        for section, values in data.items():
            if not hasattr(cfg, section):
                raise ValueError("unknown config section {0!r}".format(section))
            target = getattr(cfg, section)
            if is_dataclass(target) and isinstance(values, dict):
                _apply(target, values, section)
            else:
                setattr(cfg, section, values)
        break

    # Environment always wins: it is where the secret belongs.
    env_key = os.environ.get("TYPESAFE_API_KEY")
    if env_key:
        cfg.typesafe.api_key = env_key
    env_model = os.environ.get("TYPESAFE_MODEL")
    if env_model:
        cfg.typesafe.model = env_model

    return cfg
