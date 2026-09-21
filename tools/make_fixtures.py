#!/usr/bin/env python3
"""Regenerate the offline demo fixtures.

Market rows are real, recorded from the Gamma API. The Jev answers are written
by hand: the demo has to show all four gate outcomes, and waiting for the live
model to hand us one of each would make the walkthrough non-reproducible.

    python tools/make_fixtures.py            # rebuild from pmjev/fixtures/_raw.json
    python tools/make_fixtures.py --refetch  # re-record the market rows first
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "pmjev" / "fixtures"

RECORDED_AT = "2026-09-19T12:00:00+00:00"

SLUGS = [
    "will-united-russia-er-gain-the-most-seats-in-the-next-russian-parliamentary-election",
    "clarity-act-signed-into-law-in-2026",
    "will-the-iranian-regime-fall-by-the-end-of-2026",
    "will-jd-vance-win-the-2028-us-presidential-election",
]

# Evidence per market slug. Dates sit just behind RECORDED_AT so the staleness
# rule in scoring.evaluate has something real to act on.
EVIDENCE = {
    "will-united-russia-er-gain-the-most-seats-in-the-next-russian-parliamentary-election": [
        {
            "title": "Russia's Central Election Commission confirms United Russia took 315 of 450 State Duma seats",
            "source": "Reuters",
            "url": "https://example.invalid/cec-confirms",
            "published": "2026-09-18T09:20:00+00:00",
        },
        {
            "title": "United Russia celebrates Duma majority as opposition parties allege irregularities",
            "source": "Associated Press",
            "url": "https://example.invalid/duma-majority",
            "published": "2026-09-18T14:05:00+00:00",
        },
        {
            "title": "What the Russian parliamentary election means for Europe's energy policy",
            "source": "Politico Europe",
            "url": "https://example.invalid/energy-analysis",
            "published": "2026-09-17T06:00:00+00:00",
        },
    ],
    "clarity-act-signed-into-law-in-2026": [
        {
            "title": "Senate leader says the Digital Asset Market Clarity Act will not get floor time this Congress",
            "source": "Bloomberg",
            "url": "https://example.invalid/no-floor-time",
            "published": "2026-09-18T18:40:00+00:00",
        },
        {
            "title": "Crypto industry groups renew lobbying push for market structure legislation",
            "source": "CoinDesk",
            "url": "https://example.invalid/lobby-push",
            "published": "2026-09-16T11:00:00+00:00",
        },
    ],
    "will-the-iranian-regime-fall-by-the-end-of-2026": [
        {
            "title": "Opposition declares transitional council in Tehran as senior commanders defect",
            "source": "Iran International",
            "url": "https://example.invalid/transitional-council",
            "published": "2026-09-19T04:15:00+00:00",
        },
    ],
    "will-jd-vance-win-the-2028-us-presidential-election": [
        {
            "title": "New national poll shows Vance leading the 2028 Republican field by 18 points",
            "source": "The Hill",
            "url": "https://example.invalid/vance-poll",
            "published": "2026-09-18T13:00:00+00:00",
        },
        {
            "title": "Strategists debate whether early 2028 polling means anything at all",
            "source": "Axios",
            "url": "https://example.invalid/polling-debate",
            "published": "2026-09-17T08:30:00+00:00",
        },
    ],
}

# Screen probabilities, in the order the headlines appear above.
SCREEN = {
    "will-united-russia-er-gain-the-most-seats-in-the-next-russian-parliamentary-election": [0.96, 0.88, 0.31],
    "clarity-act-signed-into-law-in-2026": [0.91, 0.44],
    "will-the-iranian-regime-fall-by-the-end-of-2026": [0.93],
    "will-jd-vance-win-the-2028-us-presidential-election": [0.62, 0.24],
}

# Deep answers, keyed by (slug, headline index). Only headlines that clear the
# 0.55 screen threshold need an entry.
DEEP = {
    # TRADE: an official result, plainly reported, against a market still at 0.82.
    ("will-united-russia-er-gain-the-most-seats-in-the-next-russian-parliamentary-election", 0): {
        "condition_met": 0.94, "condition_precluded": 0.02,
        "stance": "supports_yes", "stance_conf": 0.90,
        "directness": 2.9, "directness_conf": 0.93,
        "ambiguity": 0.25, "ambiguity_conf": 0.88,
        "authority": 2.8, "authority_conf": 0.85,
    },
    # Same event, softer sourcing: kept, but less decisive, so it loses out.
    ("will-united-russia-er-gain-the-most-seats-in-the-next-russian-parliamentary-election", 1): {
        "condition_met": 0.71, "condition_precluded": 0.05,
        "stance": "supports_yes", "stance_conf": 0.74,
        "directness": 2.1, "directness_conf": 0.70,
        "ambiguity": 0.60, "ambiguity_conf": 0.72,
        "authority": 2.6, "authority_conf": 0.80,
    },
    # WATCH: decisive in direction, but the market already prices it.
    ("clarity-act-signed-into-law-in-2026", 0): {
        "condition_met": 0.04, "condition_precluded": 0.84,
        "stance": "supports_no", "stance_conf": 0.88,
        "directness": 2.4, "directness_conf": 0.82,
        "ambiguity": 1.20, "ambiguity_conf": 0.76,
        "authority": 2.5, "authority_conf": 0.86,
    },
    # AVOID: the event may have happened, but "ceases to govern" is disputable.
    ("will-the-iranian-regime-fall-by-the-end-of-2026", 0): {
        "condition_met": 0.81, "condition_precluded": 0.06,
        "stance": "supports_yes", "stance_conf": 0.69,
        "directness": 2.2, "directness_conf": 0.71,
        "ambiguity": 2.60, "ambiguity_conf": 0.64,
        "authority": 1.7, "authority_conf": 0.58,
    },
    # NO_VIEW: a poll is not a resolution. This is the common case.
    ("will-jd-vance-win-the-2028-us-presidential-election", 0): {
        "condition_met": 0.03, "condition_precluded": 0.02,
        "stance": "supports_yes", "stance_conf": 0.51,
        "directness": 0.9, "directness_conf": 0.80,
        "ambiguity": 0.30, "ambiguity_conf": 0.85,
        "authority": 2.2, "authority_conf": 0.83,
    },
}

STANCE_OPTIONS = ["supports_yes", "supports_no", "mixed", "not_relevant"]


def headline_key(title: str) -> str:
    return hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]


def noul(value: float) -> dict:
    return {"type": "noul", "noul": value}


def score(value: float, confidence: float, levels: int = 4) -> dict:
    return {
        "type": "score",
        "score": value,
        "legend": {str(i): "level {0}".format(i) for i in range(levels)},
        "probabilities": {},
        "confidence": confidence,
    }


def choice(pick: str, confidence: float) -> dict:
    rest = (1.0 - confidence) / (len(STANCE_OPTIONS) - 1)
    probs = {option: (confidence if option == pick else rest)
             for option in STANCE_OPTIONS}
    return {
        "type": "choice",
        "choice": pick,
        "probabilities": probs,
        "confidence": confidence,
    }


def refetch() -> list:
    import requests
    rows = []
    for slug in SLUGS:
        response = requests.get("https://gamma-api.polymarket.com/markets",
                                params={"slug": slug}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not payload:
            raise SystemExit("no market for slug {0}".format(slug))
        rows.append(payload[0])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refetch", action="store_true")
    args = parser.parse_args()

    FIXTURES.mkdir(parents=True, exist_ok=True)
    raw_path = FIXTURES / "_raw.json"

    if args.refetch or not raw_path.is_file():
        rows = refetch()
        raw_path.write_text(json.dumps(rows, indent=1, ensure_ascii=False),
                            encoding="utf-8")
    else:
        rows = json.loads(raw_path.read_text(encoding="utf-8"))

    by_slug = {row["slug"]: row for row in rows}
    missing = [s for s in SLUGS if s not in by_slug]
    if missing:
        raise SystemExit("missing recorded rows for: {0}".format(missing))

    ordered = [by_slug[s] for s in SLUGS]
    (FIXTURES / "markets.json").write_text(
        json.dumps(ordered, indent=1, ensure_ascii=False), encoding="utf-8")
    (FIXTURES / "evidence.json").write_text(
        json.dumps(EVIDENCE, indent=1, ensure_ascii=False), encoding="utf-8")
    (FIXTURES / "meta.json").write_text(
        json.dumps({"recorded_at": RECORDED_AT,
                    "note": "market rows are real; Jev answers are hand-written"},
                   indent=1), encoding="utf-8")

    script = {}
    for slug in SLUGS:
        market_id = str(by_slug[slug]["id"])
        probabilities = SCREEN[slug]
        script["screen:{0}".format(market_id)] = {
            "answers": {"h{0}".format(i): noul(p)
                        for i, p in enumerate(probabilities)}
        }
        for index, headline in enumerate(EVIDENCE[slug]):
            answers = DEEP.get((slug, index))
            if answers is None:
                continue
            tag = "deep:{0}:{1}".format(market_id, headline_key(headline["title"]))
            script[tag] = {
                "answers": {
                    "condition_met": noul(answers["condition_met"]),
                    "condition_precluded": noul(answers["condition_precluded"]),
                    "stance": choice(answers["stance"], answers["stance_conf"]),
                    "directness": score(answers["directness"], answers["directness_conf"]),
                    "rule_ambiguity": score(answers["ambiguity"], answers["ambiguity_conf"]),
                    "source_authority": score(answers["authority"], answers["authority_conf"]),
                }
            }

    (FIXTURES / "jev_script.json").write_text(
        json.dumps(script, indent=1, ensure_ascii=False), encoding="utf-8")

    print("wrote {0} markets, {1} scripted responses".format(
        len(ordered), len(script)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
