"""The question bank -- every word Jev is ever asked by this project.

Three rules govern everything in this file, and breaking any of them is how a
scanner like this starts inventing edge:

1. **Never ask about the future.** A Noul probability is P(answer is yes | the
   text supplied), not P(the event happens). Every question below is a judgment
   about whether a *published* piece of evidence satisfies a *written* rule.
2. **Never ask for arithmetic or date comparison.** Jev is documented as weak at
   both ("Jev is not a calculator"; dates are treated as text, not quantities).
   Prices, spreads, edges and deadlines are computed in scoring.py.
3. **One narrow judgment per question.** Independent questions are batched into a
   single request, which the docs measure at roughly 12x cheaper and 10x faster
   than asking them separately.

Question ids are for our code only -- they are not sent to the model, so each
question carries its full meaning in `instructions`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .models import Headline

# ---------------------------------------------------------------------------
# Stage 1: the screen.
#
# A keyword news search returns mostly noise. Rather than pay for a deep read of
# every hit, we ask one cheap Noul per headline inside a single request, and only
# the survivors get the full question set.
# ---------------------------------------------------------------------------

SCREEN_PREFIX = "h"


def screen_state(question: str, rules: str, headlines: Sequence[Headline],
                 rules_chars: int = 4000) -> Dict[str, Any]:
    """State for the screen request: the rules plus a numbered headline list."""
    return {
        "market_question": question,
        "resolution_rules": rules[:rules_chars],
        "headlines": [
            {
                "index": i,
                "title": h.title,
                "source": h.source,
                "published": h.published.strftime("%Y-%m-%d") if h.published else "unknown",
            }
            for i, h in enumerate(headlines)
        ],
    }


def screen_questions(headlines: Sequence[Headline]) -> Dict[str, Any]:
    """One Noul per headline. All independent, so they batch into one request."""
    questions: Dict[str, Any] = {}
    for i, headline in enumerate(headlines):
        questions["{0}{1}".format(SCREEN_PREFIX, i)] = {
            "type": "noul",
            "instructions": {
                "inspect": "the entry in `headlines` whose `index` is {0}".format(i),
                "question": (
                    "Does that headline report a real-world development that bears "
                    "on whether `market_question` resolves YES under "
                    "`resolution_rules`? Judge only that one headline; ignore the "
                    "others."
                ),
            },
            "criteria": {
                "true": {
                    "what": (
                        "The headline names an event, decision, announcement or "
                        "measurement that the resolution rules turn on."
                    ),
                    "examples": [
                        "It reports the specific outcome the rules describe.",
                        "It reports an official act by the body the rules name.",
                        "It reports that the outcome the rules describe will not happen.",
                    ],
                },
                "false": {
                    "what": (
                        "The headline is about a different subject, or is commentary, "
                        "speculation, betting-odds coverage or background with no new "
                        "development bearing on the rules."
                    ),
                    "examples": [
                        "An opinion piece about the topic.",
                        "A story that only mentions the subject in passing.",
                        "Coverage of what prediction markets currently think.",
                    ],
                },
            },
        }
    return questions


def screen_index(key: str) -> int:
    """Map a screen question id back to its headline index."""
    return int(key[len(SCREEN_PREFIX):])


# ---------------------------------------------------------------------------
# Stage 2: the deep read.
#
# One request per surviving (market, evidence) pair, carrying the six judgments
# below. Only one article is in state at a time -- the docs warn that accuracy
# degrades with large, irrelevant context, and this is the cheapest defence.
# ---------------------------------------------------------------------------

DEEP_KEYS: List[str] = [
    "condition_met",
    "condition_precluded",
    "stance",
    "directness",
    "rule_ambiguity",
    "source_authority",
]

STANCE_OPTIONS = ["supports_yes", "supports_no", "mixed", "not_relevant"]


def deep_state(question: str, rules: str, headline: Headline,
               rules_chars: int = 4000, snippet_chars: int = 1200) -> Dict[str, Any]:
    """State for a deep read: the market's rules and exactly one article."""
    return {
        "market_question": question,
        "resolution_rules": rules[:rules_chars],
        "evidence": {
            "headline": headline.title,
            "publisher": headline.source or "unknown",
            "published": headline.published.strftime("%Y-%m-%d") if headline.published else "unknown",
            "text": (headline.snippet or headline.title)[:snippet_chars],
        },
    }


def deep_questions() -> Dict[str, Any]:
    """The six judgments, all answerable from the supplied text alone."""
    return {
        # --- the two that decide whether we have a view at all --------------
        "condition_met": {
            "type": "noul",
            "instructions": {
                "inspect": "`evidence` and `resolution_rules`",
                "question": (
                    "Does `evidence` report that the condition written in "
                    "`resolution_rules` for a YES resolution has ALREADY been met? "
                    "Judge only what `evidence` states as having happened. Do not "
                    "consider whether it is likely to happen later."
                ),
                "focus": (
                    "Read `resolution_rules` literally. If the rules require a "
                    "specific actor, threshold, form or source, the evidence must "
                    "show that exact thing, not something similar."
                ),
            },
            "criteria": {
                "true": {
                    "what": (
                        "The evidence states, as an accomplished fact, the very "
                        "occurrence the rules require for YES."
                    ),
                    "examples": [
                        "The rules require an official confirmation and the evidence reports that confirmation was issued.",
                        "The rules require a named person to take office and the evidence reports they were sworn in.",
                    ],
                },
                "false": {
                    "what": (
                        "The evidence reports something short of that: a plan, a "
                        "proposal, a prediction, a partial step, a similar but "
                        "different event, or nothing relevant at all."
                    ),
                    "examples": [
                        "The evidence says the event is expected or scheduled.",
                        "The evidence reports a step toward the outcome but not the outcome.",
                        "The evidence describes a different actor or a different threshold than the rules name.",
                    ],
                },
            },
        },
        "condition_precluded": {
            "type": "noul",
            "instructions": {
                "inspect": "`evidence` and `resolution_rules`",
                "question": (
                    "Does `evidence` report something that makes a YES resolution "
                    "under `resolution_rules` no longer possible? Judge only what "
                    "`evidence` states as having happened."
                ),
                "focus": (
                    "This is not merely bad news for YES. The reported fact must "
                    "close off the YES outcome under the rules as written."
                ),
            },
            "criteria": {
                "true": {
                    "what": (
                        "The evidence reports an outcome that is incompatible with "
                        "YES, or reports that the opportunity for YES has passed."
                    ),
                    "examples": [
                        "The rules ask whether a candidate wins and the evidence reports a different candidate won.",
                        "The rules ask whether a deal is signed by a deadline and the evidence reports the talks were formally abandoned.",
                    ],
                },
                "false": {
                    "what": (
                        "YES remains possible on the evidence, even if it now looks "
                        "less likely, or the evidence is not relevant."
                    ),
                    "examples": [
                        "The evidence reports a setback but no final outcome.",
                        "The evidence reports criticism, delay or opposition.",
                    ],
                },
            },
        },
        # --- the three that size the haircut -------------------------------
        "directness": {
            "type": "score",
            "instructions": {
                "inspect": "`evidence` against `resolution_rules`",
                "question": (
                    "How directly does `evidence` establish the specific fact that "
                    "`resolution_rules` turns on?"
                ),
                "note": (
                    "Rate the connection between the text and the rule, not how "
                    "important or interesting the story is."
                ),
            },
            "criteria": [
                {
                    "summary": "Does not touch the fact the rules turn on",
                    "signals": [
                        "Different subject, or the same subject in passing only",
                        "No event the rules name is mentioned",
                    ],
                },
                {
                    "summary": "Same topic, but reports no event the rules name",
                    "signals": [
                        "Background, analysis or commentary",
                        "Discusses what might happen rather than what did",
                    ],
                },
                {
                    "summary": "Reports the event at second hand or in preliminary form",
                    "signals": [
                        "Cites unnamed sources, or says a decision is expected",
                        "Partial, provisional or unconfirmed account",
                        "Reports another outlet's reporting",
                    ],
                },
                {
                    "summary": "States the named event as an accomplished fact, in the rules' own terms",
                    "signals": [
                        "Direct first-hand report of the occurrence",
                        "Quotes or cites the official act, filing, result or announcement",
                    ],
                },
            ],
        },
        "rule_ambiguity": {
            "type": "score",
            "instructions": {
                "inspect": "`resolution_rules` applied to `evidence`",
                "question": (
                    "If two careful readers were given only `resolution_rules` and "
                    "`evidence`, how likely is it that they would disagree about how "
                    "this market should resolve?"
                ),
                "note": (
                    "Rate the disagreement risk in applying these rules to this "
                    "evidence. Well-written rules applied to unrelated evidence are "
                    "still unambiguous."
                ),
            },
            "criteria": [
                {
                    "summary": "No judgment call: the rules name an explicit trigger and the evidence plainly meets it or plainly does not",
                    "signals": [
                        "The rules define the trigger, the source and the deadline",
                        "Both readers would reach the same answer immediately",
                    ],
                },
                {
                    "summary": "Clear rules, needing one small uncontroversial inference",
                    "signals": [
                        "The evidence uses different wording for plainly the same thing",
                        "Any reasonable reader would make the same connection",
                    ],
                },
                {
                    "summary": "A term is undefined, or the evidence fits the spirit but not the exact wording",
                    "signals": [
                        "The rules rely on a word like major, significant or official without defining it",
                        "A reasonable reader could argue either way",
                    ],
                },
                {
                    "summary": "The rules are silent, self-contradictory, or turn on a term readers would plainly define differently",
                    "signals": [
                        "The situation the evidence describes is not contemplated by the rules",
                        "Different parts of the rules point to different resolutions",
                    ],
                },
            ],
        },
        "source_authority": {
            "type": "score",
            "instructions": {
                "inspect": "the `publisher` and attribution inside `evidence`",
                "question": (
                    "How authoritative is this evidence for establishing the fact "
                    "that `resolution_rules` turns on?"
                ),
                "note": (
                    "If `resolution_rules` names a specific resolution source, "
                    "evidence from that source belongs at the top level."
                ),
            },
            "criteria": [
                {
                    "summary": "Anonymous, user-generated, satirical, or an aggregator with no named publisher",
                    "signals": ["No identifiable newsroom", "Social post or forum content"],
                },
                {
                    "summary": "A named outlet that is partisan, niche, or known for unverified claims",
                    "signals": ["Advocacy publication", "Content farm or SEO aggregator"],
                },
                {
                    "summary": "An established news organisation reporting in its own name",
                    "signals": [
                        "Major wire service or national newspaper",
                        "Cites a named official or a document it has seen",
                    ],
                },
                {
                    "summary": "The primary source the rules name or imply -- the government body, court, company, exchange or official account itself",
                    "signals": [
                        "An official statement, filing, gazette or result page",
                        "The exact resolution source named in the rules",
                    ],
                },
            ],
        },
        # --- one categorical summary, useful for filtering and for review ---
        "stance": {
            "type": "choice",
            "instructions": {
                "inspect": "`evidence` in light of `resolution_rules`",
                "question": (
                    "Which way does this evidence point for the resolution of "
                    "`market_question`?"
                ),
            },
            "criteria": {
                "supports_yes": {
                    "what": "The reported facts push toward a YES resolution.",
                    "not_for": "Evidence that only restates the question.",
                    "examples": ["Reports the required event occurred or is being formalised."],
                },
                "supports_no": {
                    "what": "The reported facts push toward a NO resolution.",
                    "not_for": "Mere criticism with no reported outcome.",
                    "examples": ["Reports the opposite outcome, or that the process was abandoned."],
                },
                "mixed": {
                    "what": "The evidence reports facts pointing both ways.",
                    "not_for": "Evidence that is simply off-topic -- use not_relevant.",
                    "examples": ["One part of the story advances the condition, another undercuts it."],
                },
                "not_relevant": {
                    "what": "The evidence has no bearing on how this market resolves.",
                    "not_for": "Weak but genuinely relevant evidence.",
                    "examples": ["A different story that shares a keyword with the market."],
                },
            },
        },
    }
