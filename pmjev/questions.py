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


# ---------------------------------------------------------------------------
# Forecast mode.
#
# This asks how a market will RESOLVE, not what a published fact already
# settled. It is a harder question and a different bet on the model, so three
# things keep it from degenerating into a bare prior:
#
# 1. The market price is never in state. If the model sees it, it anchors, and
#    a forecast that echoes the price tells you nothing you did not already
#    know -- while quietly breaking the backtest, which compares the two.
# 2. The numeric base rate lives in code. The model classifies what KIND of
#    event this is; config.forecast.base_rates supplies the number. "Things
#    requiring an extraordinary departure rarely happen" is a base rate, not
#    something a text model should be asked to recall as a figure.
# 3. Dates stay out. Time remaining is computed in scoring.py and passed as a
#    pre-computed phrase, because Jev treats dates as text, not quantities.
# ---------------------------------------------------------------------------

FORECAST_KEYS: List[str] = [
    "resolves_yes",
    "event_class",
    "evidence_tilt",
    "forecast_rule_ambiguity",
    "evidence_sufficiency",
]

EVENT_CLASSES = [
    "scheduled_routine",
    "contested_competitive",
    "requires_specific_action",
    "requires_unusual_departure",
    "requires_extraordinary_change",
]

TILT_LEVELS = 5
SUFFICIENCY_LEVELS = 4


def forecast_state(question: str, rules: str, headlines: Sequence[Headline],
                   time_remaining: str, rules_chars: int = 4000,
                   snippet_chars: int = 600, max_items: int = 6) -> Dict[str, Any]:
    """State for a forecast. Deliberately carries no price and no raw dates.

    `time_remaining` is a phrase computed in code ("about 3 weeks left"), never
    a pair of dates for the model to subtract.
    """
    return {
        "market_question": question,
        "resolution_rules": rules[:rules_chars],
        "time_remaining": time_remaining,
        "recent_coverage": [
            {
                "headline": h.title,
                "publisher": h.source or "unknown",
                "summary": (h.snippet or h.title)[:snippet_chars],
            }
            for h in headlines[:max_items]
        ],
    }


def forecast_questions() -> Dict[str, Any]:
    """Five judgments that code combines into a probability."""
    return {
        "resolves_yes": {
            "type": "noul",
            "instructions": {
                "inspect": "`market_question`, `resolution_rules`, `recent_coverage` and `time_remaining`",
                "question": (
                    "Taking everything in state together, is this market headed "
                    "for a YES resolution under `resolution_rules`?"
                ),
                "focus": (
                    "Weigh what `recent_coverage` reports against what "
                    "`resolution_rules` literally requires, and against how much "
                    "time `time_remaining` says is left. Judge the situation as "
                    "described, not the topic in general."
                ),
            },
            "criteria": {
                "true": {
                    "what": (
                        "The situation described is on track to satisfy the rules: "
                        "the required step is under way, committed to, or is the "
                        "default outcome if nothing changes."
                    ),
                    "examples": [
                        "The process the rules describe is in motion and on schedule.",
                        "The actor the rules name has already committed to the act.",
                    ],
                },
                "false": {
                    "what": (
                        "Satisfying the rules needs something not under way: a "
                        "reversal, an unusual decision, or a step nobody has taken."
                    ),
                    "examples": [
                        "The coverage describes obstruction or delay with no resolution.",
                        "Nothing indicates the required step has begun.",
                        "The rules need an outcome that runs against the current direction.",
                    ],
                },
            },
        },
        "event_class": {
            "type": "choice",
            "instructions": {
                "inspect": "`resolution_rules` and `market_question`",
                "question": (
                    "What KIND of event does a YES resolution require? Classify "
                    "the requirement itself, not how likely you judge it to be."
                ),
            },
            "criteria": {
                "scheduled_routine": {
                    "what": "Something already scheduled that happens unless it is cancelled.",
                    "not_for": "A scheduled contest whose winner is in question.",
                    "examples": ["Will the scheduled summit take place?",
                                 "Will the report be published this quarter?"],
                },
                "contested_competitive": {
                    "what": "A genuine contest between a small number of named alternatives.",
                    "not_for": "A field so large that any one entrant is a long shot.",
                    "examples": ["Will this candidate win the election?",
                                 "Will this team win the tournament?"],
                },
                "requires_specific_action": {
                    "what": "A named actor must choose to do something they have not committed to.",
                    "not_for": "Actions already announced or under way.",
                    "examples": ["Will the central bank cut rates?",
                                 "Will the company announce an acquisition?"],
                },
                "requires_unusual_departure": {
                    "what": "A clear break from the status quo that is possible but uncommon.",
                    "not_for": "Routine political or corporate turnover.",
                    "examples": ["Will the leader resign before the end of the term?",
                                 "Will the treaty be abandoned?"],
                },
                "requires_extraordinary_change": {
                    "what": "Something with little or no precedent in comparable situations.",
                    "not_for": "Unlikely but previously observed outcomes.",
                    "examples": ["Will the government be overthrown?",
                                 "Will the alliance dissolve entirely?"],
                },
            },
        },
        "evidence_tilt": {
            "type": "score",
            "instructions": {
                "inspect": "`recent_coverage` against `resolution_rules`",
                "question": "Which way does the recent coverage lean for a YES resolution?",
                "note": (
                    "Rate the direction the reported facts point, not how "
                    "confident or dramatic the writing is."
                ),
            },
            "criteria": [
                {"summary": "Points firmly against YES",
                 "signals": ["Reports the opposite outcome, or the process abandoned"]},
                {"summary": "Leans against YES",
                 "signals": ["Setbacks, delays or opposition with no resolution"]},
                {"summary": "Neutral, mixed, or says nothing either way",
                 "signals": ["Background only", "Arguments reported on both sides"]},
                {"summary": "Leans toward YES",
                 "signals": ["Progress reported", "Supportive statements from those who decide"]},
                {"summary": "Points firmly toward YES",
                 "signals": ["The required step reported as under way or agreed"]},
            ],
        },
        "forecast_rule_ambiguity": {
            "type": "score",
            "instructions": {
                "inspect": "`resolution_rules`",
                "question": (
                    "How likely is it that two careful readers would disagree "
                    "about what these rules require for a YES resolution?"
                ),
                "note": "Rate the rules as written, independently of the coverage.",
            },
            "criteria": [
                {"summary": "The rules name an explicit trigger, source and deadline",
                 "signals": ["Nothing is left to judgment"]},
                {"summary": "Clear rules needing one small uncontroversial inference",
                 "signals": ["Any reasonable reader would read them the same way"]},
                {"summary": "A key term is left undefined",
                 "signals": ["Relies on major, official or significant without defining them"]},
                {"summary": "Silent, self-contradictory, or turning on a plainly contested term",
                 "signals": ["Readers would reach opposite conclusions in obvious cases"]},
            ],
        },
        "evidence_sufficiency": {
            "type": "score",
            "instructions": {
                "inspect": "`recent_coverage` in relation to `market_question`",
                "question": (
                    "How much does the supplied coverage actually tell you about "
                    "how this market resolves?"
                ),
                "note": (
                    "Rate how informative what is here is. Coverage that is "
                    "abundant but off-topic is still uninformative."
                ),
            },
            "criteria": [
                {"summary": "Nothing here bears on the question",
                 "signals": ["Empty, or entirely about other subjects"]},
                {"summary": "Same topic, but nothing about what the rules require",
                 "signals": ["Commentary and background only"]},
                {"summary": "Some reporting on the relevant process, incomplete",
                 "signals": ["Partial accounts", "Second-hand or unconfirmed"]},
                {"summary": "Direct, current reporting on exactly what the rules turn on",
                 "signals": ["Named sources describing the deciding process"]},
            ],
        },
    }
