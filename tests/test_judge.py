"""Tests for the question bank and the two-stage cascade.

The assertions about question *content* are not decoration. The whole premise of
this project is that Jev is never asked to forecast, so a question that slips in
a "will" is a design regression worth failing a build over.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmjev.config import Config
from pmjev.judge import best_judgment, deep_read, screen
from pmjev.models import Headline, Market, RunStats
from pmjev.questions import (DEEP_KEYS, STANCE_OPTIONS, deep_questions,
                             deep_state, screen_index, screen_questions,
                             screen_state)
from pmjev.typesafe import StubClient

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def headlines(n=3):
    return [
        Headline(title="Headline number {0}".format(i), source="Source {0}".format(i),
                 url="https://example.invalid/{0}".format(i), published=NOW)
        for i in range(n)
    ]


def market():
    return Market.from_gamma({
        "id": "77", "slug": "s", "question": "Will X happen?",
        "description": "Resolves YES if X happens." + " padding." * 2000,
        "outcomes": '["Yes", "No"]', "outcomePrices": '["0.40", "0.60"]',
        "clobTokenIds": '["1", "2"]',
    })


# --- question design ------------------------------------------------------

def test_deep_questions_cover_every_documented_key():
    questions = deep_questions()
    assert sorted(questions) == sorted(DEEP_KEYS)


def test_every_question_declares_a_valid_primitive():
    for key, question in deep_questions().items():
        assert question["type"] in ("noul", "choice", "score"), key


def test_score_questions_have_between_two_and_ten_levels():
    # The docs support 2-10 ordered levels; more than that cannot be described
    # distinctly enough for the model to separate them.
    for key, question in deep_questions().items():
        if question["type"] == "score":
            assert 2 <= len(question["criteria"]) <= 10, key


def test_score_levels_describe_concrete_situations():
    for key, question in deep_questions().items():
        if question["type"] != "score":
            continue
        for level in question["criteria"]:
            assert level["summary"].strip()
            assert level["signals"], key


def test_stance_choice_offers_a_no_match_outcome():
    # Without "not_relevant" the model has to force irrelevant news into a
    # direction, which is exactly how a keyword-search false positive becomes
    # a trade.
    criteria = deep_questions()["stance"]["criteria"]
    assert sorted(criteria) == sorted(STANCE_OPTIONS)
    assert "not_relevant" in criteria


def asked_questions():
    """Just the text the model is *asked*, not the criteria describing answers.

    The criteria legitimately mention predictions -- that is how the model is
    told a forecast does not count as evidence. Only the ask itself must stay
    free of forecasting language.
    """
    out = []
    for bank in (deep_questions(), screen_questions(headlines(1))):
        for key, question in bank.items():
            instructions = question["instructions"]
            if isinstance(instructions, dict):
                out.append((key, " ".join(
                    str(v) for k, v in instructions.items()).lower()))
            else:
                out.append((key, str(instructions).lower()))
    return out


def test_no_question_asks_the_model_to_predict_the_future():
    forbidden = ["predict", "forecast", "in the future", "will happen",
                 "will eventually", "what are the chances", "do you expect"]
    for key, text in asked_questions():
        for phrase in forbidden:
            assert phrase not in text, \
                "{0} drifted into forecasting: {1}".format(key, phrase)


def test_the_decisive_questions_are_anchored_in_the_past():
    # Both Nouls that can open a position must ask what the evidence reports as
    # having already happened. Drop the anchor and the model starts guessing.
    questions = deep_questions()
    for key in ("condition_met", "condition_precluded"):
        text = json.dumps(questions[key]).lower()
        assert "states as having happened" in text, key
        assert "do not consider" in text or "not merely" in text, key


def test_no_question_asks_for_arithmetic_or_date_comparison():
    # Jev is documented as weak at both. Everything numeric lives in scoring.py.
    forbidden = ["how many days", "calculate", "add up", "subtract",
                 "is the date", "before the deadline", "count the"]
    blob = json.dumps(deep_questions()).lower()
    for phrase in forbidden:
        assert phrase not in blob, "question bank drifted into arithmetic: {0}".format(phrase)


def test_questions_reference_state_fields_by_backticked_path():
    blob = json.dumps(deep_questions())
    for field in ("`evidence`", "`resolution_rules`", "`market_question`"):
        assert field in blob


# --- state ----------------------------------------------------------------

def test_deep_state_carries_exactly_one_article():
    state = deep_state("Will X happen?", "rules", headlines(1)[0])
    assert set(state) == {"market_question", "resolution_rules", "evidence"}
    assert isinstance(state["evidence"], dict)


def test_rules_are_truncated_to_guard_against_context_rot():
    state = deep_state("q", "x" * 9000, headlines(1)[0], rules_chars=4000)
    assert len(state["resolution_rules"]) == 4000


def test_screen_state_numbers_the_headlines():
    state = screen_state("q", "rules", headlines(3))
    assert [h["index"] for h in state["headlines"]] == [0, 1, 2]


def test_screen_question_ids_round_trip_to_headline_indexes():
    questions = screen_questions(headlines(4))
    assert len(questions) == 4
    for key in questions:
        assert 0 <= screen_index(key) < 4


# --- cascade --------------------------------------------------------------

def test_screen_keeps_only_headlines_over_the_threshold():
    cfg = Config()
    cfg.evidence.screen_threshold = 0.55
    client = StubClient({"screen:77": {"answers": {
        "h0": {"type": "noul", "noul": 0.90},
        "h1": {"type": "noul", "noul": 0.20},
        "h2": {"type": "noul", "noul": 0.60},
    }}})
    survivors = screen(client, market(), headlines(3), cfg)
    assert [h.title for h, _ in survivors] == ["Headline number 0", "Headline number 2"]


def test_screen_respects_the_deep_read_budget():
    cfg = Config()
    cfg.evidence.max_deep_reads = 1
    client = StubClient({"screen:77": {"answers": {
        "h0": {"type": "noul", "noul": 0.70},
        "h1": {"type": "noul", "noul": 0.95},
        "h2": {"type": "noul", "noul": 0.80},
    }}})
    survivors = screen(client, market(), headlines(3), cfg)
    assert len(survivors) == 1
    assert survivors[0][1] == pytest.approx(0.95)   # the strongest one


def test_screen_sends_one_request_for_all_headlines():
    # Batching is the whole reason the screen is affordable; asking one at a
    # time would resend the rules text with every headline.
    cfg = Config()
    client = StubClient({})
    screen(client, market(), headlines(8), cfg)
    assert len(client.calls) == 1
    assert len(client.calls[0]["questions"]) == 8


def test_unanswered_questions_produce_no_judgment():
    # A partial response must not be silently padded with defaults -- a missing
    # rule_ambiguity answer would read as "perfectly unambiguous" and size a
    # position off a value the model never gave.
    class PartialClient:
        def system_one(self, state, questions):
            from pmjev.typesafe import Answers
            return Answers(model="test", raw={"answers": {
                "condition_met": {"type": "noul", "noul": 0.9},
            }})

    assert deep_read(PartialClient(), market(), headlines(1)[0], Config()) is None


def test_best_judgment_keeps_the_most_decisive_article():
    cfg = Config()
    from pmjev.judge import _headline_key
    items = headlines(2)
    script = {
        "screen:77": {"answers": {
            "h0": {"type": "noul", "noul": 0.90},
            "h1": {"type": "noul", "noul": 0.90},
        }},
        "deep:77:{0}".format(_headline_key(items[0])): {"answers": _answers(0.60)},
        "deep:77:{0}".format(_headline_key(items[1])): {"answers": _answers(0.93)},
    }
    stats = RunStats()
    judgment = best_judgment(StubClient(script), market(), items, cfg, stats)
    assert judgment is not None
    assert judgment.condition_met == pytest.approx(0.93)
    assert stats.deep_judgments == 2
    assert stats.requests_made == 3     # one screen plus two deep reads


def test_best_judgment_without_evidence_is_none():
    assert best_judgment(StubClient({}), market(), [], Config()) is None


def _answers(met: float):
    return {
        "condition_met": {"type": "noul", "noul": met},
        "condition_precluded": {"type": "noul", "noul": 0.02},
        "stance": {"type": "choice", "choice": "supports_yes",
                   "probabilities": {"supports_yes": 0.9}, "confidence": 0.9},
        "directness": {"type": "score", "score": 2.8, "confidence": 0.9},
        "rule_ambiguity": {"type": "score", "score": 0.3, "confidence": 0.9},
        "source_authority": {"type": "score", "score": 2.7, "confidence": 0.9},
    }
