"""Client for the TypeSafe System One HTTP API (the Jev model).

Deliberately a thin `requests` wrapper rather than the official SDK: this project
sends one request shape and needs no extra dependency to do it. The `typesafe-sdk`
package is a drop-in alternative if you would rather have the typed helpers.

    POST https://api.typesafe.ai/v1/systemone
    Authorization: Bearer <key>
    {"state": ..., "model": "jev-latest", "questions": {"id": {...}}}

Response:
    {"model": ..., "answers": {"id": {...}}, "usage": {"input_tokens": N, ...}}
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from .config import TypeSafeConfig

log = logging.getLogger(__name__)


class TypeSafeError(RuntimeError):
    """The API refused or failed in a way retrying will not fix."""


class BudgetExceeded(TypeSafeError):
    """The configured per-run spend cap was reached. Nothing further was sent."""


@dataclass
class Answers:
    """One response, indexed the way the API returns it."""

    model: str
    raw: Dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def answers(self) -> Dict[str, Any]:
        return self.raw.get("answers", {}) or {}

    def noul(self, key: str, default: float = 0.0) -> float:
        """Probability the answer to a yes/no question is yes."""
        answer = self.answers.get(key) or {}
        value = answer.get("noul", default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def choice(self, key: str) -> str:
        return (self.answers.get(key) or {}).get("choice", "")

    def choice_probs(self, key: str) -> Dict[str, float]:
        probs = (self.answers.get(key) or {}).get("probabilities", {}) or {}
        out: Dict[str, float] = {}
        for option, value in probs.items():
            try:
                out[str(option)] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    def score(self, key: str, default: float = 0.0) -> float:
        """Probability-weighted position on the question's ordered levels."""
        answer = self.answers.get(key) or {}
        try:
            return float(answer.get("score", default))
        except (TypeError, ValueError):
            return default

    def confidence(self, key: str, default: float = 0.0) -> float:
        """How concentrated the distribution is -- not a correctness estimate."""
        answer = self.answers.get(key) or {}
        try:
            return float(answer.get("confidence", default))
        except (TypeError, ValueError):
            return default

    def missing(self, expected: Dict[str, Any]) -> list:
        return sorted(set(expected) - set(self.answers))


@dataclass
class UsageMeter:
    """Running spend for one scan, so a runaway loop cannot quietly bill you."""

    price_per_billion: float = 42.0
    budget_usd: float = 0.50
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0

    @property
    def spent_usd(self) -> float:
        return self.input_tokens * self.price_per_billion / 1e9

    def check(self) -> None:
        if self.budget_usd > 0 and self.spent_usd >= self.budget_usd:
            raise BudgetExceeded(
                "per-run budget of ${0:.4f} reached after {1} requests "
                "({2:,} input tokens)".format(
                    self.budget_usd, self.requests, self.input_tokens
                )
            )

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.requests += 1
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)


class TypeSafeClient:
    """Synchronous System One client with retry, budget accounting and logging."""

    RETRY_STATUS = {408, 429, 500, 502, 503, 504}

    def __init__(self, cfg: TypeSafeConfig,
                 session: Optional[requests.Session] = None,
                 meter: Optional[UsageMeter] = None) -> None:
        if not cfg.api_key:
            raise TypeSafeError(
                "no TypeSafe API key. Set TYPESAFE_API_KEY in the environment or "
                "in .env (get one at https://console.typesafe.ai/), or run with "
                "--demo for the offline walkthrough."
            )
        self.cfg = cfg
        self.session = session or requests.Session()
        self.meter = meter or UsageMeter(
            price_per_billion=cfg.price_per_billion_tokens,
            budget_usd=cfg.budget_usd_per_run,
        )
        self.last_request: Optional[Dict[str, Any]] = None

    def system_one(self, state: Any, questions: Dict[str, Any]) -> Answers:
        if not questions:
            raise TypeSafeError("system_one called with no questions")
        self.meter.check()

        body = {"state": state, "model": self.cfg.model, "questions": questions}
        self.last_request = body
        headers = {
            "Authorization": "Bearer {0}".format(self.cfg.api_key),
            "Content-Type": "application/json",
        }

        last_error: Optional[str] = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                response = self.session.post(
                    self.cfg.base_url, headers=headers, json=body,
                    timeout=self.cfg.timeout,
                )
            except requests.RequestException as exc:
                last_error = "{0}: {1}".format(type(exc).__name__, exc)
                self._backoff(attempt, last_error)
                continue

            if response.status_code in self.RETRY_STATUS:
                last_error = "HTTP {0}: {1}".format(
                    response.status_code, response.text[:200]
                )
                self._backoff(attempt, last_error, response=response)
                continue

            if response.status_code >= 400:
                raise TypeSafeError(
                    "TypeSafe returned HTTP {0}: {1}".format(
                        response.status_code, response.text[:400]
                    )
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise TypeSafeError(
                    "TypeSafe returned non-JSON: {0}".format(response.text[:200])
                ) from exc

            usage = payload.get("usage", {}) or {}
            self.meter.record(usage.get("input_tokens", 0), usage.get("output_tokens", 0))
            return Answers(
                model=payload.get("model", self.cfg.model),
                raw=payload,
                input_tokens=int(usage.get("input_tokens", 0) or 0),
                output_tokens=int(usage.get("output_tokens", 0) or 0),
            )

        raise TypeSafeError(
            "TypeSafe request failed after {0} attempts: {1}".format(
                self.cfg.max_retries + 1, last_error
            )
        )

    def _backoff(self, attempt: int, reason: str,
                 response: Optional[requests.Response] = None) -> None:
        if attempt >= self.cfg.max_retries:
            return
        delay = min(8.0, (2 ** attempt)) + random.uniform(0, 0.25)
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    pass
        log.warning("TypeSafe retry %d in %.1fs (%s)", attempt + 1, delay, reason)
        time.sleep(delay)


class StubClient:
    """Offline stand-in used by --demo and by the tests.

    Replays recorded answers keyed by a caller-supplied tag, so the whole
    pipeline can be exercised end to end with no API key and no network.
    """

    def __init__(self, script: Dict[str, Dict[str, Any]],
                 model: str = "jev-1.13.0-stub") -> None:
        self.script = script
        self.model = model
        self.meter = UsageMeter(budget_usd=0.0)
        self.next_tag: Optional[str] = None
        self.calls: list = []

    def system_one(self, state: Any, questions: Dict[str, Any]) -> Answers:
        tag = self.next_tag or ""
        recorded = self.script.get(tag)
        if recorded is None:
            # Unscripted call: answer every question neutrally rather than crash,
            # so a demo stays runnable when fixtures drift.
            recorded = {"answers": _neutral_answers(questions)}
        self.calls.append({"tag": tag, "questions": sorted(questions)})
        approx_tokens = len(json.dumps({"s": state, "q": questions})) // 4
        self.meter.record(approx_tokens, 20)
        return Answers(
            model=self.model,
            raw=recorded,
            input_tokens=approx_tokens,
            output_tokens=20,
        )


def _neutral_answers(questions: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, question in questions.items():
        kind = question.get("type") if isinstance(question, dict) else "noul"
        if kind == "choice":
            options = list((question.get("criteria") or {}).keys()) or ["unknown"]
            share = 1.0 / len(options)
            out[key] = {
                "type": "choice",
                "choice": options[-1],
                "probabilities": {o: share for o in options},
                "confidence": 0.0,
            }
        elif kind == "score":
            levels = question.get("criteria") or []
            top = max(len(levels) - 1, 1)
            out[key] = {
                "type": "score",
                "score": top / 2.0,
                "probabilities": {},
                "confidence": 0.0,
            }
        else:
            out[key] = {"type": "noul", "noul": 0.5}
    return out
