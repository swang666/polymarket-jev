"""Append-only JSONL log of every judgment the scanner makes.

This file is the whole point of the project's first month. Until there are
logged judgments with realised outcomes attached, there is no evidence that the
model beats the market price, and no basis for risking money on it. Each record
keeps the exact state and questions sent, so a bad call can be traced back to
the text that produced it rather than argued about from memory.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .models import Signal, utcnow


class JsonlStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=_default))
            handle.write("\n")

    def append_signal(self, signal: Signal, run_id: str,
                      state: Optional[Any] = None,
                      questions: Optional[Dict[str, Any]] = None) -> None:
        record = signal.to_dict()
        record["run_id"] = run_id
        record["kind"] = "signal"
        if state is not None:
            record["state"] = state
        if questions is not None:
            record["questions"] = questions
        self.append(record)

    def records(self) -> Iterator[Dict[str, Any]]:
        if not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    continue

    def signals(self) -> List[Dict[str, Any]]:
        return [r for r in self.records() if r.get("kind") == "signal"]


def new_run_id(now: Optional[datetime] = None) -> str:
    return (now or utcnow()).strftime("%Y%m%dT%H%M%SZ")


def _default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
