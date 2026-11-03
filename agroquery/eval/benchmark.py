"""Benchmark questions (benchmark/questions.jsonl)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from agroquery.config import BENCHMARK_FILE

DIFFICULTIES = ("easy", "medium", "hard")


@dataclass(frozen=True)
class Question:
    id: str
    difficulty: str
    question: str
    gold_sql: str
    order_matters: bool = False
    tags: list[str] = field(default_factory=list)


def load_questions(path: Path = BENCHMARK_FILE) -> list[Question]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = Question(**json.loads(line))
            assert q.difficulty in DIFFICULTIES, q
            out.append(q)
    ids = [q.id for q in out]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    return out
