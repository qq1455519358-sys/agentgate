"""AgentGate type definitions.

All data classes and enumerations used across the SDK.
Kept in a single module to avoid circular imports.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(Enum):
    """Result of a single evaluation."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


class ScoreKind(Enum):
    """Type of scoring metric."""

    SEMANTIC = "semantic"
    RULE = "rule"
    LLM_JUDGE = "llm_judge"
    COMPOSITE = "composite"
    HALLUCINATION = "hallucination"
    DRIFT = "drift"


@dataclass(frozen=True)
class EvalResult:
    """Immutable result from any evaluator."""

    score: float  # 0.0 – 1.0
    verdict: Verdict
    kind: ScoreKind
    detail: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def passed(self, threshold: float = 0.5) -> bool:
        return self.score >= threshold


@dataclass
class TestCase:
    """A single test case with input, expected output, and metadata."""

    input: str
    expected: str
    name: str = ""
    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class TestResult:
    """Result of running a single test case."""

    test_case: TestCase
    eval_results: list[EvalResult] = field(default_factory=list)
    actual_output: str = ""
    duration_ms: float = 0.0
    error: str | None = None
    timestamp: float = field(default_factory=time.time)

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        if not self.eval_results:
            return False
        return all(r.verdict == Verdict.PASS for r in self.eval_results)

    @property
    def verdict(self) -> Verdict:
        if self.error:
            return Verdict.FAIL
        if not self.eval_results:
            return Verdict.SKIP
        if all(r.verdict == Verdict.PASS for r in self.eval_results):
            return Verdict.PASS
        if any(r.verdict == Verdict.FAIL for r in self.eval_results):
            return Verdict.FAIL
        return Verdict.WARN


@dataclass
class SuiteResult:
    """Aggregated result from running a full test suite."""

    suite_name: str
    results: list[TestResult] = field(default_factory=list)
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.passed_count / self.total

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.failed_count == 0


@dataclass
class RecordEntry:
    """A single recorded agent interaction."""

    input: str
    output: str
    timestamp: float = field(default_factory=time.time)
    agent_name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    approved: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
