"""
AgentGate Types — Core data structures.

This module contains the foundational types used throughout AgentGate:
- StepKind, AgentStep, AgentTrace — execution trace primitives
- ExpectationResult — result of a single expectation check
- ScenarioResult, SingleRunResult, SuiteResult — result containers

Split from scenario.py for clarity and to avoid circular imports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from math import comb
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Trace primitives
# ---------------------------------------------------------------------------

class StepKind(Enum):
    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"
    STATE_CHANGE = "state_change"
    HUMAN_HANDOFF = "human_handoff"
    ERROR = "error"


@dataclass
class AgentStep:
    """A single step in the agent's execution trace."""
    kind: StepKind
    name: str
    input: dict[str, Any] | Any = field(default_factory=dict)
    output: Any = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentTrace:
    """Complete execution trace of an agent run."""
    input: str
    output: Any = None
    steps: list[AgentStep] = field(default_factory=list)
    total_duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_calls(self) -> list[AgentStep]:
        return [s for s in self.steps if s.kind == StepKind.TOOL_CALL]

    @property
    def tool_names(self) -> list[str]:
        return [s.name for s in self.tool_calls]

    @property
    def errors(self) -> list[AgentStep]:
        return [s for s in self.steps if s.kind == StepKind.ERROR or s.error]

    @property
    def state_changes(self) -> list[AgentStep]:
        return [s for s in self.steps if s.kind == StepKind.STATE_CHANGE]

    def get_state(self, key: str) -> Any:
        for step in reversed(self.state_changes):
            if step.name == key:
                return step.output
        return None


# ---------------------------------------------------------------------------
# Expectation results
# ---------------------------------------------------------------------------

class ExpectationResult:
    """Result of a single expectation check."""

    def __init__(self, passed: bool, description: str, detail: str = "",
                 trace_context: Optional[list[str]] = None):
        self.passed = passed
        self.description = description
        self.detail = detail
        self.trace_context = trace_context or []

    def __repr__(self) -> str:
        icon = "✅" if self.passed else "❌"
        parts = [f"{icon} {self.description}"]
        if self.detail:
            parts[0] += f" — {self.detail}"
        if not self.passed and self.trace_context:
            parts.append("    Trace:")
            for line in self.trace_context:
                parts.append(f"      {line}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class SingleRunResult:
    """Result of a single run within a multi-run scenario execution."""
    run_index: int
    passed: bool
    expectations: list[ExpectationResult]
    trace: Optional[AgentTrace]
    duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class ScenarioResult:
    """Result of running a scenario (possibly multiple times)."""
    scenario_name: str
    input: str
    passed: bool
    expectations: list[ExpectationResult]
    trace: AgentTrace
    duration_ms: float = 0.0
    run_results: list[SingleRunResult] = field(default_factory=list)
    pass_rate: float = 0.0
    min_pass_rate: float = 1.0
    timed_out: bool = False
    timeout_detail: str = ""

    @property
    def score(self) -> float:
        """Partial credit score based on milestone weights (0.0-1.0)."""
        if not self.expectations:
            return 1.0 if self.passed else 0.0

        milestones = [e for e in self.expectations if "milestone(" in e.description]
        regulars = [e for e in self.expectations if "milestone(" not in e.description]

        if not milestones:
            if not regulars:
                return 1.0
            return sum(1 for e in regulars if e.passed) / len(regulars)

        total_weight = sum(
            float(e.description.split("weight=")[1].rstrip(")"))
            if "weight=" in e.description else 1.0
            for e in milestones
        )
        achieved_weight = sum(
            float(e.description.split("weight=")[1].rstrip(")"))
            if "weight=" in e.description else 1.0
            for e in milestones if e.passed
        )
        milestone_score = achieved_weight / total_weight if total_weight > 0 else 0.0

        if regulars:
            regular_score = sum(1 for e in regulars if e.passed) / len(regulars)
            return 0.7 * milestone_score + 0.3 * regular_score
        return milestone_score

    @property
    def failed_expectations(self) -> list[ExpectationResult]:
        return [e for e in self.expectations if not e.passed]

    @property
    def statistical_summary(self) -> str:
        if not self.run_results:
            icon = "✅" if self.passed else "❌"
            return f"{icon} Single run — {'passed' if self.passed else 'failed'}"
        total = len(self.run_results)
        passed_count = sum(1 for r in self.run_results if r.passed)
        met = self.pass_rate >= self.min_pass_rate
        icon = "✅" if met else "❌"
        return (
            f"Passed {passed_count}/{total} runs ({self.pass_rate:.0%}) "
            f"— min_pass_rate: {self.min_pass_rate:.0%} → {icon}"
        )

    def summary(self) -> str:
        total = len(self.expectations)
        passed = sum(1 for e in self.expectations if e.passed)
        icon = "✅" if self.passed else "❌"
        lines: list[str] = []

        if self.run_results:
            lines.append(f"{icon} Scenario: {self.scenario_name} — {self.statistical_summary}")
        else:
            lines.append(f"{icon} Scenario: {self.scenario_name} ({passed}/{total} expectations passed)")

        if self.timed_out:
            lines.append(f"  ⏰ TIMEOUT: {self.timeout_detail}")

        for e in self.expectations:
            lines.append(f"  {e}")

        if self.run_results and not self.passed:
            lines.append(f"  --- Per-run breakdown ---")
            for rr in self.run_results:
                run_icon = "✅" if rr.passed else "❌"
                suffix = f" (error: {rr.error})" if rr.error else ""
                lines.append(f"  Run {rr.run_index + 1}: {run_icon}{suffix}")
                if not rr.passed:
                    for e in rr.expectations:
                        if not e.passed:
                            lines.append(f"    {e}")

        return "\n".join(lines)


@dataclass
class SuiteResult:
    """Result of running a full scenario suite."""
    suite_name: str
    results: list[ScenarioResult]
    duration_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def average_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def pass_at_k(self) -> float | None:
        multi_run = [r for r in self.results if r.run_results]
        if not multi_run:
            return None
        scores = []
        for r in multi_run:
            n = len(r.run_results)
            c = sum(1 for rr in r.run_results if rr.passed)
            scores.append(c / n if n > 0 else 0.0)
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def pass_power_k(self) -> float | None:
        multi_run = [r for r in self.results if r.run_results]
        if not multi_run:
            return None
        scores = []
        for r in multi_run:
            n = len(r.run_results)
            c = sum(1 for rr in r.run_results if rr.passed)
            scores.append(comb(c, n) / comb(n, n) if comb(n, n) > 0 else 0.0)
        return sum(scores) / len(scores) if scores else 0.0

    def pass_power_k_series(self) -> dict[int, float] | None:
        multi_run = [r for r in self.results if r.run_results]
        if not multi_run:
            return None
        n = max(len(r.run_results) for r in multi_run)
        series: dict[int, float] = {}
        for k in range(1, n + 1):
            task_scores = []
            for r in multi_run:
                ni = len(r.run_results)
                ci = sum(1 for rr in r.run_results if rr.passed)
                denom = comb(ni, k)
                numer = comb(ci, k)
                task_scores.append(numer / denom if denom > 0 else 0.0)
            series[k] = sum(task_scores) / len(task_scores) if task_scores else 0.0
        return series

    def summary(self) -> str:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        icon = "✅" if self.passed else "❌"
        has_milestones = any(
            "milestone(" in e.description
            for r in self.results for e in r.expectations
        )
        score_str = f" | score: {self.average_score:.1%}" if has_milestones else ""
        lines = [
            f"\n{'='*60}",
            f"{icon} Suite: {self.suite_name} — {passed}/{total} scenarios passed ({self.pass_rate:.0%}){score_str}",
            f"{'='*60}",
        ]

        series = self.pass_power_k_series()
        if series is not None:
            parts = [f"pass^{k}={v:.3f}" for k, v in series.items()]
            lines.append(f"  Consistency (τ-bench): {', '.join(parts)}")

        for r in self.results:
            lines.append(r.summary())
            lines.append("")
        return "\n".join(lines)
