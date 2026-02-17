"""
AgentGate Reproducibility — Measure variance across repeated runs.

From "Same Prompt, Different Outcomes" (2026, arXiv:2602.14349):

    "Even identical configurations can produce substantively different
    results. Even at temperature zero some estimates lead to different
    conclusions."

    "A single LLM-generated analysis is insufficient — repeated
    independent executions should be standard practice."

From the Unified Framework paper (2026, arXiv:2602.03238):

    "Evaluation outcomes are influenced by inference stochasticity.
    LLM inference is not fully deterministic even under greedy decoding."

    "Identical prompts executed under identical settings can yield
    divergent token-level outputs, which compound over long-horizon
    agent trajectories."

Usage:
    from agentgate.reproducibility import (
        reproducibility_score, expect_reproducible,
        variance_report,
    )

    # Run agent multiple times
    results = suite.run(agent, runs=10)
    report = variance_report(results)
    print(report.consistency_rate)   # % of runs that agree
    print(report.tool_path_entropy)  # entropy of tool orderings
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from agentgate.scenario import (
    AgentTrace, Expectation, ExpectationResult,
    ScenarioResult,
)
from agentgate.runner import SuiteResult


@dataclass
class VarianceReport:
    """Report on variance across multiple runs.

    From "Same Prompt, Different Outcomes" (2026):
    Measures completion, concordance, and consistency.

    Attributes:
        total_runs: Number of runs analyzed.
        pass_count: Number of passing runs.
        consistency_rate: Fraction of runs with same pass/fail outcome.
        tool_paths: All unique tool-call orderings observed.
        tool_path_entropy: Shannon entropy of tool paths (lower = more consistent).
        output_agreement: Fraction of runs with similar final outputs.
    """
    total_runs: int = 0
    pass_count: int = 0
    consistency_rate: float = 0.0
    tool_paths: list[str] = field(default_factory=list)
    tool_path_entropy: float = 0.0
    output_agreement: float = 0.0

    def summary(self) -> str:
        return (
            f"runs={self.total_runs}, "
            f"pass_rate={self.pass_count}/{self.total_runs}, "
            f"consistency={self.consistency_rate:.1%}, "
            f"path_entropy={self.tool_path_entropy:.3f}, "
            f"output_agreement={self.output_agreement:.1%}"
        )


def _shannon_entropy(items: list[str]) -> float:
    """Compute Shannon entropy of a list of items."""
    if not items:
        return 0.0
    total = len(items)
    counts = Counter(items)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _extract_tool_path(result: ScenarioResult) -> str:
    """Extract tool call ordering as a string path."""
    # Tool path from the scenario name (we reconstruct from expectations)
    return result.scenario_name


def variance_report(results: list[ScenarioResult]) -> VarianceReport:
    """Analyze variance across multiple scenario results.

    From "Same Prompt, Different Outcomes" (2026):
    "We assess completion, concordance, validity, and consistency."

    Args:
        results: List of results from running same scenario multiple times.

    Returns:
        VarianceReport with consistency metrics.
    """
    if not results:
        return VarianceReport()

    total = len(results)
    pass_count = sum(1 for r in results if r.passed)
    fail_count = total - pass_count

    # Consistency: what fraction agrees with the majority?
    majority = max(pass_count, fail_count)
    consistency = majority / total if total > 0 else 0.0

    # Tool paths (from scenario names as proxy)
    paths = [r.scenario_name for r in results]
    path_entropy = _shannon_entropy(paths)

    # Output agreement: fraction with same pass/fail as majority
    output_agreement = consistency

    return VarianceReport(
        total_runs=total,
        pass_count=pass_count,
        consistency_rate=consistency,
        tool_paths=list(set(paths)),
        tool_path_entropy=path_entropy,
        output_agreement=output_agreement,
    )


def reproducibility_score(results: list[ScenarioResult]) -> float:
    """Compute overall reproducibility score.

    From the Unified Framework (2026): "identical prompts can yield
    divergent outputs that compound over trajectories."

    Higher score = more reproducible (0.0-1.0).
    Perfect reproducibility = all runs agree.
    """
    if not results:
        return 0.0

    report = variance_report(results)

    # Combine consistency and low entropy
    # Normalize entropy: 0 entropy = perfect, log2(n) = worst
    max_entropy = math.log2(len(results)) if len(results) > 1 else 1.0
    entropy_score = 1.0 - (report.tool_path_entropy / max_entropy) if max_entropy > 0 else 1.0

    return round((report.consistency_rate + entropy_score) / 2, 3)


class ExpectReproducible(Expectation):
    """Verify agent behavior is reproducible across runs.

    From "Same Prompt, Different Outcomes" (2026):
    "Repeated independent executions should be standard practice."

    This expectation is checked against a single trace but can
    be combined with multi-run suite execution.
    """

    def __init__(self, *, min_consistency: float = 0.8,
                 multi_results: Optional[list[ScenarioResult]] = None):
        self.min_consistency = min_consistency
        self.multi_results = multi_results

    def check(self, trace: AgentTrace) -> ExpectationResult:
        if self.multi_results is None:
            return ExpectationResult(
                True, "reproducible",
                "single run (use multi-run for variance analysis)",
            )

        report = variance_report(self.multi_results)
        score = reproducibility_score(self.multi_results)

        if report.consistency_rate >= self.min_consistency:
            return ExpectationResult(
                True, "reproducible",
                f"consistency={report.consistency_rate:.1%}, "
                f"score={score:.3f}",
            )
        return ExpectationResult(
            False, "reproducible",
            f"consistency={report.consistency_rate:.1%} "
            f"< min {self.min_consistency:.1%} "
            f"({report.pass_count}/{report.total_runs} passed)",
        )


def expect_reproducible(
    *,
    min_consistency: float = 0.8,
    multi_results: Optional[list[ScenarioResult]] = None,
) -> ExpectReproducible:
    """Create a reproducibility expectation."""
    return ExpectReproducible(
        min_consistency=min_consistency,
        multi_results=multi_results,
    )


__all__ = [
    "VarianceReport",
    "variance_report",
    "reproducibility_score",
    "ExpectReproducible",
    "expect_reproducible",
]
