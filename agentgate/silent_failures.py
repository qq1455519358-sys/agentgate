"""
AgentGate Silent Failures — Detect drift, cycles, and missing details.

From "Detecting Silent Failures in Multi-Agentic AI Trajectories"
(IBM Research, 2025, arXiv:2511.04032):

    "Multi-Agentic AI systems are inherently non-deterministic and
    prone to silent failures such as drift, cycles, and missing details
    in outputs, which are difficult to detect."

    Table 1 — Silent Failure Taxonomy:
    - Drift: agent diverges from intended path
    - Cycles: agent repeatedly invokes itself/tools in loops
    - Missing Details: returns response without crucial info
    - Tool Failures: external tools fail silently
    - Context Propagation Failures: incorrect context forwarding

From "Accurate Failure Prediction in Agents Does Not Imply Effective
Failure Prevention" (2026, arXiv:2602.03338):

    "Our pilot-based framework enables teams to detect failure modes
    before production deployment."

Usage:
    from agentgate.silent_failures import (
        detect_drift, detect_cycles, detect_missing_details,
        expect_no_silent_failures,
    )

    s.expectations.append(expect_no_silent_failures(
        expected_tools=["search", "book"],
        max_tool_repeats=2,
        required_output_keywords=["confirmation", "booking_id"],
    ))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

from agentgate.scenario import (
    AgentTrace, AgentStep, StepKind,
    Expectation, ExpectationResult,
)


@dataclass
class SilentFailureReport:
    """Report of detected silent failures.

    Attributes:
        has_drift: Agent deviated from expected tool path.
        has_cycles: Redundant loops detected.
        has_missing_details: Output lacks required information.
        has_tool_failures: Tool calls returned errors silently.
        drift_details: Unexpected tools called.
        cycle_details: Which tools were repeated and how many times.
        missing_keywords: Required keywords not found in output.
        silent_tool_errors: Tools that failed without explicit error.
    """
    has_drift: bool = False
    has_cycles: bool = False
    has_missing_details: bool = False
    has_tool_failures: bool = False
    drift_details: list[str] = field(default_factory=list)
    cycle_details: dict[str, int] = field(default_factory=dict)
    missing_keywords: list[str] = field(default_factory=list)
    silent_tool_errors: list[str] = field(default_factory=list)

    @property
    def has_any_failure(self) -> bool:
        return (self.has_drift or self.has_cycles or
                self.has_missing_details or self.has_tool_failures)

    @property
    def failure_types(self) -> list[str]:
        types = []
        if self.has_drift:
            types.append("drift")
        if self.has_cycles:
            types.append("cycles")
        if self.has_missing_details:
            types.append("missing_details")
        if self.has_tool_failures:
            types.append("tool_failures")
        return types

    def summary(self) -> str:
        if not self.has_any_failure:
            return "no silent failures detected"
        parts = []
        if self.has_drift:
            parts.append(f"drift: {self.drift_details}")
        if self.has_cycles:
            parts.append(f"cycles: {self.cycle_details}")
        if self.has_missing_details:
            parts.append(f"missing: {self.missing_keywords}")
        if self.has_tool_failures:
            parts.append(f"tool errors: {self.silent_tool_errors}")
        return "; ".join(parts)


def detect_drift(
    trace: AgentTrace,
    expected_tools: list[str],
) -> tuple[bool, list[str]]:
    """Detect if agent drifted from expected tool path.

    From IBM (2025): "Drift is when an agent chooses next agent A
    and tool B instead of agent D and tool E."

    Args:
        trace: The agent trace to analyze.
        expected_tools: Tools that should have been called.

    Returns:
        (has_drift, unexpected_tools) — tools called but not expected.
    """
    called = {s.name for s in trace.tool_calls}
    expected = set(expected_tools)
    unexpected = called - expected
    return bool(unexpected), list(unexpected)


def detect_cycles(
    trace: AgentTrace,
    max_repeats: int = 2,
) -> tuple[bool, dict[str, int]]:
    """Detect redundant tool invocation loops.

    From IBM (2025): "No agent or tool should be invoked more than
    once within a single execution" (strict criterion).

    Args:
        trace: The agent trace to analyze.
        max_repeats: Maximum allowed repetitions per tool.

    Returns:
        (has_cycles, {tool_name: count}) for tools exceeding threshold.
    """
    tool_counts = Counter(s.name for s in trace.tool_calls)
    cycles = {name: count for name, count in tool_counts.items()
              if count > max_repeats}
    return bool(cycles), cycles


def detect_missing_details(
    trace: AgentTrace,
    required_keywords: list[str],
) -> tuple[bool, list[str]]:
    """Detect if output is missing crucial information.

    From IBM (2025): "The agent returns a response without errors,
    but misses crucial information requested in the input query."

    Args:
        trace: The agent trace to analyze.
        required_keywords: Keywords that must appear in output.

    Returns:
        (has_missing, missing_keywords).
    """
    # Collect all output text
    all_output = " ".join(
        str(s.output).lower() for s in trace.steps
    )

    missing = [kw for kw in required_keywords
               if kw.lower() not in all_output]
    return bool(missing), missing


def detect_silent_tool_errors(trace: AgentTrace) -> tuple[bool, list[str]]:
    """Detect tools that failed without explicit error status.

    From IBM (2025): "External tools may fail silently, return
    unexpected results, hit rate limits."

    Heuristics:
    - Empty output from tool call
    - Output containing error-like patterns without error step
    """
    error_patterns = [
        "null", "none", "undefined", "n/a",
        "timeout", "rate limit", "429", "500",
    ]
    silent_errors = []
    for step in trace.tool_calls:
        output = str(step.output).lower().strip()
        if not output or output in ("", "null", "none"):
            silent_errors.append(f"{step.name}: empty output")
        elif any(pat in output for pat in error_patterns):
            # Only flag if no explicit ERROR step follows
            if step.kind != StepKind.ERROR:
                silent_errors.append(f"{step.name}: possible silent error")

    return bool(silent_errors), silent_errors


def full_silent_failure_scan(
    trace: AgentTrace,
    expected_tools: Optional[list[str]] = None,
    max_tool_repeats: int = 2,
    required_output_keywords: Optional[list[str]] = None,
) -> SilentFailureReport:
    """Run all silent failure detectors on a trace.

    Returns a comprehensive SilentFailureReport.
    """
    report = SilentFailureReport()

    if expected_tools:
        report.has_drift, report.drift_details = detect_drift(
            trace, expected_tools
        )

    report.has_cycles, report.cycle_details = detect_cycles(
        trace, max_tool_repeats
    )

    if required_output_keywords:
        report.has_missing_details, report.missing_keywords = (
            detect_missing_details(trace, required_output_keywords)
        )

    report.has_tool_failures, report.silent_tool_errors = (
        detect_silent_tool_errors(trace)
    )

    return report


class ExpectNoSilentFailures(Expectation):
    """Verify no silent failures occurred in the trace.

    Combines all detectors from IBM's taxonomy into one expectation.
    """

    def __init__(
        self,
        expected_tools: Optional[list[str]] = None,
        max_tool_repeats: int = 2,
        required_output_keywords: Optional[list[str]] = None,
    ):
        self.expected_tools = expected_tools
        self.max_tool_repeats = max_tool_repeats
        self.required_output_keywords = required_output_keywords

    def check(self, trace: AgentTrace) -> ExpectationResult:
        report = full_silent_failure_scan(
            trace,
            expected_tools=self.expected_tools,
            max_tool_repeats=self.max_tool_repeats,
            required_output_keywords=self.required_output_keywords,
        )

        if not report.has_any_failure:
            return ExpectationResult(True, "no_silent_failures")

        return ExpectationResult(
            False,
            "no_silent_failures",
            f"silent failures: {report.summary()}",
        )


def expect_no_silent_failures(
    expected_tools: Optional[list[str]] = None,
    max_tool_repeats: int = 2,
    required_output_keywords: Optional[list[str]] = None,
) -> ExpectNoSilentFailures:
    """Create a silent failure detection expectation."""
    return ExpectNoSilentFailures(
        expected_tools, max_tool_repeats, required_output_keywords,
    )


__all__ = [
    "SilentFailureReport",
    "detect_drift",
    "detect_cycles",
    "detect_missing_details",
    "detect_silent_tool_errors",
    "full_silent_failure_scan",
    "ExpectNoSilentFailures",
    "expect_no_silent_failures",
]
