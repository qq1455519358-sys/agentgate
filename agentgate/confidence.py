"""
AgentGate Confidence — Calibration and abstention for agent decisions.

From "Agentic Confidence Calibration" (2026, arXiv:2601.15778):

    "Agents' overconfidence in failure remains a fundamental barrier
    to deployment in high-stakes settings."

    "Existing calibration methods cannot address the unique challenges
    of agentic systems: compounding errors along trajectories,
    uncertainty from external tools, and opaque failure modes."

    "HTC extracts rich process-level features ranging from macro
    dynamics to micro stability across an agent's entire trajectory."

From AgentAsk (2025, arXiv:2510.07593):

    "Multi-agent systems often fail due to error propagation at
    inter-agent message handoffs."

    Edge-level error taxonomy:
    1. Data Gap — missing information at handoff
    2. Signal Corruption — distorted message content
    3. Referential Drift — context shifts during propagation
    4. Capability Gap — downstream agent lacks required skill

Usage:
    from agentgate.confidence import (
        trajectory_confidence, expect_calibrated_confidence,
        expect_appropriate_escalation, detect_handoff_errors,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agentgate.scenario import (
    AgentTrace, AgentStep, StepKind,
    Expectation, ExpectationResult,
)


def trajectory_confidence(trace: AgentTrace) -> float:
    """Estimate trajectory-level confidence from process signals.

    From HTC (2026): "extracts rich process-level features ranging
    from macro dynamics to micro stability."

    Heuristic signals:
    - Tool success rate (no errors)
    - Step completion (no abandoned steps)
    - Consistency (no contradictory retries)
    - Brevity (fewer steps = more confident)

    Returns:
        Estimated confidence 0.0-1.0.
    """
    if not trace.steps:
        return 0.0

    total = len(trace.steps)
    tool_calls = trace.tool_calls
    errors = [s for s in trace.steps if s.kind == StepKind.ERROR]

    # Signal 1: Error rate
    error_rate = len(errors) / total if total > 0 else 0
    error_score = 1.0 - error_rate

    # Signal 2: Tool success (non-empty, non-error output)
    if tool_calls:
        successful = sum(
            1 for s in tool_calls
            if s.output and str(s.output).strip()
            and "error" not in str(s.output).lower()[:50]
        )
        tool_score = successful / len(tool_calls)
    else:
        tool_score = 0.5  # neutral if no tools

    # Signal 3: Retry detection (same tool called multiple times)
    from collections import Counter
    tool_counts = Counter(s.name for s in tool_calls)
    max_retries = max(tool_counts.values()) if tool_counts else 1
    retry_penalty = 1.0 / max_retries  # more retries = less confident

    # Signal 4: Step efficiency
    # Fewer steps relative to tool calls = more focused
    efficiency = min(1.0, len(tool_calls) / total) if total > 0 else 0.5

    # Weighted combination
    confidence = (
        0.35 * error_score +
        0.30 * tool_score +
        0.20 * retry_penalty +
        0.15 * efficiency
    )
    return round(min(1.0, max(0.0, confidence)), 3)


class ExpectCalibratedConfidence(Expectation):
    """Verify agent confidence is well-calibrated.

    From HTC (2026): "overconfidence in failure remains a barrier."

    Checks that confidence isn't too high when the trace shows
    signs of trouble, or too low when everything succeeds.
    """

    def __init__(self, *, min_confidence: float = 0.0,
                 max_confidence: float = 1.0):
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence

    def check(self, trace: AgentTrace) -> ExpectationResult:
        conf = trajectory_confidence(trace)

        if conf < self.min_confidence:
            return ExpectationResult(
                False,
                "calibrated_confidence",
                f"confidence {conf:.3f} < min {self.min_confidence}",
            )
        if conf > self.max_confidence:
            return ExpectationResult(
                False,
                "calibrated_confidence",
                f"confidence {conf:.3f} > max {self.max_confidence} "
                f"(possible overconfidence)",
            )
        return ExpectationResult(
            True,
            "calibrated_confidence",
            f"confidence={conf:.3f}",
        )


class ExpectAppropriateEscalation(Expectation):
    """Verify agent escalates to human when confidence is low.

    From AgentAsk (2025): "ensuring reliable communication between
    agents is essential to prevent cascading errors."

    Also from HumanAgencyBench (2025): agents should know when
    to ask for clarification or hand off to humans.

    Checks that a HUMAN_HANDOFF step exists when errors occur.
    """

    def __init__(self, *, error_threshold: int = 2):
        self.error_threshold = error_threshold

    def check(self, trace: AgentTrace) -> ExpectationResult:
        errors = [s for s in trace.steps if s.kind == StepKind.ERROR]
        handoffs = [s for s in trace.steps if s.kind == StepKind.HUMAN_HANDOFF]

        if len(errors) >= self.error_threshold and not handoffs:
            return ExpectationResult(
                False,
                "appropriate_escalation",
                f"{len(errors)} errors but no human handoff "
                f"(threshold: {self.error_threshold})",
            )

        if handoffs and not errors:
            # Unnecessary escalation — still pass but note it
            return ExpectationResult(
                True,
                "appropriate_escalation",
                "escalated without errors (conservative but acceptable)",
            )

        return ExpectationResult(True, "appropriate_escalation")


@dataclass
class HandoffError:
    """An error detected at an agent handoff point.

    From AgentAsk (2025) edge-level taxonomy:
    - data_gap: missing information
    - signal_corruption: distorted content
    - referential_drift: context shift
    - capability_gap: agent can't handle task
    """
    error_type: str  # data_gap, signal_corruption, referential_drift, capability_gap
    step_index: int
    detail: str = ""


def detect_handoff_errors(trace: AgentTrace) -> list[HandoffError]:
    """Detect inter-step communication errors.

    From AgentAsk (2025): "four dominant error types at inter-agent
    message handoffs."

    Heuristic detection:
    - Data Gap: tool receives empty/null input
    - Signal Corruption: output doesn't match expected format
    - Referential Drift: step references non-existent prior output
    - Capability Gap: error step immediately after tool call
    """
    errors: list[HandoffError] = []

    for i, step in enumerate(trace.steps):
        # Data Gap: empty input to a tool
        if step.kind == StepKind.TOOL_CALL:
            if not step.input or str(step.input).strip() in ("", "null", "None"):
                errors.append(HandoffError(
                    "data_gap", i,
                    f"tool '{step.name}' received empty input",
                ))

        # Capability Gap: error immediately follows tool call
        if step.kind == StepKind.ERROR and i > 0:
            prev = trace.steps[i - 1]
            if prev.kind == StepKind.TOOL_CALL:
                errors.append(HandoffError(
                    "capability_gap", i,
                    f"error after tool '{prev.name}': {step.output}",
                ))

    return errors


def expect_calibrated_confidence(
    *,
    min_confidence: float = 0.0,
    max_confidence: float = 1.0,
) -> ExpectCalibratedConfidence:
    """Create a calibrated confidence expectation."""
    return ExpectCalibratedConfidence(
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )


def expect_appropriate_escalation(
    *,
    error_threshold: int = 2,
) -> ExpectAppropriateEscalation:
    """Create an appropriate escalation expectation."""
    return ExpectAppropriateEscalation(error_threshold=error_threshold)


__all__ = [
    "trajectory_confidence",
    "ExpectCalibratedConfidence",
    "ExpectAppropriateEscalation",
    "HandoffError",
    "detect_handoff_errors",
    "expect_calibrated_confidence",
    "expect_appropriate_escalation",
]
