"""
AgentGate State Diff — Outcome verification via environment state comparison.

From Agent-Diff (2026, arXiv:2602.11224):

    "A novel state-diff contract separates process from outcome — rather
    than fuzzy trace or parameter matching, we define task success as
    whether the expected change in environment state was achieved."

    "Because diffs are computed over the full environment state, we can
    enforce invariants and detect unintended side effects."

Also from Anthropic's "Demystifying evals for AI agents" (Jan 2026):

    "The outcome is the final state in the environment at the end of the
    trial. A flight-booking agent might say 'Your flight has been booked'
    but the outcome is whether a reservation exists in the database."

Usage:
    from agentgate.state_diff import StateDiff, expect_state_diff

    diff = StateDiff(
        expected_changes={"booking": {"status": "confirmed"}},
        forbidden_changes=["user_profile", "payment_history"],
    )
    s.expectations.append(expect_state_diff(diff))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agentgate.scenario import (
    AgentTrace, Expectation, ExpectationResult,
)


@dataclass
class StateDiff:
    """Define expected and forbidden state changes.

    From Agent-Diff (2026): "ΔS = Diff(S_tstart, S_tend)"

    Attributes:
        expected_changes: Dict of {key: expected_value} that must be
            present in the post-execution state. Nested dicts supported.
        forbidden_changes: List of state keys that must NOT change.
            Agent-Diff: "detect unintended side effects (modifications
            or deletions of unrelated resources)."
        required_unchanged: Dict of {key: value} that must remain
            the same after execution.
    """
    expected_changes: dict[str, Any] = field(default_factory=dict)
    forbidden_changes: list[str] = field(default_factory=list)
    required_unchanged: dict[str, Any] = field(default_factory=dict)


def _extract_state(trace: AgentTrace) -> dict[str, Any]:
    """Extract final state from trace metadata or state_change steps.

    Checks trace.metadata["state"] or builds state from state_change steps.
    """
    # Direct state in metadata
    if "state" in trace.metadata:
        return dict(trace.metadata["state"])
    if "post_state" in trace.metadata:
        return dict(trace.metadata["post_state"])

    # Build from state_change steps
    state: dict[str, Any] = {}
    for step in trace.state_changes:
        state[step.name] = step.output
    return state


def _extract_pre_state(trace: AgentTrace) -> dict[str, Any]:
    """Extract pre-execution state."""
    if "pre_state" in trace.metadata:
        return dict(trace.metadata["pre_state"])
    return {}


def _nested_get(d: dict, key: str) -> Any:
    """Get a value from a nested dict using dot notation."""
    parts = key.split(".")
    current = d
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return _MISSING
    return current


_MISSING = object()


def _check_value_match(actual: Any, expected: Any) -> bool:
    """Check if actual value matches expected, supporting nested dicts."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(
            k in actual and _check_value_match(actual[k], v)
            for k, v in expected.items()
        )
    return actual == expected


class ExpectStateDiff(Expectation):
    """Verify environment state changes match expectations.

    From Agent-Diff (2026):
        "We define task success as whether the expected change in
        environment state was achieved."

    Checks:
    1. Expected changes are present in final state
    2. Forbidden keys were not modified
    3. Required-unchanged values are preserved
    """

    def __init__(self, diff: StateDiff):
        self.diff = diff

    def check(self, trace: AgentTrace) -> ExpectationResult:
        state = _extract_state(trace)
        pre_state = _extract_pre_state(trace)
        failures: list[str] = []

        # Check expected changes
        for key, expected_val in self.diff.expected_changes.items():
            actual = _nested_get(state, key)
            if actual is _MISSING:
                failures.append(f"expected '{key}' in state, not found")
            elif not _check_value_match(actual, expected_val):
                failures.append(
                    f"'{key}': expected {expected_val!r}, got {actual!r}"
                )

        # Check forbidden changes
        for key in self.diff.forbidden_changes:
            pre_val = _nested_get(pre_state, key)
            post_val = _nested_get(state, key)
            if pre_val is not _MISSING and post_val is not _MISSING:
                if pre_val != post_val:
                    failures.append(
                        f"forbidden change to '{key}': "
                        f"{pre_val!r} → {post_val!r}"
                    )
            elif pre_val is _MISSING and post_val is not _MISSING:
                failures.append(f"forbidden key '{key}' was created")

        # Check required unchanged
        for key, required_val in self.diff.required_unchanged.items():
            actual = _nested_get(state, key)
            if actual is _MISSING:
                failures.append(f"required key '{key}' missing from state")
            elif actual != required_val:
                failures.append(
                    f"'{key}' should be {required_val!r}, got {actual!r}"
                )

        if failures:
            return ExpectationResult(
                False,
                "expect_state_diff",
                "; ".join(failures),
            )
        return ExpectationResult(True, "expect_state_diff")


def expect_state_diff(diff: StateDiff) -> ExpectStateDiff:
    """Create a state-diff expectation.

    Convenience function that wraps StateDiff in an ExpectStateDiff.
    """
    return ExpectStateDiff(diff)


__all__ = ["StateDiff", "ExpectStateDiff", "expect_state_diff"]
