"""
AgentGate Scenario — End-to-end behavioral testing for AI agents.

Unlike unit-level eval tools (DeepEval, promptfoo) that test individual LLM outputs,
Scenarios test the *agent as a system*: tool call sequences, state transitions,
failure recovery, and multi-step behavioral correctness.

Analogy:
  - DeepEval = Jest (unit test: is this function output correct?)
  - promptfoo = Vitest (faster unit test + security scanning)
  - AgentGate Scenario = Playwright (E2E: user clicks button → 5 pages → correct result?)

Academic grounding:
  - τ-bench (Yao et al., 2024; ICLR 2025): pass^k = C(c,k)/C(n,k)
    consistency metric for non-deterministic agents. Our
    SuiteResult.pass_power_k_series() implements the exact formula
    from τ-bench source code (sierra-research/tau-bench run.py).
  - TRACE (Kim et al., 2025): multi-dimensional trajectory evaluation.
    Our expectation system provides deterministic assertions complementary
    to TRACE's LLM-as-judge approach.
  - ICLR 2026 "A Hitchhiker's Guide to Agent Evaluation": recommends
    trajectory quality metrics (tool sequence correctness, policy adherence,
    adversarial safety tests). Our framework implements all three.
  - AgentHarm (Andriushchenko et al., 2024): adversarial benchmark for
    agent safety. Our adversarial.py generates similar scenarios.
  - Gabriel et al. (2024) "Advancing Agentic Systems" (NeurIPS 2024
    Workshop): Tool F1 / SSI metrics for tool-call graph evaluation.
    Our metrics module adapts these as Node F1 and Edge F1. See
    agentgate.metrics.

Design influenced by:
  - "Evals are essentially unit tests. They test the logic of the node,
    but they do not test the integrity of the graph." — The New Stack / Signadot
  - Playwright's expect/assertion API pattern
  - Anthropic's "Demystifying evals for AI agents" (Jan 9, 2026):
    task/trial/grader/transcript/outcome terminology alignment.

Key features (v0.2):
  - **Non-deterministic handling**: before/after relative ordering, min_times/max_times,
    statistical multi-run mode with configurable pass rates.
  - **Timeout protection**: per-scenario and per-suite timeouts via threading
    (Windows-compatible, no signal.alarm).
  - **Rich failure diagnostics**: full tool-call traces, state-change histories,
    and step-by-step context attached to every failed expectation.
"""

from __future__ import annotations

import time
import re
import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence
from pathlib import Path


# ---------------------------------------------------------------------------
# Data types
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
    name: str                          # tool name, model name, state key, etc.
    input: dict[str, Any] = field(default_factory=dict)
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
        """Get the last value of a state key from state_change steps."""
        for step in reversed(self.state_changes):
            if step.name == key:
                return step.output
        return None


# ---------------------------------------------------------------------------
# Trace formatting helpers (used by diagnostics)
# ---------------------------------------------------------------------------

def _format_step_short(step: AgentStep) -> str:
    """Format a step as a concise string for trace context.

    Examples:
        "search_flights({destination: Tokyo}) → {flights: [...]}"
        "search_flights → ❌ API timeout"
    """
    parts = [step.name]
    if step.input:
        # Compact dict representation
        arg_str = ", ".join(f"{k}: {_truncate(v)}" for k, v in step.input.items())
        parts[0] += f"({{{arg_str}}})"
    if step.error:
        parts.append(f"❌ {step.error}")
    elif step.output is not None:
        parts.append(str(_truncate(step.output)))
    return " → ".join(parts)


def _truncate(value: Any, max_len: int = 60) -> str:
    """Truncate a value representation for display."""
    s = repr(value) if not isinstance(value, str) else value
    if len(s) > max_len:
        return s[:max_len - 3] + "..."
    return s


def _build_tool_call_trace(trace: AgentTrace) -> list[str]:
    """Build a list of formatted tool call strings from a trace."""
    return [_format_step_short(s) for s in trace.tool_calls]


def _build_full_step_trace(trace: AgentTrace) -> list[str]:
    """Build a list of formatted strings for ALL steps in a trace."""
    lines: list[str] = []
    for i, step in enumerate(trace.steps, 1):
        prefix = f"[{i}] {step.kind.value}: "
        lines.append(prefix + _format_step_short(step))
    return lines


# ---------------------------------------------------------------------------
# Expectations — the assertion layer
# ---------------------------------------------------------------------------

class ExpectationResult:
    """Result of a single expectation check.

    Attributes:
        passed: Whether the expectation passed.
        description: Human-readable description of what was checked.
        detail: Short detail string for failures.
        trace_context: List of formatted step strings showing what actually
            happened. Populated on failure to provide full diagnostic context.

    Example trace_context:
        [
            "search_flights({destination: Tokyo}) → ❌ API timeout",
            "search_flights({destination: Tokyo}) → ❌ API timeout",
            "❌ TimeoutError",
        ]
    """

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


class Expectation:
    """Base class for all expectations."""
    def check(self, trace: AgentTrace) -> ExpectationResult:
        raise NotImplementedError


class ExpectToolCall(Expectation):
    """Expect a specific tool to be called.

    Supports both absolute ordering (``order``) and relative ordering
    (``before`` / ``after``), as well as exact (``times``) and range-based
    (``min_times`` / ``max_times``) call-count assertions.

    Args:
        tool_name: Name of the tool expected to be called.
        order: 1-indexed absolute position in the tool call sequence
            (legacy, kept for backward compatibility).
        before: Tool name that must appear *after* this tool (i.e. this tool
            is called before ``before``).
        after: Tool name that must appear *before* this tool (i.e. this tool
            is called after ``after``).
        with_args: Dict of argument key/value pairs that at least one call
            must match.
        times: Exact number of times the tool should be called.
        min_times: Minimum number of calls (inclusive).
        max_times: Maximum number of calls (inclusive).

    Example::

        # Relative ordering — more resilient to non-deterministic agents
        scenario.expect_tool_call("search", after="login", before="checkout")

        # Range-based call count
        scenario.expect_tool_call("retry_api", min_times=1, max_times=3)
    """

    def __init__(self, tool_name: str, *,
                 order: Optional[int] = None,
                 before: Optional[str] = None,
                 after: Optional[str] = None,
                 with_args: Optional[dict] = None,
                 times: Optional[int] = None,
                 min_times: Optional[int] = None,
                 max_times: Optional[int] = None):
        self.tool_name = tool_name
        self.order = order              # 1-indexed position in tool call sequence
        self.before = before            # must appear before this tool
        self.after = after              # must appear after this tool
        self.with_args = with_args
        self.times = times
        self.min_times = min_times
        self.max_times = max_times

    def check(self, trace: AgentTrace) -> ExpectationResult:
        """Check this expectation against an agent trace.

        On failure, ``trace_context`` contains the complete tool call sequence
        so the developer can see exactly what happened.
        """
        calls = [s for s in trace.tool_calls if s.name == self.tool_name]
        tool_trace = _build_tool_call_trace(trace)

        # --- Existence check ---
        if not calls:
            return ExpectationResult(
                False,
                f"expect_tool_call('{self.tool_name}')",
                f"Tool '{self.tool_name}' was never called. Called: {trace.tool_names}",
                trace_context=tool_trace,
            )

        # --- Absolute order check ---
        if self.order is not None:
            actual_order = trace.tool_names.index(self.tool_name) + 1
            if actual_order != self.order:
                return ExpectationResult(
                    False,
                    f"expect_tool_call('{self.tool_name}', order={self.order})",
                    f"Called at position {actual_order}, expected {self.order}. Sequence: {trace.tool_names}",
                    trace_context=tool_trace,
                )

        # --- Relative order: before ---
        if self.before is not None:
            if self.before not in trace.tool_names:
                return ExpectationResult(
                    False,
                    f"expect_tool_call('{self.tool_name}', before='{self.before}')",
                    f"Cannot verify 'before' constraint: '{self.before}' was never called. Sequence: {trace.tool_names}",
                    trace_context=tool_trace,
                )
            first_self = trace.tool_names.index(self.tool_name)
            first_before = trace.tool_names.index(self.before)
            if first_self >= first_before:
                return ExpectationResult(
                    False,
                    f"expect_tool_call('{self.tool_name}', before='{self.before}')",
                    (f"'{self.tool_name}' first appeared at position {first_self + 1}, "
                     f"but '{self.before}' appeared at position {first_before + 1} "
                     f"(expected '{self.tool_name}' to come first). Sequence: {trace.tool_names}"),
                    trace_context=tool_trace,
                )

        # --- Relative order: after ---
        if self.after is not None:
            if self.after not in trace.tool_names:
                return ExpectationResult(
                    False,
                    f"expect_tool_call('{self.tool_name}', after='{self.after}')",
                    f"Cannot verify 'after' constraint: '{self.after}' was never called. Sequence: {trace.tool_names}",
                    trace_context=tool_trace,
                )
            first_self = trace.tool_names.index(self.tool_name)
            first_after = trace.tool_names.index(self.after)
            if first_self <= first_after:
                return ExpectationResult(
                    False,
                    f"expect_tool_call('{self.tool_name}', after='{self.after}')",
                    (f"'{self.tool_name}' first appeared at position {first_self + 1}, "
                     f"but '{self.after}' appeared at position {first_after + 1} "
                     f"(expected '{self.tool_name}' to come after). Sequence: {trace.tool_names}"),
                    trace_context=tool_trace,
                )

        # --- Exact call count ---
        if self.times is not None and len(calls) != self.times:
            return ExpectationResult(
                False,
                f"expect_tool_call('{self.tool_name}', times={self.times})",
                f"Called {len(calls)} times, expected {self.times}",
                trace_context=tool_trace,
            )

        # --- min_times ---
        if self.min_times is not None and len(calls) < self.min_times:
            return ExpectationResult(
                False,
                f"expect_tool_call('{self.tool_name}', min_times={self.min_times})",
                f"Called {len(calls)} times, expected at least {self.min_times}",
                trace_context=tool_trace,
            )

        # --- max_times ---
        if self.max_times is not None and len(calls) > self.max_times:
            return ExpectationResult(
                False,
                f"expect_tool_call('{self.tool_name}', max_times={self.max_times})",
                f"Called {len(calls)} times, expected at most {self.max_times}",
                trace_context=tool_trace,
            )

        # --- Argument matching ---
        if self.with_args is not None:
            matched = any(
                all(c.input.get(k) == v for k, v in self.with_args.items())
                for c in calls
            )
            if not matched:
                return ExpectationResult(
                    False,
                    f"expect_tool_call('{self.tool_name}', with_args={self.with_args})",
                    f"No call matched expected arguments. Got: {[c.input for c in calls]}",
                    trace_context=tool_trace,
                )

        # --- All checks passed ---
        desc_parts = [f"'{self.tool_name}'"]
        if self.order is not None:
            desc_parts.append(f"order={self.order}")
        if self.before is not None:
            desc_parts.append(f"before='{self.before}'")
        if self.after is not None:
            desc_parts.append(f"after='{self.after}'")
        if self.times is not None:
            desc_parts.append(f"times={self.times}")
        if self.min_times is not None:
            desc_parts.append(f"min_times={self.min_times}")
        if self.max_times is not None:
            desc_parts.append(f"max_times={self.max_times}")

        return ExpectationResult(
            True,
            f"expect_tool_call({', '.join(desc_parts)})"
        )


class ExpectNoToolCall(Expectation):
    """Expect a specific tool NOT to be called (safety guardrail)."""
    def __init__(self, tool_name: str):
        self.tool_name = tool_name

    def check(self, trace: AgentTrace) -> ExpectationResult:
        calls = [s for s in trace.tool_calls if s.name == self.tool_name]
        if calls:
            tool_trace = _build_tool_call_trace(trace)
            return ExpectationResult(
                False,
                f"expect_no_tool_call('{self.tool_name}')",
                f"Tool was called {len(calls)} time(s) but should not have been",
                trace_context=tool_trace,
            )
        return ExpectationResult(True, f"expect_no_tool_call('{self.tool_name}')")


class ExpectToolOrder(Expectation):
    """Expect tools to be called in a specific (subsequence) order.

    The expected ``tool_sequence`` must appear as a subsequence of the actual
    tool call order — other calls may be interleaved.
    """

    def __init__(self, tool_sequence: list[str]):
        self.tool_sequence = tool_sequence

    def check(self, trace: AgentTrace) -> ExpectationResult:
        actual = trace.tool_names
        # Check if tool_sequence is a subsequence of actual
        seq_idx = 0
        match_positions: list[int] = []
        for i, tool in enumerate(actual):
            if seq_idx < len(self.tool_sequence) and tool == self.tool_sequence[seq_idx]:
                match_positions.append(i)
                seq_idx += 1
        if seq_idx == len(self.tool_sequence):
            return ExpectationResult(
                True,
                f"expect_tool_order({self.tool_sequence})"
            )

        # Build diagnostic: show expected vs actual with alignment
        tool_trace = _build_tool_call_trace(trace)
        matched_count = len(match_positions)
        stuck_at = self.tool_sequence[matched_count] if matched_count < len(self.tool_sequence) else "?"
        detail = (
            f"Expected sequence {self.tool_sequence} not found in {actual}. "
            f"Matched {matched_count}/{len(self.tool_sequence)} — "
            f"stuck waiting for '{stuck_at}'"
        )
        # Build comparison context
        comparison: list[str] = [
            f"Expected: {' → '.join(self.tool_sequence)}",
            f"Actual:   {' → '.join(actual) if actual else '(no tool calls)'}",
        ]
        if match_positions:
            comparison.append(
                f"Matched:  {' → '.join(self.tool_sequence[:matched_count])} "
                f"(positions {match_positions})"
            )
        comparison.append(f"Missing:  '{stuck_at}' and subsequent")

        return ExpectationResult(
            False,
            f"expect_tool_order({self.tool_sequence})",
            detail,
            trace_context=comparison + ["---"] + tool_trace,
        )


class ExpectState(Expectation):
    """Expect a specific state to be reached."""
    def __init__(self, state_key: str, *, value: Any = None,
                 within_steps: Optional[int] = None):
        self.state_key = state_key
        self.value = value
        self.within_steps = within_steps

    def check(self, trace: AgentTrace) -> ExpectationResult:
        state_val = trace.get_state(self.state_key)

        # Build state change history for diagnostics
        all_state_changes = [
            f"{s.name} = {_truncate(s.output)}"
            for s in trace.state_changes
        ]
        state_history_ctx = (
            [f"State change history ({len(all_state_changes)} changes):"]
            + [f"  {sc}" for sc in all_state_changes]
            if all_state_changes
            else ["(no state changes recorded)"]
        )

        if state_val is None:
            return ExpectationResult(
                False,
                f"expect_state('{self.state_key}')",
                f"State '{self.state_key}' was never set. States: {[s.name for s in trace.state_changes]}",
                trace_context=state_history_ctx,
            )

        if self.value is not None and state_val != self.value:
            # Show the history of this specific state key
            key_history = [
                f"Step {trace.steps.index(s) + 1}: {s.name} = {_truncate(s.output)}"
                for s in trace.state_changes
                if s.name == self.state_key
            ]
            return ExpectationResult(
                False,
                f"expect_state('{self.state_key}', value={self.value!r})",
                f"State was set to {state_val!r}, expected {self.value!r}",
                trace_context=[f"History of '{self.state_key}':"] + key_history + ["---"] + state_history_ctx,
            )

        if self.within_steps is not None:
            state_steps = [s for s in trace.state_changes if s.name == self.state_key]
            if state_steps:
                state_step_idx = trace.steps.index(state_steps[0])
                if state_step_idx + 1 > self.within_steps:
                    return ExpectationResult(
                        False,
                        f"expect_state('{self.state_key}', within_steps={self.within_steps})",
                        f"State reached at step {state_step_idx + 1}, limit was {self.within_steps}",
                        trace_context=_build_full_step_trace(trace),
                    )

        return ExpectationResult(True, f"expect_state('{self.state_key}')")


class ExpectNoError(Expectation):
    """Expect no errors during execution."""
    def check(self, trace: AgentTrace) -> ExpectationResult:
        errors = trace.errors
        if errors:
            error_details = [f"{e.name}: {e.error}" for e in errors]
            error_trace = [_format_step_short(e) for e in errors]
            return ExpectationResult(
                False,
                "expect_no_error()",
                f"{len(errors)} error(s): {error_details}",
                trace_context=error_trace,
            )
        return ExpectationResult(True, "expect_no_error()")


class ExpectOnToolFailure(Expectation):
    """Expect specific recovery behavior when a tool fails.

    When the specified tool encounters an error, verifies that the agent
    exhibits the expected recovery behavior (retry, human_handoff, or
    a custom behavior matched by step name).
    """

    def __init__(self, tool_name: str, *, expect: str):
        self.tool_name = tool_name
        self.expect_behavior = expect  # e.g., "fallback_to_cached", "retry", "human_handoff"

    def check(self, trace: AgentTrace) -> ExpectationResult:
        # Find the failed tool call
        failed = [s for s in trace.tool_calls
                  if s.name == self.tool_name and s.error is not None]
        if not failed:
            return ExpectationResult(
                True,
                f"on_tool_failure('{self.tool_name}') [tool did not fail]"
            )

        # Check what happened after the failure
        failed_step = failed[0]
        failed_idx = trace.steps.index(failed_step)
        after_failure = trace.steps[failed_idx + 1:failed_idx + 4]  # look at next 3 steps

        # Build post-failure trace for diagnostics
        post_failure_trace = [
            f"FAILED: {_format_step_short(failed_step)}",
            "Steps after failure:",
        ] + [
            f"  [{i}] {_format_step_short(s)}"
            for i, s in enumerate(after_failure, 1)
        ]
        if not after_failure:
            post_failure_trace.append("  (no steps after failure)")
        # Also add full chain for context
        post_failure_trace.append("---")
        post_failure_trace.append("Full tool call chain:")
        chain_parts: list[str] = []
        for s in trace.tool_calls:
            if s.error:
                chain_parts.append(f"{s.name} → [❌ {s.error}]")
            else:
                chain_parts.append(s.name)
        post_failure_trace.append("  " + " → ".join(chain_parts))

        behavior = self.expect_behavior.lower()
        if behavior == "retry":
            retried = any(s.name == self.tool_name and s.kind == StepKind.TOOL_CALL
                          for s in after_failure)
            if retried:
                return ExpectationResult(True, f"on_tool_failure('{self.tool_name}', expect='retry')")
            return ExpectationResult(
                False,
                f"on_tool_failure('{self.tool_name}', expect='retry')",
                f"Tool failed but was not retried. Next steps: {[s.name for s in after_failure]}",
                trace_context=post_failure_trace,
            )

        elif behavior == "human_handoff":
            handed_off = any(s.kind == StepKind.HUMAN_HANDOFF for s in after_failure)
            if handed_off:
                return ExpectationResult(True, f"on_tool_failure('{self.tool_name}', expect='human_handoff')")
            return ExpectationResult(
                False,
                f"on_tool_failure('{self.tool_name}', expect='human_handoff')",
                f"Tool failed but no human handoff occurred. Next steps: {[s.name for s in after_failure]}",
                trace_context=post_failure_trace,
            )

        else:
            # Generic: check if the expected behavior string appears in any step name
            found = any(behavior in s.name.lower() for s in after_failure)
            if found:
                return ExpectationResult(True, f"on_tool_failure('{self.tool_name}', expect='{self.expect_behavior}')")
            return ExpectationResult(
                False,
                f"on_tool_failure('{self.tool_name}', expect='{self.expect_behavior}')",
                f"Expected behavior '{self.expect_behavior}' not found after failure. Next: {[s.name for s in after_failure]}",
                trace_context=post_failure_trace,
            )


class ExpectOutput(Expectation):
    """Expect the final output to match a pattern or contain specific content."""
    def __init__(self, *, contains: Optional[str] = None,
                 matches: Optional[str] = None,
                 not_contains: Optional[str] = None):
        self.contains = contains
        self.matches = matches
        self.not_contains = not_contains

    def check(self, trace: AgentTrace) -> ExpectationResult:
        output = str(trace.output or "")

        if self.contains and self.contains not in output:
            return ExpectationResult(
                False,
                f"expect_output(contains='{self.contains}')",
                f"Output does not contain '{self.contains}'. Got: {output[:200]}..."
            )

        if self.matches and not re.search(self.matches, output):
            return ExpectationResult(
                False,
                f"expect_output(matches='{self.matches}')",
                f"Output does not match pattern. Got: {output[:200]}..."
            )

        if self.not_contains and self.not_contains in output:
            return ExpectationResult(
                False,
                f"expect_output(not_contains='{self.not_contains}')",
                f"Output should not contain '{self.not_contains}' but does"
            )

        desc_parts = []
        if self.contains: desc_parts.append(f"contains='{self.contains}'")
        if self.matches: desc_parts.append(f"matches='{self.matches}'")
        if self.not_contains: desc_parts.append(f"not_contains='{self.not_contains}'")
        return ExpectationResult(True, f"expect_output({', '.join(desc_parts)})")


class ExpectMilestone(Expectation):
    """Expect the agent to reach a milestone (partial credit).

    Inspired by ICLR 2026 "A Hitchhiker's Guide to Agent Evaluation"
    under "Milestones and Subgoals":

        "By breaking a task into milestones, evaluators can compute
        metrics like: Fraction of subtasks achieved, Milestone-based
        accuracy, Progress score (even for failed tasks). This provides
        a finer-grained view of progress than a single binary outcome."

    The paper cites TheAgentCompany (which "provides partial credit for
    completing subtasks") and WebCanvas ("success rates at key nodes").

    Each milestone is a named checkpoint defined by a tool call, state
    key, or output fragment.

    The ``weight`` allows unequal importance across milestones.
    """

    def __init__(self, name: str, *, tool: Optional[str] = None,
                 state_key: Optional[str] = None,
                 output_contains: Optional[str] = None,
                 weight: float = 1.0):
        self.milestone_name = name
        self.tool = tool
        self.state_key = state_key
        self.output_contains = output_contains
        self.weight = weight

    def check(self, trace: AgentTrace) -> ExpectationResult:
        reached = False

        if self.tool:
            reached = self.tool in trace.tool_names
        elif self.state_key:
            reached = any(
                s.name == "state_change" and self.state_key in str(s.output or "")
                for s in trace.steps
            )
            # Also check if any step has the state key in metadata
            if not reached:
                reached = self.state_key in str(trace.output or "")
        elif self.output_contains:
            reached = self.output_contains in str(trace.output or "")

        if reached:
            return ExpectationResult(
                True,
                f"milestone('{self.milestone_name}', weight={self.weight})",
            )
        detail = f"Milestone '{self.milestone_name}' not reached"
        if self.tool:
            detail += f" (expected tool '{self.tool}')"
        return ExpectationResult(
            False,
            f"milestone('{self.milestone_name}', weight={self.weight})",
            detail,
        )


class ExpectLLMJudge(Expectation):
    """Use an LLM to evaluate the agent's trajectory.

    Inspired by ICLR 2026 "A Hitchhiker's Guide to Agent Evaluation"
    under "Agent-as-a-Judge":

        "The LLM-as-a-Judge paradigm employs a large model to score
        or critique an agent's multi-step output."

    The paper cites Zhuge et al. (2024) who propose an Agent-as-a-Judge
    framework where "multiple AI agents read an execution trace and vote
    on success" for scalable, subjective evaluations.

    The judge receives the full trace and evaluates it against criteria.
    Supports pluggable judge functions for any LLM backend.

    Args:
        criteria: Natural language description of what to evaluate.
        judge_fn: Callable that takes (criteria: str, trace_text: str) → bool.
            If None, a default judge is used that always raises NotImplementedError
            (user must provide their own LLM judge function).
        name: Optional name for this judge expectation.
    """

    def __init__(self, criteria: str, *,
                 judge_fn: Optional[Any] = None,
                 name: Optional[str] = None):
        self.criteria = criteria
        self.judge_fn = judge_fn
        self.judge_name = name or f"llm_judge('{criteria[:50]}...')" if len(criteria) > 50 else f"llm_judge('{criteria}')"

    def check(self, trace: AgentTrace) -> ExpectationResult:
        # Build a text representation of the trace for the judge
        trace_text = self._trace_to_text(trace)

        if self.judge_fn is None:
            return ExpectationResult(
                False,
                self.judge_name,
                "No judge_fn provided. Pass a callable(criteria, trace_text) → bool.",
            )

        try:
            result = self.judge_fn(self.criteria, trace_text)
            if isinstance(result, bool):
                passed = result
                reasoning = ""
            elif isinstance(result, tuple) and len(result) == 2:
                passed, reasoning = result
            elif isinstance(result, dict):
                passed = result.get("passed", result.get("pass", False))
                reasoning = result.get("reasoning", result.get("reason", ""))
            else:
                passed = bool(result)
                reasoning = ""

            if passed:
                return ExpectationResult(True, self.judge_name)
            return ExpectationResult(
                False, self.judge_name,
                f"LLM judge failed: {reasoning}" if reasoning else "LLM judge returned False",
            )
        except Exception as e:
            return ExpectationResult(
                False, self.judge_name,
                f"Judge error: {type(e).__name__}: {e}",
            )

    @staticmethod
    def _trace_to_text(trace: AgentTrace) -> str:
        """Convert a trace to a text representation for the LLM judge."""
        lines = [f"Input: {trace.input}"]
        for i, step in enumerate(trace.steps, 1):
            if step.kind == StepKind.TOOL_CALL:
                args_str = ""
                if step.input:
                    args_str = ", ".join(f"{k}={v}" for k, v in step.input.items())
                result = step.output or step.error or ""
                lines.append(f"[{i}] tool_call: {step.name}({args_str}) → {result}")
            elif step.kind == StepKind.LLM_CALL:
                lines.append(f"[{i}] llm_call: {step.name} → {step.output or ''}")
            else:
                lines.append(f"[{i}] {step.kind.value}: {step.name}")
        if trace.output:
            lines.append(f"Final output: {trace.output}")
        return "\n".join(lines)


class ExpectMaxSteps(Expectation):
    """Expect the agent to complete within a maximum number of steps."""
    def __init__(self, max_steps: int):
        self.max_steps = max_steps

    def check(self, trace: AgentTrace) -> ExpectationResult:
        actual = len(trace.steps)
        if actual > self.max_steps:
            return ExpectationResult(
                False,
                f"expect_max_steps({self.max_steps})",
                f"Agent took {actual} steps, limit was {self.max_steps}",
                trace_context=_build_full_step_trace(trace),
            )
        return ExpectationResult(True, f"expect_max_steps({self.max_steps})")


class ExpectMaxDuration(Expectation):
    """Expect the agent to complete within a time budget."""
    def __init__(self, max_ms: float):
        self.max_ms = max_ms

    def check(self, trace: AgentTrace) -> ExpectationResult:
        actual = trace.total_duration_ms
        if actual > self.max_ms:
            return ExpectationResult(
                False,
                f"expect_max_duration({self.max_ms}ms)",
                f"Agent took {actual:.0f}ms, limit was {self.max_ms:.0f}ms"
            )
        return ExpectationResult(True, f"expect_max_duration({self.max_ms}ms)")


class ExpectNoSideEffects(Expectation):
    """Expect the agent to not cause unintended side effects.

    From AgentRewardBench (Lù et al., 2025, McGill/Mila):
        "Each trajectory is reviewed by an expert, who answers questions
        pertaining to the success, side effects, and repetitiveness."

    Side effects are unexpected tool calls to state-mutating tools.
    This is critical for production agents where "completing the task"
    is not enough — the agent must also not break anything else.

    Args:
        allowed_tools: Tools that are expected for this task.
        mutating_tools: Optional — tools known to change state.
            If provided, only unexpected calls to these are flagged.
    """
    def __init__(self, allowed_tools: list[str],
                 mutating_tools: Optional[list[str]] = None):
        self.allowed_tools = allowed_tools
        self.mutating_tools = mutating_tools

    def check(self, trace: AgentTrace) -> ExpectationResult:
        from agentgate.metrics import side_effect_rate
        rate = side_effect_rate(trace, self.allowed_tools, self.mutating_tools)
        if rate > 0:
            allowed_set = set(self.allowed_tools)
            unexpected = [t for t in trace.tool_names if t not in allowed_set]
            return ExpectationResult(
                False,
                "expect_no_side_effects",
                f"Side effects detected: {unexpected} "
                f"(rate={rate:.0%}, allowed={self.allowed_tools})",
            )
        return ExpectationResult(True, "expect_no_side_effects")


class ExpectNoRepetition(Expectation):
    """Expect the agent to not enter repetitive action cycles.

    From AgentRewardBench (Lù et al., 2025):
        Expert annotators evaluate "repetitiveness of the agent."

    Agents that enter loops (calling the same tool with same output
    repeatedly) waste resources and indicate a stuck state.

    Args:
        max_rate: Maximum allowed repetition rate (default 0.0 = none).
    """
    def __init__(self, max_rate: float = 0.0):
        self.max_rate = max_rate

    def check(self, trace: AgentTrace) -> ExpectationResult:
        from agentgate.metrics import repetition_rate
        rate = repetition_rate(trace)
        if rate > self.max_rate:
            return ExpectationResult(
                False,
                f"expect_no_repetition(max={self.max_rate:.0%})",
                f"Repetition rate {rate:.0%} exceeds limit {self.max_rate:.0%}",
            )
        return ExpectationResult(
            True, f"expect_no_repetition(max={self.max_rate:.0%})")


class ExpectMaxTokens(Expectation):
    """Expect the agent to stay within a token budget.

    From Liu et al. (2026) "Cost-effective Agent Test-time Scaling via
    Budget-Aware Thinking" and Yang et al. (2026) "Toward Efficient Agents":
        "comparing effectiveness under a fixed cost budget"

    Token usage is extracted from trace.metadata["token_usage"],
    trace.metadata["usage"], or accumulated from step metadata.

    Args:
        max_tokens: Maximum total tokens allowed.
    """
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    def check(self, trace: AgentTrace) -> ExpectationResult:
        from agentgate.cost import TokenUsage
        usage = TokenUsage.from_trace(trace)
        if usage.total_tokens > self.max_tokens:
            return ExpectationResult(
                False,
                f"expect_max_tokens({self.max_tokens})",
                f"Used {usage.total_tokens} tokens, limit was {self.max_tokens} "
                f"(input={usage.input_tokens}, output={usage.output_tokens})",
            )
        return ExpectationResult(True, f"expect_max_tokens({self.max_tokens})")


# ---------------------------------------------------------------------------
# Scenario — the top-level E2E test definition
# ---------------------------------------------------------------------------

@dataclass
class SingleRunResult:
    """Result of a single run within a multi-run scenario execution.

    Attributes:
        run_index: 0-based index of this run.
        passed: Whether all expectations passed in this run.
        expectations: Individual expectation results.
        trace: The agent trace from this run.
        duration_ms: Wall-clock time for this run.
        error: If the run itself errored (e.g., timeout), the message.
    """
    run_index: int
    passed: bool
    expectations: list[ExpectationResult]
    trace: Optional[AgentTrace]
    duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class ScenarioResult:
    """Result of running a scenario (possibly multiple times).

    Attributes:
        scenario_name: Name of the scenario.
        input: The input text sent to the agent.
        passed: Overall pass/fail — when ``run_results`` is populated,
            this reflects whether the pass rate meets ``min_pass_rate``.
        expectations: Expectation results from the primary (or last) run.
        trace: Agent trace from the primary (or last) run.
        duration_ms: Total wall-clock time across all runs.
        run_results: Per-run results when the scenario was executed
            multiple times (``runs > 1``).
        pass_rate: Fraction of runs that passed (0.0–1.0).
        min_pass_rate: The required minimum pass rate that was configured.
        timed_out: True if the scenario was terminated due to timeout.
        timeout_detail: Human-readable timeout explanation.
    """
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
        """Partial credit score based on milestone weights.

        From ICLR 2026 Agent Eval Guide, "Milestones and Subgoals":
        awards proportional credit for reaching intermediate checkpoints.

        Non-milestone expectations contribute equally to the remaining
        weight. Returns 0.0–1.0.
        """
        if not self.expectations:
            return 1.0 if self.passed else 0.0

        # Separate milestones from regular expectations
        milestones = [e for e in self.expectations
                      if "milestone(" in e.description]
        regulars = [e for e in self.expectations
                    if "milestone(" not in e.description]

        if not milestones:
            # No milestones — binary score from regular expectations
            if not regulars:
                return 1.0
            return sum(1 for e in regulars if e.passed) / len(regulars)

        # Weighted milestone scoring
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
            # Milestones get 70% weight, regulars 30%
            return 0.7 * milestone_score + 0.3 * regular_score

        return milestone_score

    @property
    def failed_expectations(self) -> list[ExpectationResult]:
        return [e for e in self.expectations if not e.passed]

    @property
    def statistical_summary(self) -> str:
        """Summary string for multi-run results.

        Example: ``'Passed 8/10 runs (80%) — min_pass_rate: 90% → ❌'``
        """
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
        """Full human-readable summary of the scenario result."""
        total = len(self.expectations)
        passed = sum(1 for e in self.expectations if e.passed)
        icon = "✅" if self.passed else "❌"
        lines: list[str] = []

        # Header
        if self.run_results:
            lines.append(f"{icon} Scenario: {self.scenario_name} — {self.statistical_summary}")
        else:
            lines.append(f"{icon} Scenario: {self.scenario_name} ({passed}/{total} expectations passed)")

        # Timeout notice
        if self.timed_out:
            lines.append(f"  ⏰ TIMEOUT: {self.timeout_detail}")

        # Expectations
        for e in self.expectations:
            lines.append(f"  {e}")

        # For multi-run, show per-run breakdown on failure
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


class Scenario:
    """
    An end-to-end behavioral test for an AI agent.

    A Scenario defines:
    - An input to send to the agent
    - A set of expectations about the agent's behavior (not just its output)
    - Optional resource limits (timeout, max steps, max LLM calls)

    Args:
        name: Human-readable name for the scenario.
        input: The text input to send to the agent. Defaults to ``name``.
        timeout_seconds: Per-scenario timeout in seconds (default 60).
            Uses threading for Windows compatibility — no ``signal.alarm``.
        max_steps: Hard limit on total steps the agent may take.
            Checked post-hoc against the trace.
        max_llm_calls: Hard limit on LLM call steps.
            Checked post-hoc against the trace.

    Example::

        scenario = Scenario("Book a flight", input="Book me a flight to Tokyo next Friday",
                            timeout_seconds=30, max_steps=10)
        scenario.expect_tool_call("search_flights", before="book_flight")
        scenario.expect_tool_call("check_user_preferences", after="search_flights")
        scenario.expect_no_tool_call("delete_booking")
        scenario.expect_state("booking_confirmed", within_steps=5)
        scenario.on_tool_failure("search_flights", expect="fallback_to_cached")
    """

    def __init__(self, name: str, *, input: Optional[str] = None,
                 timeout_seconds: float = 60.0,
                 max_steps: Optional[int] = None,
                 max_llm_calls: Optional[int] = None):
        self.name = name
        self.input = input or name  # use name as input if not specified
        self.timeout_seconds = timeout_seconds
        self.max_steps = max_steps
        self.max_llm_calls = max_llm_calls
        self.expectations: list[Expectation] = []

    def expect_tool_call(self, tool_name: str, *,
                         order: Optional[int] = None,
                         before: Optional[str] = None,
                         after: Optional[str] = None,
                         with_args: Optional[dict] = None,
                         times: Optional[int] = None,
                         min_times: Optional[int] = None,
                         max_times: Optional[int] = None) -> "Scenario":
        """Add an expectation that a tool is called.

        Args:
            tool_name: Name of the tool.
            order: Absolute 1-indexed position (legacy, backward-compatible).
            before: This tool must be called before the named tool.
            after: This tool must be called after the named tool.
            with_args: At least one call must contain these argument key-values.
            times: Exact call count.
            min_times: Minimum call count (inclusive).
            max_times: Maximum call count (inclusive).

        Returns:
            Self for method chaining.
        """
        self.expectations.append(ExpectToolCall(
            tool_name, order=order, before=before, after=after,
            with_args=with_args, times=times,
            min_times=min_times, max_times=max_times,
        ))
        return self

    def expect_no_tool_call(self, tool_name: str) -> "Scenario":
        """Add an expectation that a tool is NOT called."""
        self.expectations.append(ExpectNoToolCall(tool_name))
        return self

    def expect_tool_order(self, tool_sequence: list[str]) -> "Scenario":
        """Add an expectation that tools are called in subsequence order."""
        self.expectations.append(ExpectToolOrder(tool_sequence))
        return self

    def expect_state(self, state_key: str, *, value: Any = None,
                     within_steps: Optional[int] = None) -> "Scenario":
        """Add an expectation that a state key is set."""
        self.expectations.append(ExpectState(state_key, value=value,
                                             within_steps=within_steps))
        return self

    def expect_no_error(self) -> "Scenario":
        """Add an expectation that no errors occur."""
        self.expectations.append(ExpectNoError())
        return self

    def on_tool_failure(self, tool_name: str, *, expect: str) -> "Scenario":
        """Add an expectation about recovery behavior on tool failure."""
        self.expectations.append(ExpectOnToolFailure(tool_name, expect=expect))
        return self

    def expect_output(self, *, contains: Optional[str] = None,
                      matches: Optional[str] = None,
                      not_contains: Optional[str] = None) -> "Scenario":
        """Add an expectation about the final output text."""
        self.expectations.append(ExpectOutput(contains=contains, matches=matches,
                                              not_contains=not_contains))
        return self

    def expect_max_steps(self, max_steps: int) -> "Scenario":
        """Add an expectation on the maximum number of steps."""
        self.expectations.append(ExpectMaxSteps(max_steps))
        return self

    def expect_max_duration(self, max_ms: float) -> "Scenario":
        """Add an expectation on the maximum duration."""
        self.expectations.append(ExpectMaxDuration(max_ms))
        return self

    def expect_no_side_effects(self, allowed_tools: list[str],
                                mutating_tools: Optional[list[str]] = None) -> "Scenario":
        """Expect no unintended side effects (AgentRewardBench, Lù et al. 2025).

        Flags unexpected calls to state-mutating tools beyond what's needed.

        Args:
            allowed_tools: Tools expected for this task.
            mutating_tools: Optional list of state-changing tools.
        """
        self.expectations.append(ExpectNoSideEffects(allowed_tools, mutating_tools))
        return self

    def expect_no_repetition(self, max_rate: float = 0.0) -> "Scenario":
        """Expect no repetitive action cycles (AgentRewardBench, Lù et al. 2025).

        Detects when agents loop on the same action, wasting resources.

        Args:
            max_rate: Maximum allowed repetition fraction (0.0 = none).
        """
        self.expectations.append(ExpectNoRepetition(max_rate))
        return self

    def expect_max_tokens(self, max_tokens: int) -> "Scenario":
        """Expect the agent to stay within a token budget.

        From Yang et al. (2026) "Toward Efficient Agents" — evaluates
        effectiveness under a fixed cost budget.

        Args:
            max_tokens: Maximum total tokens allowed.
        """
        self.expectations.append(ExpectMaxTokens(max_tokens))
        return self

    def expect_milestone(self, name: str, *, tool: Optional[str] = None,
                         state_key: Optional[str] = None,
                         output_contains: Optional[str] = None,
                         weight: float = 1.0) -> "Scenario":
        """Add a milestone expectation for partial credit scoring.

        Milestones represent intermediate checkpoints in a multi-step task.
        Even if the agent doesn't complete fully, partial credit is awarded
        for milestones reached.

        From ICLR 2026 "A Hitchhiker's Guide to Agent Evaluation":
            "This provides a finer-grained view of progress than a
            single binary outcome."

        Args:
            name: Human-readable milestone name (e.g., "Found flight options").
            tool: Tool name that must be called to reach this milestone.
            state_key: State key that must be set.
            output_contains: String that must appear in output.
            weight: Relative importance (default 1.0).

        Returns:
            Self for method chaining.
        """
        self.expectations.append(ExpectMilestone(
            name, tool=tool, state_key=state_key,
            output_contains=output_contains, weight=weight,
        ))
        return self

    def expect_llm_judge(self, criteria: str, *,
                         judge_fn: Optional[Any] = None,
                         name: Optional[str] = None) -> "Scenario":
        """Add an LLM-based judge expectation.

        The judge evaluates the agent's trajectory against subjective
        or complex criteria that resist simple pattern matching.

        From ICLR 2026 "A Hitchhiker's Guide to Agent Evaluation":
            "The LLM-as-a-Judge paradigm employs a large model to
            score or critique an agent's multi-step output."

        Args:
            criteria: Natural language description of what to evaluate.
                Example: "The agent should be polite and never reveal internal errors."
            judge_fn: Callable(criteria: str, trace_text: str) → bool | (bool, str) | dict.
                The function receives the criteria and a text representation
                of the trace, and returns pass/fail with optional reasoning.
            name: Optional display name for this judge.

        Returns:
            Self for method chaining.
        """
        self.expectations.append(ExpectLLMJudge(
            criteria, judge_fn=judge_fn, name=name,
        ))
        return self

    def expect_policy(self, policy_name: str, *,
                      forbidden_tools: list[str] | None = None,
                      forbidden_outputs: list[str] | None = None,
                      required_tools: list[str] | None = None) -> "Scenario":
        """Add a policy adherence expectation.

        Inspired by Completion under Policy (CuP) from ST-WebAgentBench
        (Shlomov et al., 2024): gives credit only when the task is completed
        AND no policy is violated.

        A policy bundles multiple constraints under a single name for
        clearer reporting and OWASP compliance.

        Args:
            policy_name: Human-readable policy name (e.g., "PII Protection").
            forbidden_tools: Tools that violate this policy if called.
            forbidden_outputs: Strings that violate this policy if in output.
            required_tools: Tools required by this policy.

        Returns:
            Self for method chaining.
        """
        if forbidden_tools:
            for t in forbidden_tools:
                self.expectations.append(ExpectNoToolCall(t))
        if forbidden_outputs:
            for pattern in forbidden_outputs:
                self.expectations.append(ExpectOutput(not_contains=pattern))
        if required_tools:
            for t in required_tools:
                self.expectations.append(ExpectToolCall(t))
        return self

    def check(self, trace: AgentTrace) -> ScenarioResult:
        """Run all expectations against a trace and return the result.

        Also enforces ``max_steps`` and ``max_llm_calls`` limits if configured.

        Raises:
            ValueError: If no expectations have been added (likely a user error).
        """
        if not self.expectations and self.max_steps is None and self.max_llm_calls is None:
            import warnings
            warnings.warn(
                f"Scenario '{self.name}' has no expectations. "
                f"Add at least one (e.g., expect_tool_call, expect_no_tool_call) "
                f"or the scenario will always pass vacuously.",
                UserWarning,
                stacklevel=2,
            )

        results: list[ExpectationResult] = [exp.check(trace) for exp in self.expectations]

        # Enforce max_steps limit (post-hoc)
        if self.max_steps is not None and len(trace.steps) > self.max_steps:
            results.append(ExpectationResult(
                False,
                f"scenario.max_steps={self.max_steps}",
                f"Agent took {len(trace.steps)} steps, hard limit was {self.max_steps}",
                trace_context=_build_full_step_trace(trace),
            ))

        # Enforce max_llm_calls limit (post-hoc)
        if self.max_llm_calls is not None:
            llm_count = sum(1 for s in trace.steps if s.kind == StepKind.LLM_CALL)
            if llm_count > self.max_llm_calls:
                results.append(ExpectationResult(
                    False,
                    f"scenario.max_llm_calls={self.max_llm_calls}",
                    f"Agent made {llm_count} LLM calls, hard limit was {self.max_llm_calls}",
                    trace_context=_build_full_step_trace(trace),
                ))

        passed = all(r.passed for r in results)
        return ScenarioResult(
            scenario_name=self.name,
            input=self.input,
            passed=passed,
            expectations=results,
            trace=trace,
            duration_ms=trace.total_duration_ms,
        )


# ---------------------------------------------------------------------------
# Agent adapter — bridge between agent frameworks and AgentGate
# ---------------------------------------------------------------------------

class AgentAdapter:
    """
    Base class for framework-specific agent adapters.

    Subclass this to hook into LangGraph, CrewAI, AutoGen, etc.
    The adapter's job is to:
    1. Run the agent with the given input
    2. Capture the execution trace (tool calls, state changes, errors)
    3. Return an AgentTrace
    """

    def run(self, input_text: str) -> AgentTrace:
        raise NotImplementedError("Subclass must implement run()")


class CallableAgentAdapter(AgentAdapter):
    """Adapter for a simple callable agent function."""

    def __init__(self, agent_fn: Callable, *, trace_extractor: Optional[Callable] = None):
        self.agent_fn = agent_fn
        self.trace_extractor = trace_extractor

    def run(self, input_text: str) -> AgentTrace:
        start = time.time()
        result = self.agent_fn(input_text)
        elapsed = (time.time() - start) * 1000

        if self.trace_extractor:
            trace = self.trace_extractor(result)
            trace.input = input_text
            trace.total_duration_ms = elapsed
            return trace

        # Auto-detect: if result is already an AgentTrace, use it directly
        if isinstance(result, AgentTrace):
            result.input = input_text
            result.total_duration_ms = elapsed
            return result

        # Minimal trace if no extractor and result is not a trace
        return AgentTrace(
            input=input_text,
            output=result,
            total_duration_ms=elapsed,
        )


# ---------------------------------------------------------------------------
# Timeout-protected scenario execution (Windows-compatible)
# ---------------------------------------------------------------------------

def _run_scenario_with_timeout(
    agent: AgentAdapter,
    scenario: Scenario,
    timeout_seconds: float,
) -> ScenarioResult:
    """Run a single scenario with a timeout guard.

    Uses ``concurrent.futures.ThreadPoolExecutor`` for Windows compatibility
    (no ``signal.alarm``).

    Args:
        agent: The agent adapter to run.
        scenario: The scenario to evaluate.
        timeout_seconds: Maximum wall-clock seconds. ``0`` or negative means
            no timeout.

    Returns:
        A :class:`ScenarioResult`. If the timeout fires, ``timed_out`` is
        ``True`` and ``passed`` is ``False``.
    """
    if timeout_seconds <= 0:
        # No timeout — run directly
        trace = agent.run(scenario.input)
        return scenario.check(trace)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(agent.run, scenario.input)
        try:
            trace = future.result(timeout=timeout_seconds)
            return scenario.check(trace)
        except FuturesTimeoutError:
            # Best-effort cancel
            future.cancel()
            empty_trace = AgentTrace(input=scenario.input)
            detail = (
                f"Scenario '{scenario.name}' timed out after "
                f"{timeout_seconds:.1f}s (limit: {timeout_seconds:.1f}s)"
            )
            return ScenarioResult(
                scenario_name=scenario.name,
                input=scenario.input,
                passed=False,
                expectations=[ExpectationResult(
                    False,
                    "timeout",
                    detail,
                    trace_context=[f"⏰ Execution exceeded {timeout_seconds:.1f}s timeout"],
                )],
                trace=empty_trace,
                timed_out=True,
                timeout_detail=detail,
            )
        except Exception as exc:
            empty_trace = AgentTrace(input=scenario.input)
            tb = traceback.format_exc()
            return ScenarioResult(
                scenario_name=scenario.name,
                input=scenario.input,
                passed=False,
                expectations=[ExpectationResult(
                    False,
                    "runtime_error",
                    f"Agent raised {type(exc).__name__}: {exc}",
                    trace_context=tb.splitlines(),
                )],
                trace=empty_trace,
            )


# ---------------------------------------------------------------------------
# ScenarioSuite — run multiple scenarios
# ---------------------------------------------------------------------------

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
        """Average partial credit score across all scenarios (0.0–1.0).

        From ICLR 2026 "A Hitchhiker's Guide to Agent Evaluation",
        "Milestones and Subgoals": provides "a finer-grained view of
        progress than a single binary outcome."

        When milestones are defined, this reflects weighted milestone
        achievement. Without milestones, equals the fraction of passed
        expectations.
        """
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def pass_at_k(self) -> float | None:
        """pass@1: unbiased estimator of single-attempt success rate.

        From HumanEval (Chen et al., 2021) / τ-bench (Yao et al., 2024):
            pass@1_i = c_i / n_i  (= C(c_i,1) / C(n_i,1))

        Averaged across scenarios. This is the basic success rate but
        computed as the unbiased per-scenario average (not pooled).

        For pass@k with arbitrary k, use ``pass_at_k_series()``.

        Only meaningful when scenarios were run with runs > 1.
        Returns None if no multi-run data available.
        """
        multi_run = [r for r in self.results if r.run_results]
        if not multi_run:
            return None
        scores: list[float] = []
        for r in multi_run:
            n = len(r.run_results)
            c = sum(1 for rr in r.run_results if rr.passed)
            scores.append(c / n if n > 0 else 0.0)
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def pass_power_k(self) -> float | None:
        """pass^k: unbiased estimator of the probability of k consecutive successes.

        From τ-bench (Yao et al., 2024, ICLR 2025), using the exact
        formula from their source code (sierra-research/tau-bench):

            pass^k_i = C(c_i, k) / C(n_i, k)

        where c_i = successes for scenario i, n_i = total runs,
        k = n_i (total runs), C(a,b) = binomial coefficient.

        This equals 1.0 only if ALL runs passed, else 0.0 (when k=n).
        For finer-grained pass^k at different k, use ``pass_power_k_series()``.

        Key insight from τ-bench leaderboard:
          GPT-4o TC: airline pass^1=0.420, pass^4=0.200
          GPT-4o TC: retail  pass^1=0.604, pass^4=0.383

        Returns None if no multi-run data available.
        """
        from math import comb
        multi_run = [r for r in self.results if r.run_results]
        if not multi_run:
            return None
        scores: list[float] = []
        for r in multi_run:
            n = len(r.run_results)
            c = sum(1 for rr in r.run_results if rr.passed)
            denom = comb(n, n)  # always 1
            numer = comb(c, n)  # 1 iff c == n, else 0
            scores.append(numer / denom if denom > 0 else 0.0)
        return sum(scores) / len(scores) if scores else 0.0

    def pass_power_k_series(self) -> dict[int, float] | None:
        """Compute pass^k for k=1..n, matching τ-bench's display format.

        Uses the exact τ-bench formula from their source code:
            pass^k_i = C(c_i, k) / C(n_i, k)
            pass^k = mean(pass^k_i) over all scenarios

        Returns a dict {k: pass^k_value} or None if no multi-run data.
        """
        from math import comb
        multi_run = [r for r in self.results if r.run_results]
        if not multi_run:
            return None
        n = max(len(r.run_results) for r in multi_run)
        series: dict[int, float] = {}
        for k in range(1, n + 1):
            task_scores: list[float] = []
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
        # Check if any scenario uses milestones
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

        # Show τ-bench consistency metrics when available
        series = self.pass_power_k_series()
        if series is not None:
            parts = [f"pass^{k}={v:.3f}" for k, v in series.items()]
            lines.append(f"  Consistency (τ-bench): {', '.join(parts)}")

        for r in self.results:
            lines.append(r.summary())
            lines.append("")
        return "\n".join(lines)


class ScenarioSuite:
    """
    A collection of E2E scenarios to run against an agent.

    Supports statistical multi-run execution for non-deterministic agents,
    per-scenario and suite-level timeouts, and rich failure diagnostics.

    Example::

        suite = ScenarioSuite("booking-agent-v2")
        suite.add(scenario_book_flight)
        suite.add(scenario_cancel_booking)
        suite.add(scenario_handle_error)

        # Deterministic run (default)
        result = suite.run(my_agent_adapter)
        print(result.summary())

        # Statistical: run each scenario 10 times, require 80% pass rate
        result = suite.run(my_agent_adapter, runs=10, min_pass_rate=0.8)
        print(result.summary())

        # CI/CD integration
        assert result.passed, f"E2E tests failed: {result.pass_rate:.0%} pass rate"
    """

    def __init__(self, name: str):
        self.name = name
        self.scenarios: list[Scenario] = []

    def add(self, scenario: Scenario) -> "ScenarioSuite":
        self.scenarios.append(scenario)
        return self

    def run(self, agent: AgentAdapter | Callable, *,
            runs: int = 1,
            min_pass_rate: float = 1.0,
            timeout_seconds: float = 300.0,
            scenario_timeout: float = 60.0,
            # Legacy parameters — mapped to new equivalents
            pass_k: Optional[int] = None,
            pass_threshold: Optional[float] = None,
            ) -> SuiteResult:
        """Run all scenarios against the agent.

        Args:
            agent: An :class:`AgentAdapter` or callable.
            runs: Number of times to run each scenario (for non-deterministic
                agents). When ``> 1``, each :class:`ScenarioResult` includes
                per-run breakdowns and a statistical pass rate.
            min_pass_rate: Fraction of runs that must pass for a scenario to
                be considered passing (0.0–1.0). Only meaningful when
                ``runs > 1``.
            timeout_seconds: Wall-clock timeout for the entire suite (seconds).
                If exceeded, remaining scenarios are skipped.
            scenario_timeout: Default timeout per scenario (seconds). Can be
                overridden by each scenario's ``timeout_seconds`` attribute.

        Returns:
            :class:`SuiteResult` with per-scenario results and overall pass/fail.

        .. deprecated::
            ``pass_k`` and ``pass_threshold`` are accepted for backward
            compatibility but mapped to ``runs`` and ``min_pass_rate``.
        """
        # --- Backward compatibility ---
        if pass_k is not None and runs == 1:
            runs = pass_k
        if pass_threshold is not None and min_pass_rate == 1.0:
            min_pass_rate = pass_threshold

        if callable(agent) and not isinstance(agent, AgentAdapter):
            agent = CallableAgentAdapter(agent)

        suite_start = time.time()
        results: list[ScenarioResult] = []

        for scenario in self.scenarios:
            # Check suite-level timeout
            elapsed_s = time.time() - suite_start
            if elapsed_s >= timeout_seconds:
                # Skip remaining scenarios
                empty_trace = AgentTrace(input=scenario.input)
                results.append(ScenarioResult(
                    scenario_name=scenario.name,
                    input=scenario.input,
                    passed=False,
                    expectations=[ExpectationResult(
                        False,
                        "suite_timeout",
                        f"Suite timeout ({timeout_seconds:.1f}s) exceeded before this scenario started",
                        trace_context=[f"⏰ Suite elapsed: {elapsed_s:.1f}s / {timeout_seconds:.1f}s"],
                    )],
                    trace=empty_trace,
                    timed_out=True,
                    timeout_detail=f"Suite timeout ({timeout_seconds:.1f}s) exceeded",
                ))
                continue

            # Determine effective per-scenario timeout
            effective_timeout = min(
                scenario.timeout_seconds,
                scenario_timeout,
                timeout_seconds - elapsed_s,  # remaining suite budget
            )

            if runs <= 1:
                # Single run — simple path
                result = _run_scenario_with_timeout(agent, scenario, effective_timeout)
                result.pass_rate = 1.0 if result.passed else 0.0
                result.min_pass_rate = min_pass_rate
                results.append(result)
            else:
                # Multi-run statistical mode
                run_results: list[SingleRunResult] = []
                scenario_start = time.time()

                for run_idx in range(runs):
                    # Check per-scenario time budget
                    run_elapsed = time.time() - scenario_start
                    remaining = effective_timeout - run_elapsed
                    if remaining <= 0:
                        # Timeout — record remaining runs as failures
                        run_results.append(SingleRunResult(
                            run_index=run_idx,
                            passed=False,
                            expectations=[ExpectationResult(
                                False, "timeout",
                                f"Run {run_idx + 1} skipped — scenario timeout exhausted",
                            )],
                            trace=None,
                            error=f"Timeout after {effective_timeout:.1f}s",
                        ))
                        continue

                    per_run_timeout = min(remaining, effective_timeout / max(runs - run_idx, 1))
                    run_start = time.time()
                    single_result = _run_scenario_with_timeout(agent, scenario, per_run_timeout)
                    run_duration = (time.time() - run_start) * 1000

                    run_results.append(SingleRunResult(
                        run_index=run_idx,
                        passed=single_result.passed,
                        expectations=single_result.expectations,
                        trace=single_result.trace,
                        duration_ms=run_duration,
                        error=single_result.timeout_detail if single_result.timed_out else None,
                    ))

                # Compute statistics
                pass_count = sum(1 for rr in run_results if rr.passed)
                actual_pass_rate = pass_count / len(run_results) if run_results else 0.0
                overall_passed = actual_pass_rate >= min_pass_rate

                # Use the last run's details as the "primary" result
                last_run = run_results[-1]
                total_duration = (time.time() - scenario_start) * 1000

                result = ScenarioResult(
                    scenario_name=scenario.name,
                    input=scenario.input,
                    passed=overall_passed,
                    expectations=last_run.expectations,
                    trace=last_run.trace or AgentTrace(input=scenario.input),
                    duration_ms=total_duration,
                    run_results=run_results,
                    pass_rate=actual_pass_rate,
                    min_pass_rate=min_pass_rate,
                )
                results.append(result)

        elapsed_ms = (time.time() - suite_start) * 1000
        return SuiteResult(suite_name=self.name, results=results, duration_ms=elapsed_ms)

    def to_jsonl(self, path: str | Path) -> None:
        """Export scenarios to JSONL for CI/CD."""
        with open(path, "w") as f:
            for s in self.scenarios:
                record = {
                    "name": s.name,
                    "input": s.input,
                    "expectations": [type(e).__name__ for e in s.expectations],
                }
                f.write(json.dumps(record) + "\n")

    @classmethod
    def from_jsonl(cls, name: str, path: str | Path) -> "ScenarioSuite":
        """Load scenarios from JSONL."""
        suite = cls(name)
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                scenario = Scenario(data["name"], input=data.get("input"))
                suite.add(scenario)
        return suite
