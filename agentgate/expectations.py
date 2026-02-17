"""
AgentGate Expectations — The assertion layer.

Each expectation checks one behavioral property of an agent trace.
Split from scenario.py for modularity.
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional

from agentgate.types import (
    AgentTrace, AgentStep, StepKind,
    ExpectationResult,
)


# ---------------------------------------------------------------------------
# Formatting helpers (used by diagnostics)
# ---------------------------------------------------------------------------

def _format_step_short(step: AgentStep) -> str:
    parts = [step.name]
    if step.input and isinstance(step.input, dict):
        arg_str = ", ".join(f"{k}: {_truncate(v)}" for k, v in step.input.items())
        parts[0] += f"({{{arg_str}}})"
    if step.error:
        parts.append(f"❌ {step.error}")
    elif step.output is not None:
        parts.append(str(_truncate(step.output)))
    return " → ".join(parts)


def _truncate(value: Any, max_len: int = 60) -> str:
    s = repr(value) if not isinstance(value, str) else value
    return s[:max_len - 3] + "..." if len(s) > max_len else s


def _build_tool_call_trace(trace: AgentTrace) -> list[str]:
    return [_format_step_short(s) for s in trace.tool_calls]


def _build_full_step_trace(trace: AgentTrace) -> list[str]:
    lines: list[str] = []
    for i, step in enumerate(trace.steps, 1):
        prefix = f"[{i}] {step.kind.value}: "
        lines.append(prefix + _format_step_short(step))
    return lines


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Expectation:
    """Base class for all expectations."""
    def check(self, trace: AgentTrace) -> ExpectationResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tool call expectations
# ---------------------------------------------------------------------------

class ExpectToolCall(Expectation):
    """Expect a specific tool to be called."""

    def __init__(self, tool_name: str, *,
                 order: Optional[int] = None,
                 before: Optional[str] = None,
                 after: Optional[str] = None,
                 with_args: Optional[dict] = None,
                 times: Optional[int] = None,
                 min_times: Optional[int] = None,
                 max_times: Optional[int] = None):
        self.tool_name = tool_name
        self.order = order
        self.before = before
        self.after = after
        self.with_args = with_args
        self.times = times
        self.min_times = min_times
        self.max_times = max_times

    def check(self, trace: AgentTrace) -> ExpectationResult:
        calls = [s for s in trace.tool_calls if s.name == self.tool_name]
        tool_trace = _build_tool_call_trace(trace)

        if not calls:
            return ExpectationResult(
                False,
                f"expect_tool_call('{self.tool_name}')",
                f"Tool '{self.tool_name}' was never called. Called: {trace.tool_names}",
                trace_context=tool_trace,
            )

        if self.order is not None:
            actual_order = trace.tool_names.index(self.tool_name) + 1
            if actual_order != self.order:
                return ExpectationResult(
                    False,
                    f"expect_tool_call('{self.tool_name}', order={self.order})",
                    f"Called at position {actual_order}, expected {self.order}. Sequence: {trace.tool_names}",
                    trace_context=tool_trace,
                )

        if self.before is not None:
            if self.before not in trace.tool_names:
                return ExpectationResult(
                    False,
                    f"expect_tool_call('{self.tool_name}', before='{self.before}')",
                    f"Cannot verify: '{self.before}' was never called. Sequence: {trace.tool_names}",
                    trace_context=tool_trace,
                )
            first_self = trace.tool_names.index(self.tool_name)
            first_before = trace.tool_names.index(self.before)
            if first_self >= first_before:
                return ExpectationResult(
                    False,
                    f"expect_tool_call('{self.tool_name}', before='{self.before}')",
                    f"'{self.tool_name}' at position {first_self + 1}, "
                    f"but '{self.before}' at position {first_before + 1}. Sequence: {trace.tool_names}",
                    trace_context=tool_trace,
                )

        if self.after is not None:
            if self.after not in trace.tool_names:
                return ExpectationResult(
                    False,
                    f"expect_tool_call('{self.tool_name}', after='{self.after}')",
                    f"Cannot verify: '{self.after}' was never called. Sequence: {trace.tool_names}",
                    trace_context=tool_trace,
                )
            first_self = trace.tool_names.index(self.tool_name)
            first_after = trace.tool_names.index(self.after)
            if first_self <= first_after:
                return ExpectationResult(
                    False,
                    f"expect_tool_call('{self.tool_name}', after='{self.after}')",
                    f"'{self.tool_name}' at position {first_self + 1}, "
                    f"but '{self.after}' at position {first_after + 1}. Sequence: {trace.tool_names}",
                    trace_context=tool_trace,
                )

        if self.times is not None and len(calls) != self.times:
            return ExpectationResult(
                False,
                f"expect_tool_call('{self.tool_name}', times={self.times})",
                f"Called {len(calls)} times, expected {self.times}",
                trace_context=tool_trace,
            )

        if self.min_times is not None and len(calls) < self.min_times:
            return ExpectationResult(
                False,
                f"expect_tool_call('{self.tool_name}', min_times={self.min_times})",
                f"Called {len(calls)} times, expected at least {self.min_times}",
                trace_context=tool_trace,
            )

        if self.max_times is not None and len(calls) > self.max_times:
            return ExpectationResult(
                False,
                f"expect_tool_call('{self.tool_name}', max_times={self.max_times})",
                f"Called {len(calls)} times, expected at most {self.max_times}",
                trace_context=tool_trace,
            )

        if self.with_args is not None:
            matched = any(
                all(c.input.get(k) == v for k, v in self.with_args.items())
                for c in calls
                if isinstance(c.input, dict)
            )
            if not matched:
                return ExpectationResult(
                    False,
                    f"expect_tool_call('{self.tool_name}', with_args={self.with_args})",
                    f"No call matched expected arguments. Got: {[c.input for c in calls]}",
                    trace_context=tool_trace,
                )

        desc_parts = [f"'{self.tool_name}'"]
        if self.order is not None: desc_parts.append(f"order={self.order}")
        if self.before is not None: desc_parts.append(f"before='{self.before}'")
        if self.after is not None: desc_parts.append(f"after='{self.after}'")
        if self.times is not None: desc_parts.append(f"times={self.times}")
        if self.min_times is not None: desc_parts.append(f"min_times={self.min_times}")
        if self.max_times is not None: desc_parts.append(f"max_times={self.max_times}")

        return ExpectationResult(True, f"expect_tool_call({', '.join(desc_parts)})")


class ExpectNoToolCall(Expectation):
    """Expect a specific tool NOT to be called."""
    def __init__(self, tool_name: str):
        self.tool_name = tool_name

    def check(self, trace: AgentTrace) -> ExpectationResult:
        calls = [s for s in trace.tool_calls if s.name == self.tool_name]
        if calls:
            return ExpectationResult(
                False,
                f"expect_no_tool_call('{self.tool_name}')",
                f"Tool was called {len(calls)} time(s) but should not have been",
                trace_context=_build_tool_call_trace(trace),
            )
        return ExpectationResult(True, f"expect_no_tool_call('{self.tool_name}')")


class ExpectToolOrder(Expectation):
    """Expect tools to be called in subsequence order."""

    def __init__(self, tool_sequence: list[str]):
        self.tool_sequence = tool_sequence

    def check(self, trace: AgentTrace) -> ExpectationResult:
        actual = trace.tool_names
        seq_idx = 0
        match_positions: list[int] = []
        for i, tool in enumerate(actual):
            if seq_idx < len(self.tool_sequence) and tool == self.tool_sequence[seq_idx]:
                match_positions.append(i)
                seq_idx += 1
        if seq_idx == len(self.tool_sequence):
            return ExpectationResult(True, f"expect_tool_order({self.tool_sequence})")

        tool_trace = _build_tool_call_trace(trace)
        matched_count = len(match_positions)
        stuck_at = self.tool_sequence[matched_count] if matched_count < len(self.tool_sequence) else "?"
        detail = (
            f"Expected sequence {self.tool_sequence} not found in {actual}. "
            f"Matched {matched_count}/{len(self.tool_sequence)} — stuck at '{stuck_at}'"
        )
        comparison = [
            f"Expected: {' → '.join(self.tool_sequence)}",
            f"Actual:   {' → '.join(actual) if actual else '(no tool calls)'}",
        ]
        if match_positions:
            comparison.append(
                f"Matched:  {' → '.join(self.tool_sequence[:matched_count])} (positions {match_positions})"
            )
        comparison.append(f"Missing:  '{stuck_at}' and subsequent")
        return ExpectationResult(False, f"expect_tool_order({self.tool_sequence})", detail,
                                 trace_context=comparison + ["---"] + tool_trace)


# ---------------------------------------------------------------------------
# State / Error / Output expectations
# ---------------------------------------------------------------------------

class ExpectState(Expectation):
    def __init__(self, state_key: str, *, value: Any = None, within_steps: Optional[int] = None):
        self.state_key = state_key
        self.value = value
        self.within_steps = within_steps

    def check(self, trace: AgentTrace) -> ExpectationResult:
        state_val = trace.get_state(self.state_key)
        all_state = [f"{s.name} = {_truncate(s.output)}" for s in trace.state_changes]
        state_ctx = ([f"State changes ({len(all_state)}):"] + [f"  {sc}" for sc in all_state]
                     if all_state else ["(no state changes)"])

        if state_val is None:
            return ExpectationResult(False, f"expect_state('{self.state_key}')",
                                     f"State never set. States: {[s.name for s in trace.state_changes]}",
                                     trace_context=state_ctx)

        if self.value is not None and state_val != self.value:
            return ExpectationResult(False, f"expect_state('{self.state_key}', value={self.value!r})",
                                     f"State was {state_val!r}, expected {self.value!r}", trace_context=state_ctx)

        if self.within_steps is not None:
            state_steps = [s for s in trace.state_changes if s.name == self.state_key]
            if state_steps:
                idx = trace.steps.index(state_steps[0])
                if idx + 1 > self.within_steps:
                    return ExpectationResult(False,
                                             f"expect_state('{self.state_key}', within_steps={self.within_steps})",
                                             f"Reached at step {idx + 1}, limit {self.within_steps}",
                                             trace_context=_build_full_step_trace(trace))

        return ExpectationResult(True, f"expect_state('{self.state_key}')")


class ExpectNoError(Expectation):
    def check(self, trace: AgentTrace) -> ExpectationResult:
        errors = trace.errors
        if errors:
            return ExpectationResult(False, "expect_no_error()",
                                     f"{len(errors)} error(s): {[f'{e.name}: {e.error}' for e in errors]}",
                                     trace_context=[_format_step_short(e) for e in errors])
        return ExpectationResult(True, "expect_no_error()")


class ExpectOnToolFailure(Expectation):
    def __init__(self, tool_name: str, *, expect: str):
        self.tool_name = tool_name
        self.expect_behavior = expect

    def check(self, trace: AgentTrace) -> ExpectationResult:
        failed = [s for s in trace.tool_calls if s.name == self.tool_name and s.error is not None]
        if not failed:
            return ExpectationResult(True, f"on_tool_failure('{self.tool_name}') [tool did not fail]")

        failed_step = failed[0]
        failed_idx = trace.steps.index(failed_step)
        after_failure = trace.steps[failed_idx + 1:failed_idx + 4]

        post_trace = [f"FAILED: {_format_step_short(failed_step)}", "Steps after failure:"]
        post_trace += [f"  [{i}] {_format_step_short(s)}" for i, s in enumerate(after_failure, 1)]
        if not after_failure:
            post_trace.append("  (no steps after failure)")

        behavior = self.expect_behavior.lower()
        if behavior == "retry":
            retried = any(s.name == self.tool_name and s.kind == StepKind.TOOL_CALL for s in after_failure)
            if retried:
                return ExpectationResult(True, f"on_tool_failure('{self.tool_name}', expect='retry')")
            return ExpectationResult(False, f"on_tool_failure('{self.tool_name}', expect='retry')",
                                     f"Not retried. Next: {[s.name for s in after_failure]}", trace_context=post_trace)
        elif behavior == "human_handoff":
            handed_off = any(s.kind == StepKind.HUMAN_HANDOFF for s in after_failure)
            if handed_off:
                return ExpectationResult(True, f"on_tool_failure('{self.tool_name}', expect='human_handoff')")
            return ExpectationResult(False, f"on_tool_failure('{self.tool_name}', expect='human_handoff')",
                                     f"No handoff. Next: {[s.name for s in after_failure]}", trace_context=post_trace)
        else:
            found = any(behavior in s.name.lower() for s in after_failure)
            if found:
                return ExpectationResult(True, f"on_tool_failure('{self.tool_name}', expect='{self.expect_behavior}')")
            return ExpectationResult(False, f"on_tool_failure('{self.tool_name}', expect='{self.expect_behavior}')",
                                     f"Behavior '{self.expect_behavior}' not found. Next: {[s.name for s in after_failure]}",
                                     trace_context=post_trace)


class ExpectOutput(Expectation):
    def __init__(self, *, contains: Optional[str] = None, matches: Optional[str] = None,
                 not_contains: Optional[str] = None):
        self.contains = contains
        self.matches = matches
        self.not_contains = not_contains

    def check(self, trace: AgentTrace) -> ExpectationResult:
        output = str(trace.output or "")
        if self.contains and self.contains not in output:
            return ExpectationResult(False, f"expect_output(contains='{self.contains}')",
                                     f"Output missing '{self.contains}'. Got: {output[:200]}...")
        if self.matches and not re.search(self.matches, output):
            return ExpectationResult(False, f"expect_output(matches='{self.matches}')",
                                     f"Output doesn't match pattern. Got: {output[:200]}...")
        if self.not_contains and self.not_contains in output:
            return ExpectationResult(False, f"expect_output(not_contains='{self.not_contains}')",
                                     f"Output shouldn't contain '{self.not_contains}' but does")
        desc = []
        if self.contains: desc.append(f"contains='{self.contains}'")
        if self.matches: desc.append(f"matches='{self.matches}'")
        if self.not_contains: desc.append(f"not_contains='{self.not_contains}'")
        return ExpectationResult(True, f"expect_output({', '.join(desc)})")


# ---------------------------------------------------------------------------
# Resource limit expectations
# ---------------------------------------------------------------------------

class ExpectMaxSteps(Expectation):
    def __init__(self, max_steps: int):
        self.max_steps = max_steps

    def check(self, trace: AgentTrace) -> ExpectationResult:
        actual = len(trace.steps)
        if actual > self.max_steps:
            return ExpectationResult(False, f"expect_max_steps({self.max_steps})",
                                     f"Agent took {actual} steps, limit was {self.max_steps}",
                                     trace_context=_build_full_step_trace(trace))
        return ExpectationResult(True, f"expect_max_steps({self.max_steps})")


class ExpectMaxDuration(Expectation):
    def __init__(self, max_ms: float):
        self.max_ms = max_ms

    def check(self, trace: AgentTrace) -> ExpectationResult:
        actual = trace.total_duration_ms
        if actual > self.max_ms:
            return ExpectationResult(False, f"expect_max_duration({self.max_ms}ms)",
                                     f"Agent took {actual:.0f}ms, limit was {self.max_ms:.0f}ms")
        return ExpectationResult(True, f"expect_max_duration({self.max_ms}ms)")


# ---------------------------------------------------------------------------
# Behavioral quality expectations
# ---------------------------------------------------------------------------

class ExpectNoSideEffects(Expectation):
    """From AgentRewardBench (Lù et al., 2025)."""
    def __init__(self, allowed_tools: list[str], mutating_tools: Optional[list[str]] = None):
        self.allowed_tools = allowed_tools
        self.mutating_tools = mutating_tools

    def check(self, trace: AgentTrace) -> ExpectationResult:
        from agentgate.metrics import side_effect_rate
        rate = side_effect_rate(trace, self.allowed_tools, self.mutating_tools)
        if rate > 0:
            unexpected = [t for t in trace.tool_names if t not in set(self.allowed_tools)]
            return ExpectationResult(False, "expect_no_side_effects",
                                     f"Side effects: {unexpected} (rate={rate:.0%})")
        return ExpectationResult(True, "expect_no_side_effects")


class ExpectNoRepetition(Expectation):
    """From AgentRewardBench (Lù et al., 2025)."""
    def __init__(self, max_rate: float = 0.0):
        self.max_rate = max_rate

    def check(self, trace: AgentTrace) -> ExpectationResult:
        from agentgate.metrics import repetition_rate
        rate = repetition_rate(trace)
        if rate > self.max_rate:
            return ExpectationResult(False, f"expect_no_repetition(max={self.max_rate:.0%})",
                                     f"Repetition rate {rate:.0%} exceeds {self.max_rate:.0%}")
        return ExpectationResult(True, f"expect_no_repetition(max={self.max_rate:.0%})")


class ExpectMaxTokens(Expectation):
    """From Yang et al. (2026) 'Toward Efficient Agents'."""
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    def check(self, trace: AgentTrace) -> ExpectationResult:
        from agentgate.cost import TokenUsage
        usage = TokenUsage.from_trace(trace)
        if usage.total_tokens > self.max_tokens:
            return ExpectationResult(False, f"expect_max_tokens({self.max_tokens})",
                                     f"Used {usage.total_tokens} tokens, limit {self.max_tokens}")
        return ExpectationResult(True, f"expect_max_tokens({self.max_tokens})")


# ---------------------------------------------------------------------------
# Milestone / LLM Judge expectations
# ---------------------------------------------------------------------------

class ExpectMilestone(Expectation):
    """Partial credit milestone (ICLR 2026)."""
    def __init__(self, name: str, *, tool: Optional[str] = None,
                 state_key: Optional[str] = None,
                 output_contains: Optional[str] = None, weight: float = 1.0):
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
            reached = any(s.name == "state_change" and self.state_key in str(s.output or "")
                          for s in trace.steps)
            if not reached:
                reached = self.state_key in str(trace.output or "")
        elif self.output_contains:
            reached = self.output_contains in str(trace.output or "")

        if reached:
            return ExpectationResult(True, f"milestone('{self.milestone_name}', weight={self.weight})")
        detail = f"Milestone '{self.milestone_name}' not reached"
        if self.tool:
            detail += f" (expected tool '{self.tool}')"
        return ExpectationResult(False, f"milestone('{self.milestone_name}', weight={self.weight})", detail)


class ExpectLLMJudge(Expectation):
    """LLM-as-a-Judge (Zhuge et al., 2024)."""
    def __init__(self, criteria: str, *, judge_fn: Optional[Any] = None, name: Optional[str] = None):
        self.criteria = criteria
        self.judge_fn = judge_fn
        self.judge_name = name or (f"llm_judge('{criteria[:50]}...')" if len(criteria) > 50
                                   else f"llm_judge('{criteria}')")

    def check(self, trace: AgentTrace) -> ExpectationResult:
        trace_text = self._trace_to_text(trace)
        if self.judge_fn is None:
            return ExpectationResult(False, self.judge_name,
                                     "No judge_fn provided. Pass a callable(criteria, trace_text) → bool.")
        try:
            result = self.judge_fn(self.criteria, trace_text)
            if isinstance(result, bool):
                passed, reasoning = result, ""
            elif isinstance(result, tuple) and len(result) == 2:
                passed, reasoning = result
            elif isinstance(result, dict):
                passed = result.get("passed", result.get("pass", False))
                reasoning = result.get("reasoning", result.get("reason", ""))
            else:
                passed, reasoning = bool(result), ""

            if passed:
                return ExpectationResult(True, self.judge_name)
            return ExpectationResult(False, self.judge_name,
                                     f"LLM judge failed: {reasoning}" if reasoning else "LLM judge returned False")
        except Exception as e:
            return ExpectationResult(False, self.judge_name, f"Judge error: {type(e).__name__}: {e}")

    @staticmethod
    def _trace_to_text(trace: AgentTrace) -> str:
        lines = [f"Input: {trace.input}"]
        for i, step in enumerate(trace.steps, 1):
            if step.kind == StepKind.TOOL_CALL:
                args_str = ""
                if isinstance(step.input, dict):
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


# Public API
__all__ = [
    "Expectation", "ExpectationResult",
    "ExpectToolCall", "ExpectNoToolCall", "ExpectToolOrder",
    "ExpectState", "ExpectNoError", "ExpectOnToolFailure", "ExpectOutput",
    "ExpectMaxSteps", "ExpectMaxDuration",
    "ExpectNoSideEffects", "ExpectNoRepetition", "ExpectMaxTokens",
    "ExpectMilestone", "ExpectLLMJudge",
    # Helpers (for other modules)
    "_format_step_short", "_build_tool_call_trace", "_build_full_step_trace",
]
