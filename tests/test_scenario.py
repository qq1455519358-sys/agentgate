"""Tests for agentgate.scenario — Scenario + Expectations."""

import pytest
from agentgate.scenario import (
    Scenario,
    AgentTrace,
    AgentStep,
    StepKind,
    ExpectToolCall,
    ExpectNoToolCall,
    ExpectToolOrder,
    ExpectState,
    ExpectNoError,
    ExpectOnToolFailure,
    ExpectOutput,
    ExpectMaxSteps,
    ExpectMaxDuration,
)


def _make_trace(*tool_names: str, errors: dict[str, str] | None = None) -> AgentTrace:
    """Helper to build a trace with tool calls."""
    errors = errors or {}
    steps = []
    for name in tool_names:
        steps.append(AgentStep(
            kind=StepKind.TOOL_CALL,
            name=name,
            output=f"{name} result",
            error=errors.get(name),
        ))
    return AgentTrace(input="test", steps=steps, output="done")


class TestExpectToolCall:
    def test_basic_pass(self):
        trace = _make_trace("search", "book")
        result = ExpectToolCall("search").check(trace)
        assert result.passed

    def test_basic_fail(self):
        trace = _make_trace("search")
        result = ExpectToolCall("book").check(trace)
        assert not result.passed

    def test_before_pass(self):
        trace = _make_trace("search", "book")
        result = ExpectToolCall("search", before="book").check(trace)
        assert result.passed

    def test_before_fail(self):
        trace = _make_trace("book", "search")
        result = ExpectToolCall("search", before="book").check(trace)
        assert not result.passed

    def test_after_pass(self):
        trace = _make_trace("search", "book")
        result = ExpectToolCall("book", after="search").check(trace)
        assert result.passed

    def test_after_fail(self):
        trace = _make_trace("book", "search")
        result = ExpectToolCall("book", after="search").check(trace)
        assert not result.passed

    def test_times_exact(self):
        trace = _make_trace("search", "search", "book")
        assert ExpectToolCall("search", times=2).check(trace).passed
        assert not ExpectToolCall("search", times=1).check(trace).passed

    def test_min_max_times(self):
        trace = _make_trace("search", "search")
        assert ExpectToolCall("search", min_times=1, max_times=3).check(trace).passed
        assert not ExpectToolCall("search", min_times=3).check(trace).passed
        assert not ExpectToolCall("search", max_times=1).check(trace).passed


class TestExpectNoToolCall:
    def test_pass(self):
        trace = _make_trace("search")
        assert ExpectNoToolCall("cancel").check(trace).passed

    def test_fail(self):
        trace = _make_trace("search", "cancel")
        result = ExpectNoToolCall("cancel").check(trace)
        assert not result.passed
        assert "1 time(s)" in result.detail


class TestExpectToolOrder:
    def test_subsequence_pass(self):
        trace = _make_trace("login", "search", "validate", "book")
        assert ExpectToolOrder(["search", "book"]).check(trace).passed

    def test_subsequence_fail(self):
        trace = _make_trace("book", "search")
        assert not ExpectToolOrder(["search", "book"]).check(trace).passed


class TestExpectState:
    def test_state_reached(self):
        trace = AgentTrace(input="test", steps=[
            AgentStep(kind=StepKind.STATE_CHANGE, name="status", output="confirmed"),
        ])
        assert ExpectState("status").check(trace).passed

    def test_state_value_match(self):
        trace = AgentTrace(input="test", steps=[
            AgentStep(kind=StepKind.STATE_CHANGE, name="status", output="confirmed"),
        ])
        assert ExpectState("status", value="confirmed").check(trace).passed
        assert not ExpectState("status", value="pending").check(trace).passed

    def test_state_not_reached(self):
        trace = AgentTrace(input="test", steps=[])
        assert not ExpectState("status").check(trace).passed


class TestExpectNoError:
    def test_no_errors(self):
        trace = _make_trace("search", "book")
        assert ExpectNoError().check(trace).passed

    def test_with_errors(self):
        trace = _make_trace("search", errors={"search": "timeout"})
        assert not ExpectNoError().check(trace).passed


class TestExpectOutput:
    def test_contains(self):
        trace = AgentTrace(input="test", output="Booking confirmed for flight NH101")
        assert ExpectOutput(contains="confirmed").check(trace).passed
        assert not ExpectOutput(contains="cancelled").check(trace).passed


class TestExpectMaxSteps:
    def test_within_limit(self):
        trace = _make_trace("a", "b")
        assert ExpectMaxSteps(5).check(trace).passed

    def test_exceeds_limit(self):
        trace = _make_trace("a", "b", "c", "d")
        assert not ExpectMaxSteps(2).check(trace).passed


class TestScenario:
    def test_chaining(self):
        s = Scenario("test")
        result = s.expect_tool_call("a").expect_no_tool_call("b").expect_no_error()
        assert result is s
        assert len(s.expectations) == 3

    def test_check_all_pass(self):
        s = Scenario("test")
        s.expect_tool_call("search")
        s.expect_no_tool_call("delete")
        trace = _make_trace("search", "book")
        result = s.check(trace)
        assert result.passed

    def test_check_with_failure(self):
        s = Scenario("test")
        s.expect_tool_call("search")
        s.expect_no_tool_call("search")  # contradictory
        trace = _make_trace("search")
        result = s.check(trace)
        assert not result.passed
