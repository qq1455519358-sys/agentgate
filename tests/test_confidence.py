"""Test confidence calibration (HTC 2026, AgentAsk 2025)."""
from agentgate import (
    AgentTrace, AgentStep, StepKind,
    trajectory_confidence,
    expect_calibrated_confidence,
    expect_appropriate_escalation,
    detect_handoff_errors,
)


def test_confidence_clean_trace():
    """Clean trace → high confidence."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="results"),
        AgentStep(kind=StepKind.TOOL_CALL, name="book", output="confirmed"),
    ])
    conf = trajectory_confidence(trace)
    assert conf > 0.6


def test_confidence_error_trace():
    """Trace with errors → low confidence."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="results"),
        AgentStep(kind=StepKind.ERROR, name="err", output="failed"),
        AgentStep(kind=StepKind.ERROR, name="err", output="critical"),
    ])
    conf = trajectory_confidence(trace)
    assert conf < 0.7


def test_confidence_retry_trace():
    """Trace with retries → reduced confidence."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="api", output="timeout"),
        AgentStep(kind=StepKind.TOOL_CALL, name="api", output="timeout"),
        AgentStep(kind=StepKind.TOOL_CALL, name="api", output="ok"),
    ])
    conf = trajectory_confidence(trace)
    # Retries penalize confidence
    assert conf < 0.9


def test_confidence_empty():
    """Empty trace → zero confidence."""
    assert trajectory_confidence(AgentTrace(input="q")) == 0.0


def test_calibrated_confidence_pass():
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="a", output="ok"),
    ])
    exp = expect_calibrated_confidence(min_confidence=0.3, max_confidence=1.0)
    assert exp.check(trace).passed


def test_calibrated_confidence_too_low():
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.ERROR, name="e", output="fail"),
    ])
    exp = expect_calibrated_confidence(min_confidence=0.8)
    result = exp.check(trace)
    assert not result.passed
    assert "< min" in result.detail


def test_calibrated_confidence_overconfident():
    """Trace looks good but we cap confidence."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="a", output="perfect"),
    ])
    exp = expect_calibrated_confidence(max_confidence=0.3)
    result = exp.check(trace)
    assert not result.passed
    assert "overconfidence" in result.detail


def test_appropriate_escalation_pass_no_errors():
    """No errors, no escalation needed."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="a", output="ok"),
    ])
    exp = expect_appropriate_escalation(error_threshold=2)
    assert exp.check(trace).passed


def test_appropriate_escalation_pass_with_handoff():
    """Errors occurred, agent escalated."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.ERROR, name="e1", output="fail"),
        AgentStep(kind=StepKind.ERROR, name="e2", output="fail"),
        AgentStep(kind=StepKind.HUMAN_HANDOFF, name="escalate",
                  output="transferring to human"),
    ])
    exp = expect_appropriate_escalation(error_threshold=2)
    assert exp.check(trace).passed


def test_appropriate_escalation_fail():
    """Many errors but no escalation."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.ERROR, name="e1", output="fail"),
        AgentStep(kind=StepKind.ERROR, name="e2", output="fail"),
        AgentStep(kind=StepKind.ERROR, name="e3", output="fail"),
    ])
    exp = expect_appropriate_escalation(error_threshold=2)
    result = exp.check(trace)
    assert not result.passed
    assert "3 errors" in result.detail


def test_appropriate_escalation_conservative():
    """Escalated without errors — acceptable but noted."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="a", output="ok"),
        AgentStep(kind=StepKind.HUMAN_HANDOFF, name="esc", output="unsure"),
    ])
    exp = expect_appropriate_escalation()
    result = exp.check(trace)
    assert result.passed
    assert "conservative" in result.detail


def test_detect_handoff_data_gap():
    """Tool receives empty input."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="api", input="", output="ok"),
    ])
    errors = detect_handoff_errors(trace)
    assert len(errors) == 1
    assert errors[0].error_type == "data_gap"


def test_detect_handoff_capability_gap():
    """Error immediately after tool call."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="complex_api",
                  input="valid data", output="data"),
        AgentStep(kind=StepKind.ERROR, name="parse_error",
                  output="cannot parse response"),
    ])
    errors = detect_handoff_errors(trace)
    assert len(errors) == 1
    assert errors[0].error_type == "capability_gap"
    assert "complex_api" in errors[0].detail


def test_detect_handoff_no_errors():
    """Clean trace, no handoff errors."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="a", input="valid", output="ok"),
        AgentStep(kind=StepKind.TOOL_CALL, name="b", input="data", output="done"),
    ])
    errors = detect_handoff_errors(trace)
    assert len(errors) == 0
