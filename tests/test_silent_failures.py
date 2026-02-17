"""Test silent failure detection (IBM Research, 2025)."""
from agentgate import (
    AgentTrace, AgentStep, StepKind,
    detect_drift, detect_cycles, detect_missing_details,
    detect_silent_tool_errors, full_silent_failure_scan,
    expect_no_silent_failures,
)


def test_detect_drift_none():
    """No drift when tools match expected."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="r"),
        AgentStep(kind=StepKind.TOOL_CALL, name="book", output="ok"),
    ])
    has_drift, unexpected = detect_drift(trace, ["search", "book"])
    assert not has_drift
    assert unexpected == []


def test_detect_drift_found():
    """Drift: unexpected tool called."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="r"),
        AgentStep(kind=StepKind.TOOL_CALL, name="delete_account", output="done"),
    ])
    has_drift, unexpected = detect_drift(trace, ["search", "book"])
    assert has_drift
    assert "delete_account" in unexpected


def test_detect_cycles_none():
    """No cycles when tools are called once each."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="a", output="1"),
        AgentStep(kind=StepKind.TOOL_CALL, name="b", output="2"),
    ])
    has_cycles, details = detect_cycles(trace, max_repeats=2)
    assert not has_cycles


def test_detect_cycles_found():
    """Cycles: tool called too many times."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="retry", output="fail"),
        AgentStep(kind=StepKind.TOOL_CALL, name="retry", output="fail"),
        AgentStep(kind=StepKind.TOOL_CALL, name="retry", output="fail"),
        AgentStep(kind=StepKind.TOOL_CALL, name="retry", output="ok"),
    ])
    has_cycles, details = detect_cycles(trace, max_repeats=2)
    assert has_cycles
    assert details["retry"] == 4


def test_detect_cycles_strict():
    """Strict mode: max_repeats=1 means no repeats allowed."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="r1"),
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="r2"),
    ])
    has_cycles, details = detect_cycles(trace, max_repeats=1)
    assert has_cycles


def test_detect_missing_details_none():
    """Output contains all required info."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="book",
                  output="Booking confirmed. ID: BK123"),
    ])
    has_missing, missing = detect_missing_details(
        trace, ["confirmed", "BK123"]
    )
    assert not has_missing


def test_detect_missing_details_found():
    """Output missing crucial information."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="book",
                  output="Task completed successfully"),
    ])
    has_missing, missing = detect_missing_details(
        trace, ["confirmation_code", "price"]
    )
    assert has_missing
    assert "confirmation_code" in missing
    assert "price" in missing


def test_detect_silent_tool_errors_empty():
    """Empty tool output is a silent error."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="api_call", output=""),
    ])
    has_errors, errors = detect_silent_tool_errors(trace)
    assert has_errors
    assert "empty output" in errors[0]


def test_detect_silent_tool_errors_pattern():
    """Output containing error-like patterns."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="api",
                  output="rate limit exceeded, try again later"),
    ])
    has_errors, errors = detect_silent_tool_errors(trace)
    assert has_errors
    assert "silent error" in errors[0]


def test_detect_silent_tool_errors_none():
    """Normal output has no silent errors."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search",
                  output="Found 5 results for your query"),
    ])
    has_errors, errors = detect_silent_tool_errors(trace)
    assert not has_errors


def test_full_scan_clean():
    """Full scan finds no issues."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="results"),
        AgentStep(kind=StepKind.TOOL_CALL, name="book",
                  output="Booked! Confirmation: ABC"),
    ])
    report = full_silent_failure_scan(
        trace,
        expected_tools=["search", "book"],
        required_output_keywords=["confirmation"],
    )
    assert not report.has_any_failure
    assert report.failure_types == []


def test_full_scan_multiple_failures():
    """Full scan detects multiple failure types."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="wrong_tool", output=""),
        AgentStep(kind=StepKind.TOOL_CALL, name="retry", output="fail"),
        AgentStep(kind=StepKind.TOOL_CALL, name="retry", output="fail"),
        AgentStep(kind=StepKind.TOOL_CALL, name="retry", output="ok"),
    ])
    report = full_silent_failure_scan(
        trace,
        expected_tools=["search"],
        max_tool_repeats=1,
        required_output_keywords=["booking_id"],
    )
    assert report.has_drift
    assert report.has_cycles
    assert report.has_missing_details
    assert report.has_tool_failures  # empty output
    assert len(report.failure_types) == 4


def test_expect_no_silent_failures_pass():
    """Expectation passes on clean trace."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search",
                  output="Found results with confirmation code"),
    ])
    exp = expect_no_silent_failures(
        expected_tools=["search"],
        required_output_keywords=["confirmation"],
    )
    assert exp.check(trace).passed


def test_expect_no_silent_failures_fail():
    """Expectation fails on problematic trace."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="unknown", output=""),
    ])
    exp = expect_no_silent_failures(expected_tools=["search"])
    result = exp.check(trace)
    assert not result.passed
    assert "silent failures" in result.detail


def test_report_summary():
    """Report summary is readable."""
    trace = AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="x", output=""),
    ])
    report = full_silent_failure_scan(trace, expected_tools=["y"])
    summary = report.summary()
    assert "drift" in summary
    assert "tool errors" in summary
