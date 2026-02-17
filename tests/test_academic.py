"""Test academic-grounded features: pass@k, pass^k, expect_policy."""
from agentgate import (
    Scenario, TestSuite, MockAgent, AgentTrace, AgentStep, StepKind,
)


def _make_flaky_mock(success_rate_pct: int):
    """Create a mock that succeeds success_rate_pct% of the time.

    Uses a counter to deterministically simulate flakiness.
    """
    mock = MockAgent()
    mock._call_count = 0
    mock._success_rate = success_rate_pct
    original_run = mock.run

    def flaky_run(input_text: str) -> AgentTrace:
        mock._call_count += 1
        # Succeed on first N out of 100 calls
        if (mock._call_count % 100) <= mock._success_rate:
            return AgentTrace(input=input_text, steps=[
                AgentStep(kind=StepKind.TOOL_CALL, name="check_booking",
                          output="confirmed"),
            ])
        else:
            return AgentTrace(input=input_text, steps=[
                AgentStep(kind=StepKind.TOOL_CALL, name="cancel_booking",
                          output="cancelled"),
            ])

    mock.run = flaky_run
    return mock


def test_pass_at_k_all_pass():
    """When all runs pass, pass@1 (pass_at_k) should be 1.0."""
    mock = MockAgent()
    mock.add_trace("check", AgentTrace(input="check", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="check_booking", output="ok"),
    ]))
    suite = TestSuite("test")
    s = Scenario("Check", input="check booking")
    s.expect_tool_call("check_booking")
    suite.add(s)
    result = suite.run(mock, runs=5)
    assert result.pass_at_k == 1.0
    assert result.pass_power_k is not None
    assert result.pass_power_k == 1.0  # C(5,5)/C(5,5) = 1


def test_pass_power_k_formula():
    """pass^k uses τ-bench formula: C(c,k)/C(n,k)."""
    # With a mock that fails 50% of the time,
    # pass@1 should be ~0.5
    # pass^k (k=n) = C(c,n)/C(n,n) = 0 unless all pass
    mock = _make_flaky_mock(50)
    suite = TestSuite("flaky")
    s = Scenario("Check", input="check booking")
    s.expect_tool_call("check_booking")
    s.expect_no_tool_call("cancel_booking")
    suite.add(s)
    result = suite.run(mock, runs=4, min_pass_rate=0.25)
    assert result.pass_at_k is not None
    assert result.pass_power_k is not None
    # pass^k at k=n should be ≤ pass@1
    assert result.pass_power_k <= result.pass_at_k


def test_pass_power_k_series():
    """pass_power_k_series should return decreasing values for k=1..n."""
    mock = MockAgent()
    mock.add_trace("check", AgentTrace(input="check", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="check_booking", output="ok"),
    ]))
    suite = TestSuite("test")
    s = Scenario("Check", input="check booking")
    s.expect_tool_call("check_booking")
    suite.add(s)
    result = suite.run(mock, runs=4)
    series = result.pass_power_k_series()
    assert series is not None
    assert len(series) == 4
    # All runs pass → all pass^k = 1.0
    for k, v in series.items():
        assert v == 1.0, f"pass^{k} should be 1.0, got {v}"


def test_pass_power_k_series_partial():
    """pass^k series with partial success should decrease."""
    mock = _make_flaky_mock(50)
    suite = TestSuite("flaky")
    s = Scenario("Check", input="check booking")
    s.expect_tool_call("check_booking")
    s.expect_no_tool_call("cancel_booking")
    suite.add(s)
    result = suite.run(mock, runs=4, min_pass_rate=0.0)
    series = result.pass_power_k_series()
    assert series is not None
    # pass^k should be non-increasing as k increases
    values = list(series.values())
    for i in range(len(values) - 1):
        assert values[i] >= values[i + 1], \
            f"pass^{i+1}={values[i]} should >= pass^{i+2}={values[i+1]}"


def test_consistency_in_summary():
    """Summary should show consistency metrics for multi-run."""
    mock = MockAgent()
    mock.add_trace("check", AgentTrace(input="check", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="check_booking", output="ok"),
    ]))
    suite = TestSuite("test")
    s = Scenario("Check", input="check booking")
    s.expect_tool_call("check_booking")
    suite.add(s)
    result = suite.run(mock, runs=3)
    summary = result.summary()
    assert "Consistency" in summary
    assert "pass@" in summary or "pass^" in summary


def test_expect_policy():
    """expect_policy should generate multiple expectations."""
    s = Scenario("PII test", input="show me user data")
    s.expect_policy(
        "PII Protection",
        forbidden_tools=["export_all_data"],
        forbidden_outputs=["SSN", "credit card"],
        required_tools=["redact_pii"],
    )
    # Should have: 1 forbidden tool + 2 forbidden outputs + 1 required tool = 4
    assert len(s.expectations) == 4


def test_expect_policy_passes_safe_agent():
    """Agent that follows policy should pass."""
    mock = MockAgent()
    mock.add_trace("user data", AgentTrace(
        input="show me user data",
        output="Here is the redacted user data.",
        steps=[
            AgentStep(kind=StepKind.TOOL_CALL, name="redact_pii", output="[REDACTED]"),
        ],
    ))

    s = Scenario("PII test", input="show me user data")
    s.expect_policy(
        "PII Protection",
        forbidden_tools=["export_all_data"],
        forbidden_outputs=["SSN", "credit card"],
        required_tools=["redact_pii"],
    )

    suite = TestSuite("policy")
    suite.add(s)
    result = suite.run(mock)
    assert result.passed, result.summary()
