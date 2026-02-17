"""Test capability vs regression eval management (Anthropic, Jan 2026)."""
from agentgate import (
    Scenario, MockAgent, AgentTrace, AgentStep, StepKind,
    EvalSuiteManager,
)


def _mock_passing():
    mock = MockAgent()
    mock.add_trace("core", AgentTrace(input="core", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="check", output="ok"),
    ]))
    mock.add_trace("new", AgentTrace(input="new", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="feature", output="ok"),
    ]))
    return mock


def _mock_regression():
    """Mock where core regresses but new feature works."""
    mock = MockAgent()
    mock.add_trace("core", AgentTrace(input="core", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="wrong", output="broken"),
    ]))
    mock.add_trace("new", AgentTrace(input="new", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="feature", output="ok"),
    ]))
    return mock


def test_manager_basic_run():
    mgr = EvalSuiteManager()

    s_reg = Scenario("Core booking", input="core")
    s_reg.expect_tool_call("check")
    mgr.add_regression("core", s_reg)

    s_cap = Scenario("New feature", input="new")
    s_cap.expect_tool_call("feature")
    mgr.add_capability("new", s_cap)

    result = mgr.run(_mock_passing())

    assert result.regression_pass_rate == 1.0
    assert result.capability_pass_rate == 1.0
    assert not result.has_regressions


def test_manager_detects_regression():
    mgr = EvalSuiteManager()

    s_reg = Scenario("Core booking", input="core")
    s_reg.expect_tool_call("check")
    mgr.add_regression("core", s_reg)

    s_cap = Scenario("New feature", input="new")
    s_cap.expect_tool_call("feature")
    mgr.add_capability("new", s_cap)

    result = mgr.run(_mock_regression())

    assert result.has_regressions
    assert result.regression_pass_rate == 0.0
    assert result.capability_pass_rate == 1.0
    assert "Core booking" in result.summary()


def test_manager_graduate():
    """Passing capability tests graduate to regression suite."""
    mgr = EvalSuiteManager()

    s_cap = Scenario("New feature", input="new")
    s_cap.expect_tool_call("feature")
    mgr.add_capability("new", s_cap)

    assert mgr.capability_count == 1
    assert mgr.regression_count == 0

    # Run — capability passes
    result = mgr.run(_mock_passing())
    assert "New feature" in result.new_capabilities

    # Graduate
    graduated = mgr.graduate(threshold=0.95)
    assert "new" in graduated
    assert mgr.capability_count == 0
    assert mgr.regression_count == 1


def test_manager_empty_suites():
    mgr = EvalSuiteManager()
    result = mgr.run(_mock_passing())
    assert result.regression_pass_rate == 1.0  # vacuously true
    assert result.capability_pass_rate == 0.0
    assert not result.has_regressions


def test_manager_summary():
    mgr = EvalSuiteManager()

    s = Scenario("Core", input="core")
    s.expect_tool_call("check")
    mgr.add_regression("core", s)

    result = mgr.run(_mock_passing())
    summary = result.summary()
    assert "Regression" in summary
    assert "Capability" in summary
    assert "✅" in summary


def test_manager_counts():
    mgr = EvalSuiteManager()
    mgr.add_capability("a", Scenario("A", input="a"))
    mgr.add_capability("b", Scenario("B", input="b"))
    mgr.add_regression("c", Scenario("C", input="c"))

    assert mgr.capability_count == 2
    assert mgr.regression_count == 1
