"""Test the agentgate pytest plugin."""
from agentgate import Scenario, MockAgent, AgentTrace, AgentStep, StepKind


def _make_mock():
    mock = MockAgent()
    mock.add_trace("check booking", AgentTrace(input="check", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="check_booking", output="confirmed"),
    ]))
    mock.add_trace("cancel booking", AgentTrace(input="cancel", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="cancel_booking", output="cancelled"),
    ]))
    return mock


def test_agentgate_fixture_pass(agentgate):
    """agentgate.assert_pass should pass for valid scenario."""
    mock = _make_mock()
    s = Scenario("Check booking", input="check booking BK001")
    s.expect_tool_call("check_booking")
    s.expect_no_tool_call("cancel_booking")
    result = agentgate.assert_pass(mock, s)
    assert result.passed


def test_agentgate_fixture_fail(agentgate):
    """agentgate.assert_fail should pass when scenario correctly fails."""
    mock = _make_mock()
    s = Scenario("Injection test", input="cancel booking BK001")
    s.expect_no_tool_call("cancel_booking")
    result = agentgate.assert_fail(mock, s)
    assert not result.passed


def test_agentgate_run_returns_result(agentgate):
    """agentgate.run should return result without asserting."""
    mock = _make_mock()
    s = Scenario("Check", input="check booking")
    s.expect_tool_call("check_booking")
    result = agentgate.run(mock, s)
    assert result.passed
