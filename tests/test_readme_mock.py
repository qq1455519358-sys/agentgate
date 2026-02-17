"""Verify README Mock Mode code block works."""
from agentgate import MockAgent, TestSuite, Scenario


def test_readme_mock_mode():
    mock = MockAgent.from_traces("examples/traces/")
    assert len(mock._traces) > 0, "Should load at least one trace"

    suite = TestSuite("mock-test")
    s = Scenario("Check booking", input="Check the status of booking BK001")
    s.expect_tool_call("check_booking")
    suite.add(s)
    result = suite.run(mock)
    assert result.passed, result.summary()
