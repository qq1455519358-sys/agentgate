"""Tests for agentgate.runner — TestSuite execution."""

from agentgate import Scenario, TestSuite, MockAgent, AgentTrace, AgentStep, StepKind


def _make_mock() -> MockAgent:
    mock = MockAgent()
    mock.add_trace("hello", AgentTrace(
        input="hello",
        output="Hi there!",
        steps=[
            AgentStep(kind=StepKind.TOOL_CALL, name="greet", output="Hi there!"),
        ],
    ))
    return mock


class TestTestSuite:
    def test_single_scenario_pass(self):
        mock = _make_mock()
        s = Scenario("greet", input="hello")
        s.expect_tool_call("greet")

        suite = TestSuite("test")
        suite.add(s)
        result = suite.run(mock)
        assert result.passed

    def test_single_scenario_fail(self):
        mock = _make_mock()
        s = Scenario("greet", input="hello")
        s.expect_tool_call("nonexistent")

        suite = TestSuite("test")
        suite.add(s)
        result = suite.run(mock)
        assert not result.passed

    def test_multiple_scenarios(self):
        mock = _make_mock()
        s1 = Scenario("greet", input="hello")
        s1.expect_tool_call("greet")

        s2 = Scenario("no cancel", input="hello")
        s2.expect_no_tool_call("cancel")

        suite = TestSuite("test")
        suite.add(s1).add(s2)
        result = suite.run(mock)
        assert result.passed
        assert len(result.results) == 2

    def test_summary_output(self):
        mock = _make_mock()
        s = Scenario("greet", input="hello")
        s.expect_tool_call("greet")

        suite = TestSuite("test")
        suite.add(s)
        result = suite.run(mock)
        summary = result.summary()
        assert "test" in summary
        assert "✅" in summary

    def test_callable_agent(self):
        """TestSuite should accept a plain callable."""
        def my_agent(input_text: str):
            return AgentTrace(
                input=input_text,
                output="ok",
                steps=[AgentStep(kind=StepKind.TOOL_CALL, name="do_thing", output="ok")],
            )

        # Need to wrap it properly — callable returns AgentTrace
        from agentgate.scenario import CallableAgentAdapter
        adapter = CallableAgentAdapter(my_agent, trace_extractor=lambda x: x)

        s = Scenario("test", input="go")
        s.expect_tool_call("do_thing")

        suite = TestSuite("test")
        suite.add(s)
        result = suite.run(adapter)
        assert result.passed
