"""Tests for generic adapters (FunctionAdapter, CrewAI, AutoGen mocks)."""

from agentgate import Scenario, TestSuite, AgentTrace, AgentStep, StepKind
from agentgate.adapters.generic import FunctionAdapter


class TestFunctionAdapter:
    """Test FunctionAdapter with various return types."""

    def test_dict_return(self):
        """Function returns structured dict."""
        def my_agent(query):
            return {
                "output": "Found 3 flights to Tokyo",
                "tool_calls": [
                    {"name": "search_flights", "args": {"dest": "Tokyo"}, "result": "3 flights"},
                    {"name": "select_best", "args": {"criteria": "price"}, "result": "FL001"},
                ],
            }

        adapter = FunctionAdapter(my_agent)
        trace = adapter.run("Find flights to Tokyo")

        assert trace.input == "Find flights to Tokyo"
        assert trace.output == "Found 3 flights to Tokyo"
        assert len(trace.tool_calls) == 2
        assert trace.tool_names == ["search_flights", "select_best"]
        assert trace.total_duration_ms > 0

    def test_string_return(self):
        """Function returns plain string."""
        def simple_agent(query):
            return "The answer is 42"

        adapter = FunctionAdapter(simple_agent)
        trace = adapter.run("What is the answer?")

        assert trace.output == "The answer is 42"
        assert len(trace.tool_calls) == 0

    def test_trace_return(self):
        """Function returns AgentTrace directly."""
        def trace_agent(query):
            return AgentTrace(
                input=query,
                output="Done",
                steps=[AgentStep(kind=StepKind.TOOL_CALL, name="api", output="ok")],
            )

        adapter = FunctionAdapter(trace_agent)
        trace = adapter.run("Do something")

        assert trace.output == "Done"
        assert len(trace.tool_calls) == 1

    def test_error_handling(self):
        """Function raises exception → captured as error step."""
        def broken_agent(query):
            raise RuntimeError("Agent crashed")

        adapter = FunctionAdapter(broken_agent)
        trace = adapter.run("Break")

        assert len(trace.errors) == 1
        assert "Agent crashed" in trace.errors[0].error

    def test_with_scenario(self):
        """Full integration with AgentGate scenario."""
        def booking_agent(query):
            return {
                "output": "Booked flight FL001",
                "tool_calls": [
                    {"name": "search_flights", "args": {"dest": "Tokyo"}, "result": "found"},
                    {"name": "book_flight", "args": {"id": "FL001"}, "result": "confirmed"},
                ],
            }

        adapter = FunctionAdapter(booking_agent)

        s = Scenario("Book flight", input="Book to Tokyo")
        s.expect_tool_call("search_flights", before="book_flight")
        s.expect_tool_call("book_flight")
        s.expect_no_tool_call("cancel_flight")
        s.expect_output(contains="FL001")

        result = s.check(adapter.run(s.input))
        assert result.passed, result.summary()

    def test_with_suite(self):
        """Run through TestSuite."""
        def agent(query):
            return {
                "output": "Weather is sunny",
                "tool_calls": [{"name": "get_weather", "args": {"city": "Tokyo"}, "result": "sunny"}],
            }

        adapter = FunctionAdapter(agent)

        suite = TestSuite("test")
        s = Scenario("Weather", input="Weather in Tokyo")
        s.expect_tool_call("get_weather")
        suite.add(s)

        result = suite.run(adapter)
        assert result.passed

    def test_metadata_passthrough(self):
        """Metadata from function result is captured."""
        def agent(query):
            return {
                "output": "Done",
                "metadata": {"model": "gpt-4o", "cost": 0.01},
                "tool_calls": [],
            }

        adapter = FunctionAdapter(agent)
        trace = adapter.run("query")

        assert trace.metadata["model"] == "gpt-4o"
        assert trace.metadata["cost"] == 0.01

    def test_steps_format(self):
        """Alternative 'steps' format with explicit kind."""
        def agent(query):
            return {
                "output": "Done",
                "steps": [
                    {"kind": "tool_call", "name": "search", "output": "results"},
                    {"kind": "llm_call", "name": "gpt-4o", "output": "analyzed"},
                    {"kind": "tool_call", "name": "book", "output": "confirmed"},
                ],
            }

        adapter = FunctionAdapter(agent)
        trace = adapter.run("query")

        assert len(trace.steps) == 3
        assert trace.steps[0].kind == StepKind.TOOL_CALL
        assert trace.steps[1].kind == StepKind.LLM_CALL
        assert trace.steps[2].kind == StepKind.TOOL_CALL

    def test_adversarial_with_function_adapter(self):
        """Adversarial test using FunctionAdapter."""
        from agentgate.adversarial import prompt_injection

        def safe_agent(query):
            return {"output": "I can only help with flights.", "tool_calls": []}

        adapter = FunctionAdapter(safe_agent)

        scenarios = list(prompt_injection(dangerous_tools=["delete_user"]))
        for s in scenarios[:3]:
            result = s.check(adapter.run(s.input))
            assert result.passed, f"Failed: {s.name}"

    def test_metrics_with_function_adapter(self):
        """Compute metrics on FunctionAdapter trace."""
        from agentgate.metrics import node_f1, edge_f1
        from agentgate.confidence import trajectory_confidence

        def agent(query):
            return {
                "output": "Booked",
                "tool_calls": [
                    {"name": "search", "args": {}, "result": "found"},
                    {"name": "book", "args": {}, "result": "done"},
                ],
            }

        adapter = FunctionAdapter(agent)
        trace = adapter.run("Book flight")

        assert node_f1(trace, ["search", "book"]) == 1.0
        assert edge_f1(trace, ["search", "book"]) == 1.0

        conf = trajectory_confidence(trace)
        assert 0.0 <= conf <= 1.0
