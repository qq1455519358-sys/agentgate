"""Tests for agentgate.mock — MockAgent + serialization."""

import json
import tempfile
from pathlib import Path

from agentgate import MockAgent, AgentTrace, AgentStep, StepKind


class TestMockAgent:
    def test_exact_match(self):
        mock = MockAgent({"hello": AgentTrace(input="hello", output="hi")})
        trace = mock.run("hello")
        assert trace.output == "hi"

    def test_substring_match(self):
        mock = MockAgent({"book flight": AgentTrace(input="book", output="booked")})
        trace = mock.run("Please book flight to Tokyo")
        assert trace.output == "booked"

    def test_no_match_returns_error(self):
        mock = MockAgent()
        trace = mock.run("unknown input")
        assert len(trace.steps) == 1
        assert trace.steps[0].kind == StepKind.ERROR

    def test_add_trace(self):
        mock = MockAgent()
        mock.add_trace("hi", AgentTrace(input="hi", output="hello"))
        trace = mock.run("hi")
        assert trace.output == "hello"

    def test_case_insensitive(self):
        mock = MockAgent({"Hello World": AgentTrace(input="", output="matched")})
        trace = mock.run("hello world")
        assert trace.output == "matched"


class TestTraceSerialization:
    def test_save_and_load_roundtrip(self):
        original = AgentTrace(
            input="test input",
            output="test output",
            total_duration_ms=100.0,
            steps=[
                AgentStep(kind=StepKind.TOOL_CALL, name="search",
                          input={"q": "test"}, output="found"),
                AgentStep(kind=StepKind.LLM_CALL, name="agent",
                          output="Here are the results"),
            ],
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            MockAgent.save_trace(original, f.name)
            loaded = MockAgent.load_trace(f.name)

        assert loaded.input == original.input
        assert loaded.output == original.output
        assert len(loaded.steps) == 2
        assert loaded.steps[0].name == "search"
        assert loaded.steps[0].kind == StepKind.TOOL_CALL

    def test_from_traces_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace1 = AgentTrace(input="search", output="found",
                                steps=[AgentStep(kind=StepKind.TOOL_CALL,
                                                 name="search", output="ok")])
            trace2 = AgentTrace(input="book", output="booked",
                                steps=[AgentStep(kind=StepKind.TOOL_CALL,
                                                 name="book", output="ok")])

            MockAgent.save_trace(trace1, Path(tmpdir) / "search_flights.json")
            MockAgent.save_trace(trace2, Path(tmpdir) / "book_flight.json")

            mock = MockAgent.from_traces(tmpdir)
            # The keys should be derived from filenames
            t1 = mock.run("search flights")
            assert t1.output == "found"
            t2 = mock.run("book flight")
            assert t2.output == "booked"
