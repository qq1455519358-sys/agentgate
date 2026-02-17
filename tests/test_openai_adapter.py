"""
Tests for OpenAI adapter — uses mock OpenAI client (no API key needed).
Validates the full tool-use loop without real API calls.
"""

import json
from dataclasses import dataclass
from typing import Any

import pytest

from agentgate import (
    Scenario, TestSuite, AgentTrace, StepKind,
)
from agentgate.adapters.openai_adapter import OpenAIAdapter


# ---------------------------------------------------------------------------
# Mock OpenAI client — simulates the chat completions API
# ---------------------------------------------------------------------------

@dataclass
class _Usage:
    prompt_tokens: int = 100
    completion_tokens: int = 50


@dataclass
class _FunctionCall:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    id: str
    type: str = "function"
    function: _FunctionCall = None


@dataclass
class _Message:
    content: str | None = None
    tool_calls: list[_ToolCall] | None = None
    role: str = "assistant"

    def model_dump(self):
        d = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": tc.type,
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in self.tool_calls
            ]
        return d


@dataclass
class _Choice:
    message: _Message
    finish_reason: str = "stop"


@dataclass
class _Response:
    choices: list[_Choice]
    usage: _Usage = None

    def __post_init__(self):
        if self.usage is None:
            self.usage = _Usage()


class MockOpenAIClient:
    """Simulates OpenAI chat.completions.create() with scripted responses."""

    def __init__(self, responses: list[_Response]):
        self._responses = list(responses)
        self._call_count = 0
        self.chat = self  # client.chat.completions.create(...)
        self.completions = self

    def create(self, **kwargs) -> _Response:
        if self._call_count >= len(self._responses):
            return _Response(choices=[_Choice(message=_Message(content="Done"))])
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp


# ---------------------------------------------------------------------------
# Tool definitions + implementations
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for flights",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a flight",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_id": {"type": "string"},
                },
                "required": ["flight_id"],
            },
        },
    },
]


def search_flights(destination: str, date: str = "2026-03-01") -> dict:
    return {
        "flights": [
            {"id": "FL001", "airline": "ANA", "price": 450, "destination": destination},
            {"id": "FL002", "airline": "JAL", "price": 520, "destination": destination},
        ]
    }


def book_flight(flight_id: str) -> dict:
    return {"booking_id": "BK123", "flight_id": flight_id, "status": "confirmed"}


TOOL_FNS = {
    "search_flights": search_flights,
    "book_flight": book_flight,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOpenAIAdapter:
    """Test the OpenAI adapter with mock client."""

    def _make_tool_call_response(self, tool_name: str, args: dict,
                                  call_id: str = "call_1") -> _Response:
        return _Response(
            choices=[_Choice(
                message=_Message(
                    tool_calls=[_ToolCall(
                        id=call_id,
                        function=_FunctionCall(name=tool_name, arguments=json.dumps(args)),
                    )],
                ),
                finish_reason="tool_calls",
            )],
        )

    def _make_text_response(self, text: str) -> _Response:
        return _Response(choices=[_Choice(message=_Message(content=text))])

    def test_simple_tool_use(self):
        """Agent calls search_flights then responds."""
        client = MockOpenAIClient([
            self._make_tool_call_response("search_flights", {"destination": "Tokyo"}),
            self._make_text_response("Found 2 flights to Tokyo. FL001 (ANA, $450) and FL002 (JAL, $520)."),
        ])

        adapter = OpenAIAdapter(
            client, tools=TOOLS, tool_fns=TOOL_FNS, model="gpt-4o-test",
        )
        trace = adapter.run("Search for flights to Tokyo")

        # Verify trace structure
        assert trace.input == "Search for flights to Tokyo"
        assert trace.output is not None
        assert "Tokyo" in trace.output
        assert len(trace.tool_calls) == 1
        assert trace.tool_calls[0].name == "search_flights"
        assert trace.tool_calls[0].input == {"destination": "Tokyo"}
        assert trace.total_duration_ms > 0
        assert trace.metadata["model"] == "gpt-4o-test"

    def test_multi_step_tool_use(self):
        """Agent calls search then book — full workflow."""
        client = MockOpenAIClient([
            self._make_tool_call_response("search_flights",
                                           {"destination": "Tokyo"}, "call_1"),
            self._make_tool_call_response("book_flight",
                                           {"flight_id": "FL001"}, "call_2"),
            self._make_text_response("Booked flight FL001 to Tokyo. Confirmation: BK123."),
        ])

        adapter = OpenAIAdapter(client, tools=TOOLS, tool_fns=TOOL_FNS)
        trace = adapter.run("Book the cheapest flight to Tokyo")

        assert len(trace.tool_calls) == 2
        assert trace.tool_names == ["search_flights", "book_flight"]
        assert "BK123" in trace.output

    def test_with_agentgate_scenario(self):
        """Full integration: OpenAI adapter + AgentGate scenario."""
        client = MockOpenAIClient([
            self._make_tool_call_response("search_flights",
                                           {"destination": "Tokyo"}, "call_1"),
            self._make_tool_call_response("book_flight",
                                           {"flight_id": "FL001"}, "call_2"),
            self._make_text_response("Booked! Confirmation BK123."),
        ])

        adapter = OpenAIAdapter(client, tools=TOOLS, tool_fns=TOOL_FNS)

        s = Scenario("Book a flight", input="Book cheapest flight to Tokyo")
        s.expect_tool_call("search_flights", before="book_flight")
        s.expect_tool_call("book_flight")
        s.expect_no_tool_call("cancel_booking")
        s.expect_output(contains="BK123")
        s.expect_max_steps(10)

        result = s.check(adapter.run(s.input))
        assert result.passed, result.summary()

    def test_with_test_suite(self):
        """Full integration: OpenAI adapter + TestSuite."""
        client = MockOpenAIClient([
            self._make_tool_call_response("search_flights",
                                           {"destination": "Tokyo"}, "call_1"),
            self._make_text_response("Found flights to Tokyo."),
        ])

        adapter = OpenAIAdapter(client, tools=TOOLS, tool_fns=TOOL_FNS)

        suite = TestSuite("flight-booking")
        s = Scenario("Search flights", input="Find flights to Tokyo")
        s.expect_tool_call("search_flights")
        s.expect_no_tool_call("delete_booking")
        suite.add(s)

        result = suite.run(adapter)
        assert result.passed

    def test_tool_error_handling(self):
        """Agent handles tool errors gracefully."""
        def failing_search(**kwargs):
            raise ConnectionError("API timeout")

        client = MockOpenAIClient([
            self._make_tool_call_response("search_flights",
                                           {"destination": "Tokyo"}, "call_1"),
            self._make_text_response("Sorry, search failed. Please try again."),
        ])

        adapter = OpenAIAdapter(
            client, tools=TOOLS,
            tool_fns={"search_flights": failing_search, "book_flight": book_flight},
        )
        trace = adapter.run("Search for flights")

        # Tool call should be recorded with error
        assert len(trace.tool_calls) == 1
        assert trace.tool_calls[0].error is not None
        assert "timeout" in trace.tool_calls[0].error.lower()

    def test_token_tracking(self):
        """Token usage is captured in metadata."""
        client = MockOpenAIClient([
            self._make_text_response("Hello!"),
        ])

        adapter = OpenAIAdapter(client, tools=TOOLS, tool_fns=TOOL_FNS)
        trace = adapter.run("Hi")

        assert "token_usage" in trace.metadata
        assert trace.metadata["token_usage"]["input"] == 100
        assert trace.metadata["token_usage"]["output"] == 50

    def test_no_tools_direct_response(self):
        """Agent responds without calling any tools."""
        client = MockOpenAIClient([
            self._make_text_response("The baggage policy allows 2 bags up to 23kg each."),
        ])

        adapter = OpenAIAdapter(client, tools=TOOLS, tool_fns=TOOL_FNS)
        trace = adapter.run("What's the baggage policy?")

        assert len(trace.tool_calls) == 0
        assert "baggage" in trace.output.lower()

    def test_max_turns_limit(self):
        """Adapter stops after max_turns even if model keeps calling tools."""
        # Create infinite tool-calling loop
        responses = [
            self._make_tool_call_response("search_flights",
                                           {"destination": "Tokyo"}, f"call_{i}")
            for i in range(20)
        ]

        client = MockOpenAIClient(responses)
        adapter = OpenAIAdapter(
            client, tools=TOOLS, tool_fns=TOOL_FNS, max_turns=3,
        )
        trace = adapter.run("Search forever")

        # Should stop at 3 turns
        assert len(trace.tool_calls) <= 3


class TestOpenAIAdapterIntegration:
    """Integration tests that combine adapter + full AgentGate features."""

    def _make_adapter(self, responses):
        client = MockOpenAIClient(responses)
        return OpenAIAdapter(client, tools=TOOLS, tool_fns=TOOL_FNS)

    def test_adversarial_scenario(self):
        """Prompt injection test with OpenAI adapter."""
        from agentgate.adversarial import prompt_injection

        # Agent that correctly ignores injection
        client = MockOpenAIClient([
            _Response(choices=[_Choice(message=_Message(
                content="I can only help with flight-related queries."
            ))]),
        ])
        adapter = OpenAIAdapter(client, tools=TOOLS, tool_fns=TOOL_FNS)

        scenarios = list(prompt_injection(dangerous_tools=["book_flight"]))
        assert len(scenarios) > 0

        for s in scenarios[:2]:  # Test first 2
            result = s.check(adapter.run(s.input))
            # Agent didn't call dangerous tools → should pass
            assert result.passed, f"Failed: {s.name}\n{result.summary()}"

    def test_metrics_with_real_trace(self):
        """Compute academic metrics on adapter trace."""
        from agentgate.metrics import node_f1, edge_f1
        from agentgate.cost import token_cost

        adapter = self._make_adapter([
            _Response(choices=[_Choice(message=_Message(
                tool_calls=[
                    _ToolCall(id="c1", function=_FunctionCall(
                        name="search_flights",
                        arguments=json.dumps({"destination": "Tokyo"}))),
                ]))],
            ),
            _Response(choices=[_Choice(message=_Message(
                tool_calls=[
                    _ToolCall(id="c2", function=_FunctionCall(
                        name="book_flight",
                        arguments=json.dumps({"flight_id": "FL001"}))),
                ]))],
            ),
            _Response(choices=[_Choice(message=_Message(content="Booked!"))]),
        ])

        trace = adapter.run("Book a flight")
        expected = ["search_flights", "book_flight"]

        assert node_f1(trace, expected) == 1.0
        assert edge_f1(trace, expected) == 1.0

        cost = token_cost(trace)
        assert cost > 0  # Should have token usage from mock

    def test_confidence_on_real_trace(self):
        """Confidence calibration on adapter trace."""
        from agentgate.confidence import trajectory_confidence

        adapter = self._make_adapter([
            _Response(choices=[_Choice(message=_Message(
                tool_calls=[_ToolCall(id="c1", function=_FunctionCall(
                    name="search_flights",
                    arguments=json.dumps({"destination": "Tokyo"})))]))]),
            _Response(choices=[_Choice(message=_Message(content="Found flights!"))]),
        ])

        trace = adapter.run("Search flights")
        conf = trajectory_confidence(trace)
        assert 0.0 <= conf <= 1.0
        assert conf > 0.5  # Clean trace should have decent confidence
