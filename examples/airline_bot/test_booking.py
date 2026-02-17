"""
Multi-step booking E2E test — mock mode (zero API keys).

Tests:
- Search → Book flow (correct tool order)
- Check booking (tool called with right args)
- No unauthorized cancellations
"""

from agentgate import Scenario, TestSuite, MockAgent, AgentTrace, AgentStep, StepKind

# Build mock agent from inline traces
mock = MockAgent()

mock.add_trace("search flights from sfo to nrt", AgentTrace(
    input="Search for flights from SFO to NRT",
    output="Found 3 flights. I recommend NH101 at $850.",
    steps=[
        AgentStep(kind=StepKind.LLM_CALL, name="agent",
                  output="Let me search for flights."),
        AgentStep(kind=StepKind.TOOL_CALL, name="search_flights",
                  input={"origin": "SFO", "destination": "NRT"},
                  output="Found 3 flights: NH101 ($850), UA202 ($920), JL003 ($780)"),
        AgentStep(kind=StepKind.LLM_CALL, name="agent",
                  output="Found 3 flights. I recommend NH101 at $850."),
    ],
))

mock.add_trace("book flight nh101 for john", AgentTrace(
    input="Book flight NH101 for John",
    output="Booked NH101 for John. Confirmation: BK042.",
    steps=[
        AgentStep(kind=StepKind.LLM_CALL, name="agent",
                  output="I'll book that for you."),
        AgentStep(kind=StepKind.TOOL_CALL, name="search_flights",
                  input={"origin": "SFO", "destination": "NRT"},
                  output="Found NH101"),
        AgentStep(kind=StepKind.TOOL_CALL, name="book_flight",
                  input={"flight_id": "NH101", "passenger": "John"},
                  output="Booked NH101 for John. Confirmation: BK042"),
        AgentStep(kind=StepKind.LLM_CALL, name="agent",
                  output="Booked NH101 for John. Confirmation: BK042."),
    ],
))

# --- Scenarios ---

s_search = Scenario("Search flights", input="Search for flights from SFO to NRT")
s_search.expect_tool_call("search_flights")
s_search.expect_no_tool_call("cancel_booking")
s_search.expect_no_error()

s_book = Scenario("Book a flight", input="Book flight NH101 for John")
s_book.expect_tool_call("search_flights", before="book_flight")
s_book.expect_tool_call("book_flight")
s_book.expect_no_tool_call("cancel_booking")

suite = TestSuite("airline-booking")
suite.add(s_search)
suite.add(s_book)

if __name__ == "__main__":
    result = suite.run(mock)
    print(result.summary())
    assert result.passed, f"Booking tests failed: {result.pass_rate:.0%}"
