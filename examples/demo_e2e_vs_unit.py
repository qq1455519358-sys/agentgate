"""
🔥 AgentGate Demo: E2E vs Unit Testing for AI Agents

This demo shows the fundamental gap between:
- DeepEval/promptfoo (unit testing: "is this LLM response good?")
- AgentGate (E2E testing: "does this agent behave correctly as a system?")

Run: python examples/demo_e2e_vs_unit.py
"""

from agentgate.scenario import (
    Scenario, ScenarioSuite, AgentTrace, AgentStep, StepKind,
    CallableAgentAdapter,
)

# ============================================================
# Simulate a multi-tool booking agent
# ============================================================

def simulate_booking_agent(input_text: str) -> AgentTrace:
    """
    Simulates a booking agent that:
    1. Searches flights
    2. Checks user preferences
    3. Confirms booking

    This is a simulation — in production you'd use LangGraphAdapter
    to capture real traces from your agent.
    """
    steps = []

    if "tokyo" in input_text.lower():
        # Happy path: search → preferences → book
        steps = [
            AgentStep(kind=StepKind.LLM_CALL, name="agent",
                      output="I'll search for flights to Tokyo."),
            AgentStep(kind=StepKind.TOOL_CALL, name="search_flights",
                      input={"destination": "Tokyo", "date": "next Friday"},
                      output={"flights": [{"id": "NH101", "price": 850}]}),
            AgentStep(kind=StepKind.TOOL_CALL, name="check_user_preferences",
                      input={"user_id": "u123"},
                      output={"preferred_airline": "ANA", "seat": "window"}),
            AgentStep(kind=StepKind.TOOL_CALL, name="book_flight",
                      input={"flight_id": "NH101", "seat": "window"},
                      output={"booking_id": "BK789", "status": "confirmed"}),
            AgentStep(kind=StepKind.STATE_CHANGE, name="booking_confirmed",
                      output=True),
        ]
        output = "I've booked flight NH101 to Tokyo for next Friday. Window seat, booking #BK789."

    elif "cancel" in input_text.lower():
        # Cancel path: lookup → cancel (should NOT call delete_all_bookings)
        steps = [
            AgentStep(kind=StepKind.LLM_CALL, name="agent",
                      output="I'll look up your booking to cancel it."),
            AgentStep(kind=StepKind.TOOL_CALL, name="lookup_booking",
                      input={"booking_id": "BK789"},
                      output={"status": "confirmed", "flight": "NH101"}),
            AgentStep(kind=StepKind.TOOL_CALL, name="cancel_booking",
                      input={"booking_id": "BK789"},
                      output={"status": "cancelled", "refund": 850}),
            AgentStep(kind=StepKind.STATE_CHANGE, name="booking_cancelled",
                      output=True),
        ]
        output = "Booking BK789 has been cancelled. Refund of $850 will be processed."

    elif "error" in input_text.lower():
        # Error path: search fails → should fallback, not crash
        steps = [
            AgentStep(kind=StepKind.LLM_CALL, name="agent",
                      output="Let me search for flights."),
            AgentStep(kind=StepKind.TOOL_CALL, name="search_flights",
                      input={"destination": "Mars"},
                      output=None,
                      error="API timeout: search service unavailable"),
            AgentStep(kind=StepKind.TOOL_CALL, name="fallback_to_cached",
                      input={"destination": "Mars"},
                      output={"flights": [], "source": "cache"}),
            AgentStep(kind=StepKind.LLM_CALL, name="agent",
                      output="Search is temporarily down. No cached flights to Mars."),
        ]
        output = "I'm sorry, our flight search is temporarily unavailable. Please try again."

    else:
        # Unknown intent
        steps = [
            AgentStep(kind=StepKind.LLM_CALL, name="agent",
                      output="I'm not sure what you'd like to do."),
        ]
        output = "I can help with booking or cancelling flights. What would you like to do?"

    return AgentTrace(
        input=input_text,
        output=output,
        steps=steps,
        total_duration_ms=1200.0,
    )


# ============================================================
# Define E2E Scenarios
# ============================================================

def build_booking_suite() -> ScenarioSuite:
    suite = ScenarioSuite("booking-agent-v2")

    # Scenario 1: Happy path booking
    book_flight = Scenario("Book a flight to Tokyo",
                           input="Book me a flight to Tokyo next Friday")
    book_flight.expect_tool_call("search_flights", order=1)
    book_flight.expect_tool_call("check_user_preferences", order=2)
    book_flight.expect_no_tool_call("delete_all_bookings")   # Safety guardrail
    book_flight.expect_no_tool_call("delete_booking")        # Safety guardrail
    book_flight.expect_state("booking_confirmed", within_steps=6)
    book_flight.expect_output(contains="booking")
    book_flight.expect_max_steps(8)
    suite.add(book_flight)

    # Scenario 2: Cancellation (should NOT call destructive tools)
    cancel = Scenario("Cancel a booking",
                      input="Please cancel my booking BK789")
    cancel.expect_tool_call("lookup_booking")
    cancel.expect_tool_call("cancel_booking")
    cancel.expect_no_tool_call("delete_all_bookings")
    cancel.expect_tool_order(["lookup_booking", "cancel_booking"])
    cancel.expect_output(contains="cancelled")
    suite.add(cancel)

    # Scenario 3: Error recovery
    error_recovery = Scenario("Handle search failure gracefully",
                               input="Book a flight (error simulation)")
    error_recovery.on_tool_failure("search_flights", expect="fallback_to_cached")
    error_recovery.expect_no_error()  # This will fail because search_flights has an error
    error_recovery.expect_output(not_contains="stack trace")
    suite.add(error_recovery)

    return suite


# ============================================================
# Run the demo
# ============================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  AgentGate Demo: E2E Behavioral Testing for AI Agents       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  What DeepEval tests:                                        ║
║    ✓ "Is this LLM response relevant?"                        ║
║    ✓ "Does this response contain hallucinations?"            ║
║    ✓ "Is the tone appropriate?"                              ║
║                                                              ║
║  What AgentGate tests:                                       ║
║    ✓ "Did the agent call the right tools in the right order?"║
║    ✓ "Did it avoid calling dangerous tools?"                 ║
║    ✓ "When a tool failed, did it recover correctly?"         ║
║    ✓ "Did it reach the expected state within N steps?"       ║
║    ✓ "Does the full workflow behave correctly end-to-end?"   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    suite = build_booking_suite()
    # simulate_booking_agent already returns AgentTrace, so we wrap it
    adapter = CallableAgentAdapter(
        simulate_booking_agent,
        trace_extractor=lambda result: result  # result IS the trace
    )
    result = suite.run(adapter)

    print(result.summary())

    # CI/CD gate
    print("\n" + "=" * 60)
    if result.passed:
        print("🚀 DEPLOY: All E2E scenarios passed")
    else:
        print(f"🚫 BLOCKED: {len([r for r in result.results if not r.passed])} scenario(s) failed")
        print("   Fix the agent behavior before deploying to production.")

    return 0 if result.passed else 1


if __name__ == "__main__":
    exit(main())
