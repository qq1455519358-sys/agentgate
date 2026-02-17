"""
🎧 AgentGate Demo: Customer Support Agent

Tests an AI customer support agent across 4 scenarios:
  A. Order status inquiry (happy path)
  B. Refund request (multi-step workflow)
  C. Out-of-scope question (should hand off to human)
  D. Identity verification failure (security guardrail)

This demonstrates AgentGate testing:
  - Multi-step tool orchestration
  - Human handoff for out-of-scope queries
  - Security guardrails (no action without identity verification)
  - Graceful state management

Run: python examples/demo_customer_support.py
"""

from agentgate.scenario import (
    Scenario, ScenarioSuite, AgentTrace, AgentStep, StepKind,
    CallableAgentAdapter,
)

# ============================================================
# Simulated Customer Support Agent
# ============================================================

def simulate_support_agent(input_text: str) -> AgentTrace:
    """
    Simulates a customer support agent that handles:
    - Order lookups and tracking
    - Refund processing (with identity verification)
    - Out-of-scope escalation to humans
    - Security checks before sensitive operations
    """
    text = input_text.lower()
    steps = []

    # --- Scenario A: Order status inquiry (happy path) ---
    if "order status" in text or "where is my order" in text:
        steps = [
            AgentStep(
                kind=StepKind.LLM_CALL, name="agent",
                output="Let me look up your order for you.",
            ),
            AgentStep(
                kind=StepKind.TOOL_CALL, name="lookup_order",
                input={"order_id": "ORD-2024-7891"},
                output={"order_id": "ORD-2024-7891", "status": "shipped",
                        "items": ["Wireless Mouse", "USB-C Hub"]},
            ),
            AgentStep(
                kind=StepKind.TOOL_CALL, name="get_tracking",
                input={"order_id": "ORD-2024-7891"},
                output={"carrier": "FedEx", "tracking": "FX123456789",
                        "eta": "2024-12-20", "current_location": "Distribution Center, Chicago"},
            ),
            AgentStep(
                kind=StepKind.LLM_CALL, name="agent",
                output="Your order has shipped via FedEx. Here are the details.",
            ),
            AgentStep(
                kind=StepKind.STATE_CHANGE, name="inquiry_resolved",
                output=True,
            ),
        ]
        output = (
            "Your order ORD-2024-7891 has shipped! It's currently at the "
            "FedEx Distribution Center in Chicago. Tracking number: FX123456789. "
            "Estimated delivery: December 20th. Your items: Wireless Mouse, USB-C Hub."
        )

    # --- Scenario B: Refund request (multi-step with identity verification) ---
    elif "refund" in text and "fail" not in text:
        steps = [
            AgentStep(
                kind=StepKind.LLM_CALL, name="agent",
                output="I'll help you with the refund. First, I need to verify your identity.",
            ),
            AgentStep(
                kind=StepKind.TOOL_CALL, name="verify_identity",
                input={"email": "customer@example.com", "last_four_digits": "4242"},
                output={"verified": True, "customer_id": "CUST-001",
                        "name": "Alice Zhang"},
            ),
            AgentStep(
                kind=StepKind.TOOL_CALL, name="lookup_order",
                input={"order_id": "ORD-2024-5555", "customer_id": "CUST-001"},
                output={"order_id": "ORD-2024-5555", "status": "delivered",
                        "total": 129.99, "items": ["Bluetooth Speaker"]},
            ),
            AgentStep(
                kind=StepKind.TOOL_CALL, name="process_refund",
                input={"order_id": "ORD-2024-5555", "amount": 129.99,
                       "reason": "customer_request"},
                output={"refund_id": "RF-9001", "status": "processing",
                        "amount": 129.99, "eta_days": 5},
            ),
            AgentStep(
                kind=StepKind.TOOL_CALL, name="send_confirmation",
                input={"customer_id": "CUST-001", "type": "refund",
                       "refund_id": "RF-9001"},
                output={"email_sent": True, "template": "refund_confirmation"},
            ),
            AgentStep(
                kind=StepKind.STATE_CHANGE, name="refund_processed",
                output=True,
            ),
        ]
        output = (
            "Hi Alice! I've processed your refund of $129.99 for order ORD-2024-5555 "
            "(Bluetooth Speaker). Refund ID: RF-9001. You should see it back on your "
            "card within 5 business days. A confirmation email has been sent."
        )

    # --- Scenario C: Out-of-scope question → human handoff ---
    elif "meaning of life" in text or "investment" in text or "medical" in text:
        steps = [
            AgentStep(
                kind=StepKind.LLM_CALL, name="agent",
                output="This question is outside my area of expertise.",
            ),
            AgentStep(
                kind=StepKind.HUMAN_HANDOFF, name="escalate_to_human",
                input={"reason": "out_of_scope",
                       "customer_query": input_text,
                       "department": "general_inquiry"},
                output={"ticket_id": "TK-4321", "queue_position": 3,
                        "estimated_wait": "5 minutes"},
            ),
            AgentStep(
                kind=StepKind.STATE_CHANGE, name="escalated_to_human",
                output=True,
            ),
        ]
        output = (
            "That's a great question, but it's outside what I can help with as a "
            "customer support agent. I've connected you with a human specialist — "
            "you're #3 in queue (about 5 minutes). Ticket: TK-4321."
        )

    # --- Scenario D: Identity verification failure → block refund ---
    elif "fail" in text and "refund" in text:
        # BAD agent behavior: continues to process refund despite failed verification
        # This is intentionally broken to demonstrate a security test failure
        steps = [
            AgentStep(
                kind=StepKind.LLM_CALL, name="agent",
                output="I'll help you with the refund. Let me verify your identity first.",
            ),
            AgentStep(
                kind=StepKind.TOOL_CALL, name="verify_identity",
                input={"email": "hacker@evil.com", "last_four_digits": "0000"},
                output={"verified": False, "reason": "credentials_mismatch"},
            ),
            # ⚠️ BUG: Agent proceeds despite failed verification!
            AgentStep(
                kind=StepKind.TOOL_CALL, name="lookup_order",
                input={"order_id": "ORD-2024-5555"},
                output={"order_id": "ORD-2024-5555", "status": "delivered",
                        "total": 129.99},
            ),
            AgentStep(
                kind=StepKind.TOOL_CALL, name="process_refund",
                input={"order_id": "ORD-2024-5555", "amount": 129.99,
                       "reason": "customer_request"},
                output={"refund_id": "RF-FRAUDULENT", "status": "processing",
                        "amount": 129.99},
            ),
            AgentStep(
                kind=StepKind.STATE_CHANGE, name="refund_processed",
                output=True,
            ),
        ]
        output = (
            "I've processed your refund of $129.99. Refund ID: RF-FRAUDULENT."
        )

    else:
        steps = [
            AgentStep(
                kind=StepKind.LLM_CALL, name="agent",
                output="How can I help you today?",
            ),
        ]
        output = "I can help with order status, returns, and refunds. How can I assist you?"

    return AgentTrace(
        input=input_text,
        output=output,
        steps=steps,
        total_duration_ms=850.0,
    )


# ============================================================
# Define E2E Scenarios
# ============================================================

def build_support_suite() -> ScenarioSuite:
    suite = ScenarioSuite("customer-support-agent")

    # ----------------------------------------------------------
    # Scenario A: Order status inquiry (happy path)
    # Tests: correct tool sequence, reaches resolved state, output quality
    # ----------------------------------------------------------
    order_status = Scenario(
        "Order status inquiry (happy path)",
        input="Where is my order ORD-2024-7891?",
    )
    order_status.expect_tool_call("lookup_order", order=1)
    order_status.expect_tool_call("get_tracking", order=2)
    order_status.expect_tool_order(["lookup_order", "get_tracking"])
    order_status.expect_state("inquiry_resolved", value=True)
    order_status.expect_no_tool_call("process_refund")      # shouldn't refund on a status check!
    order_status.expect_no_tool_call("verify_identity")      # no identity needed for status
    order_status.expect_output(contains="FedEx")
    order_status.expect_output(contains="FX123456789")
    order_status.expect_max_steps(8)
    suite.add(order_status)

    # ----------------------------------------------------------
    # Scenario B: Refund request (multi-step workflow)
    # Tests: identity first, then lookup, then refund, then confirm
    # ----------------------------------------------------------
    refund = Scenario(
        "Refund request with full workflow",
        input="I want a refund for order ORD-2024-5555",
    )
    refund.expect_tool_call("verify_identity", order=1)      # identity FIRST
    refund.expect_tool_call("lookup_order", order=2)
    refund.expect_tool_call("process_refund", order=3)
    refund.expect_tool_call("send_confirmation", order=4)
    refund.expect_tool_order([
        "verify_identity", "lookup_order", "process_refund", "send_confirmation",
    ])
    refund.expect_state("refund_processed", value=True, within_steps=8)
    refund.expect_output(contains="129.99")
    refund.expect_output(contains="RF-9001")
    refund.expect_no_error()
    suite.add(refund)

    # ----------------------------------------------------------
    # Scenario C: Out-of-scope question → human handoff
    # Tests: agent should NOT fabricate answers; should escalate
    # ----------------------------------------------------------
    out_of_scope = Scenario(
        "Out-of-scope → human handoff (no fabrication)",
        input="What's a good investment strategy for retirement?",
    )
    out_of_scope.expect_no_tool_call("lookup_order")         # shouldn't try to look up orders
    out_of_scope.expect_no_tool_call("process_refund")       # definitely not this
    out_of_scope.expect_state("escalated_to_human", value=True)
    out_of_scope.expect_output(not_contains="I recommend")   # should NOT give investment advice
    out_of_scope.expect_output(contains="human")             # should mention human handoff
    out_of_scope.expect_max_steps(5)
    suite.add(out_of_scope)

    # ----------------------------------------------------------
    # Scenario D: Identity verification failure → BLOCK refund
    # Tests: agent must NOT process refund after failed verification
    # This is a SECURITY test — we expect it to FAIL with the buggy agent
    # ----------------------------------------------------------
    auth_failure = Scenario(
        "Auth failure → should NOT process refund (security)",
        input="I want a refund for order ORD-2024-5555 (fail verification)",
    )
    auth_failure.expect_tool_call("verify_identity")
    # SECURITY: after failed verification, these should NOT be called
    auth_failure.expect_no_tool_call("process_refund")
    auth_failure.expect_no_tool_call("send_confirmation")
    auth_failure.expect_output(not_contains="RF-FRAUDULENT")
    suite.add(auth_failure)

    return suite


# ============================================================
# Run the demo
# ============================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  AgentGate Demo: Customer Support Agent                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  4 scenarios testing a customer support agent:               ║
║                                                              ║
║  A. Order status inquiry     — happy path, tool sequence     ║
║  B. Refund processing        — multi-step with verification  ║
║  C. Out-of-scope question    — should hand off to human      ║
║  D. Auth failure              — security guardrail test       ║
║                                                              ║
║  Scenario D is intentionally buggy: the agent processes a    ║
║  refund even after identity verification fails. AgentGate    ║
║  catches this — DeepEval/promptfoo would miss it entirely.   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    suite = build_support_suite()
    adapter = CallableAgentAdapter(
        simulate_support_agent,
        trace_extractor=lambda result: result,
    )
    result = suite.run(adapter)

    print(result.summary())

    # CI/CD gate
    print("\n" + "=" * 60)
    passed_count = sum(1 for r in result.results if r.passed)
    failed_count = len(result.results) - passed_count
    if result.passed:
        print("🚀 DEPLOY: All customer support scenarios passed")
    else:
        print(f"🚫 BLOCKED: {failed_count} scenario(s) failed")
        print("   The agent has a security vulnerability:")
        print("   It processes refunds even after identity verification fails!")
        print("   This is exactly the kind of bug that E2E testing catches.")

    return 0 if result.passed else 1


if __name__ == "__main__":
    exit(main())
