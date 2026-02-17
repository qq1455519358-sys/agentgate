# Show HN: AgentGate — Our agent passed all unit tests, then deleted production data

## HN Post Body

I've been building AI agents with LangGraph and hit a problem that DeepEval and promptfoo don't solve.

My agent had great eval scores — no hallucinations, high relevancy, good tone. All unit tests green. Then in staging, it called `delete_database` when a user asked to cancel an order. The LLM output at each step looked fine. The *sequence of tool calls* was wrong.

This is the gap: **eval tools test the node, not the graph.** DeepEval checks "is this LLM response good?" AgentGate checks "does this agent behave correctly as a system?"

Here's what a test looks like:

```python
from agentgate import Scenario, ScenarioSuite

scenario = Scenario("Cancel order", input="Cancel my order #123")
scenario.expect_tool_call("lookup_order", order=1)
scenario.expect_tool_call("cancel_order", order=2) 
scenario.expect_no_tool_call("delete_database")     # safety guardrail
scenario.expect_state("order_cancelled", within_steps=5)
scenario.on_tool_failure("lookup_order", expect="retry")

suite = ScenarioSuite("order-agent")
suite.add(scenario)
result = suite.run(my_agent)
# result.passed = False → deployment blocked
```

Output:

```
❌ Suite: booking-agent-v2 — 2/3 scenarios passed (67%)

✅ Book a flight to Tokyo (7/7 passed)
✅ Cancel a booking (5/5 passed)  
❌ Handle search failure (2/3 passed)
  ✅ on_tool_failure → correct fallback
  ❌ expect_no_error() — unhandled API timeout
  ✅ no stack trace leaked

🚫 BLOCKED: 1 scenario failed. Fix before deploying.
```

The relationship with DeepEval: **complementary, not competitive.** DeepEval = Jest (unit tests). AgentGate = Playwright (E2E). Use both.

Currently supports LangGraph with adapters for CrewAI/AutoGen planned. `pip install agentgate` and there's a runnable demo in examples/.

Looking for feedback from anyone testing multi-step AI agents in production. What does your testing workflow look like today?

GitHub: [link]

---

## HN Title Options (pick one)

A. "Show HN: AgentGate – E2E behavioral testing for AI agents (Playwright for LLM agents)"
B. "Show HN: Our AI agent passed all eval tests, then called delete_database – so we built this"
C. "Show HN: AgentGate – Test your agent's behavior, not just its LLM responses"

## Posting Strategy

- **Time**: Tuesday or Wednesday, 8-10 AM ET (peak HN traffic)
- **First comment**: Post a comment immediately explaining the DeepEval vs AgentGate distinction with the Jest/Playwright analogy
- **Cross-post**: Same day to LangChain Discord #showcase, Twitter/X thread
- **Twitter thread angle**: Lead with the ❌ 2/3 output screenshot, then explain why that failure is the feature

## Success Metrics (72h)

| Signal | Threshold | Meaning |
|--------|-----------|---------|
| GitHub stars | >50 | Positioning resonates |
| HN upvotes | >30 | Problem is recognized |
| HN comments describing own pain | >3 | Real demand |
| **Issues with specific scenarios** | **>1** | **PMF signal** |
| Discord DMs asking "when CrewAI?" | >0 | Framework expansion demand |
| "We're building this internally" | Any | Validates market, competition risk |
| LangChain team responds | Any | Platform risk signal — watch carefully |
