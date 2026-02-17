"""Test cost & efficiency metrics (Yang et al. 2026 + HAL + ICLR 2026)."""
from agentgate import (
    AgentTrace, AgentStep, StepKind,
    TokenUsage, token_cost, cost_score,
    AgentResult, pareto_frontier, is_pareto_optimal,
)


def _trace_with_tokens(input_tokens=1000, output_tokens=500, steps=5, ms=3000):
    return AgentTrace(
        input="test",
        steps=[AgentStep(kind=StepKind.TOOL_CALL, name=f"t{i}", output="ok")
               for i in range(steps)],
        total_duration_ms=ms,
        metadata={"token_usage": {"input": input_tokens, "output": output_tokens}},
    )


# --- TokenUsage extraction ---

def test_token_usage_from_token_usage_key():
    trace = _trace_with_tokens(2000, 500)
    usage = TokenUsage.from_trace(trace)
    assert usage.input_tokens == 2000
    assert usage.output_tokens == 500
    assert usage.total_tokens == 2500


def test_token_usage_from_openai_style():
    trace = AgentTrace(input="t", metadata={
        "usage": {"prompt_tokens": 1500, "completion_tokens": 300}
    })
    usage = TokenUsage.from_trace(trace)
    assert usage.input_tokens == 1500
    assert usage.output_tokens == 300


def test_token_usage_from_steps():
    trace = AgentTrace(input="t", steps=[
        AgentStep(kind=StepKind.LLM_CALL, name="llm",
                  metadata={"input_tokens": 500, "output_tokens": 100}),
        AgentStep(kind=StepKind.LLM_CALL, name="llm",
                  metadata={"input_tokens": 300, "output_tokens": 50}),
    ])
    usage = TokenUsage.from_trace(trace)
    assert usage.input_tokens == 800
    assert usage.output_tokens == 150


def test_token_usage_empty():
    trace = AgentTrace(input="t")
    usage = TokenUsage.from_trace(trace)
    assert usage.total_tokens == 0


# --- token_cost ---

def test_token_cost_default_pricing():
    # Claude Sonnet 4: $3/1M input, $15/1M output
    trace = _trace_with_tokens(1_000_000, 100_000)
    cost = token_cost(trace)  # default pricing
    assert abs(cost - (3.0 + 1.5)) < 0.01  # $3 input + $1.5 output


def test_token_cost_custom_pricing():
    trace = _trace_with_tokens(1_000_000, 1_000_000)
    cost = token_cost(trace, input_price=1.0, output_price=2.0)
    assert abs(cost - 3.0) < 0.01


def test_token_cost_zero():
    trace = AgentTrace(input="t")
    cost = token_cost(trace)
    assert cost == 0.0


# --- cost_score ---

def test_cost_score_efficient():
    trace = _trace_with_tokens(1000, 500, steps=3, ms=5000)
    score = cost_score(trace, max_tokens=10000, max_steps=20, max_ms=60000)
    assert 0.5 < score < 1.0  # well under budget


def test_cost_score_over_budget():
    trace = _trace_with_tokens(15000, 5000, steps=25, ms=90000)
    score = cost_score(trace, max_tokens=10000, max_steps=20, max_ms=60000)
    assert score < 0.1  # over budget on everything


def test_cost_score_perfect():
    trace = AgentTrace(input="t", total_duration_ms=0, metadata={
        "token_usage": {"input": 0, "output": 0}
    })
    score = cost_score(trace, max_tokens=10000, max_steps=20, max_ms=60000)
    assert score == 1.0


# --- Pareto frontier ---

def test_pareto_frontier_basic():
    results = [
        AgentResult("A", success_rate=0.9, cost=10.0),  # Pareto
        AgentResult("B", success_rate=0.8, cost=5.0),   # Pareto (cheaper)
        AgentResult("C", success_rate=0.7, cost=15.0),  # Dominated by A
        AgentResult("D", success_rate=0.6, cost=3.0),   # Pareto (cheapest)
    ]
    frontier = pareto_frontier(results)
    names = {r.name for r in frontier}
    assert "A" in names  # best success
    assert "B" in names  # better cost trade-off
    assert "D" in names  # cheapest
    assert "C" not in names  # dominated


def test_pareto_frontier_single():
    results = [AgentResult("A", success_rate=0.9, cost=10.0)]
    frontier = pareto_frontier(results)
    assert len(frontier) == 1


def test_pareto_frontier_all_same():
    results = [
        AgentResult("A", success_rate=0.5, cost=5.0),
        AgentResult("B", success_rate=0.5, cost=5.0),
    ]
    frontier = pareto_frontier(results)
    assert len(frontier) == 2  # neither dominates


def test_is_pareto_optimal():
    results = [
        AgentResult("A", success_rate=0.9, cost=10.0),
        AgentResult("B", success_rate=0.8, cost=5.0),
        AgentResult("C", success_rate=0.7, cost=15.0),
    ]
    assert is_pareto_optimal(results[0], results)
    assert is_pareto_optimal(results[1], results)
    assert not is_pareto_optimal(results[2], results)


def test_pareto_ordered_by_cost():
    results = [
        AgentResult("Expensive", success_rate=0.95, cost=50.0),
        AgentResult("Cheap", success_rate=0.70, cost=2.0),
        AgentResult("Mid", success_rate=0.85, cost=10.0),
    ]
    frontier = pareto_frontier(results)
    costs = [r.cost for r in frontier]
    assert costs == sorted(costs)  # ordered by cost ascending
