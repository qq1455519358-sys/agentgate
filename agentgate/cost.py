"""
AgentGate Cost & Efficiency — Token-aware evaluation metrics.

Implements cost-efficiency evaluation from recent literature:

- **Cost Score**: Normalized cost metric combining token usage, latency,
    and step count. From Yang et al. (2026) "Toward Efficient Agents"
    (arXiv:2601.14192): "characterize efficiency in two complementary
    ways: comparing effectiveness under a fixed cost budget, and
    comparing cost at a comparable level of effectiveness."

- **Pareto Dominance**: Check if one agent run Pareto-dominates another
    across (success, cost) dimensions. From HAL (Holistic Agent
    Leaderboard) which "emphasizes Pareto frontiers of accuracy vs.
    inference cost" (ICLR 2026 Agent Eval Guide).

- **Token Budget Enforcement**: From Liu et al. (2026) "Cost-effective
    Agent Test-time Scaling via Budget-Aware Thinking" — enforce token
    budgets as a first-class constraint.

Usage:
    from agentgate.cost import token_cost, cost_score, is_pareto_optimal

    trace.metadata["token_usage"] = {"input": 2000, "output": 500}
    print(f"Cost: ${token_cost(trace, input_price=3.0, output_price=15.0):.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from agentgate.scenario import AgentTrace


@dataclass
class TokenUsage:
    """Token usage for a single trace.

    Can be extracted from trace.metadata["token_usage"] or
    trace.metadata["usage"] (OpenAI-style).
    """
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_trace(cls, trace: AgentTrace) -> "TokenUsage":
        """Extract token usage from trace metadata.

        Looks for common keys: "token_usage", "usage",
        "prompt_tokens"/"completion_tokens".
        """
        meta = trace.metadata

        # Direct token_usage dict
        if "token_usage" in meta:
            u = meta["token_usage"]
            inp = u.get("input", u.get("prompt_tokens", u.get("input_tokens", 0)))
            out = u.get("output", u.get("completion_tokens", u.get("output_tokens", 0)))
            return cls(input_tokens=inp, output_tokens=out, total_tokens=inp + out)

        # OpenAI-style "usage"
        if "usage" in meta:
            u = meta["usage"]
            inp = u.get("prompt_tokens", 0)
            out = u.get("completion_tokens", 0)
            return cls(input_tokens=inp, output_tokens=out, total_tokens=inp + out)

        # Flat keys
        inp = meta.get("input_tokens", meta.get("prompt_tokens", 0))
        out = meta.get("output_tokens", meta.get("completion_tokens", 0))
        total = meta.get("total_tokens", inp + out)

        # Also accumulate from steps
        for step in trace.steps:
            step_meta = step.metadata
            if "tokens" in step_meta:
                total += step_meta["tokens"]
            if "input_tokens" in step_meta:
                inp += step_meta["input_tokens"]
            if "output_tokens" in step_meta:
                out += step_meta["output_tokens"]

        return cls(input_tokens=inp, output_tokens=out,
                   total_tokens=total if total > 0 else inp + out)


def token_cost(trace: AgentTrace, *,
               input_price: float = 3.0,
               output_price: float = 15.0) -> float:
    """Compute dollar cost of a trace based on token pricing.

    Default prices reflect Claude Sonnet 4 pricing ($3/$15 per 1M tokens).

    Args:
        trace: Agent execution trace with token metadata.
        input_price: Cost per 1M input tokens.
        output_price: Cost per 1M output tokens.

    Returns:
        Cost in dollars.
    """
    usage = TokenUsage.from_trace(trace)
    return (usage.input_tokens * input_price + usage.output_tokens * output_price) / 1_000_000


def cost_score(trace: AgentTrace, *,
               max_tokens: int = 10000,
               max_steps: int = 20,
               max_ms: float = 60000,
               token_weight: float = 0.4,
               step_weight: float = 0.3,
               time_weight: float = 0.3) -> float:
    """Composite efficiency score (0.0–1.0, higher = more efficient).

    From Yang et al. (2026) "Toward Efficient Agents":
        "efficiency in two complementary ways: comparing effectiveness
        under a fixed cost budget, and comparing cost at a comparable
        level of effectiveness."

    Normalizes tokens, steps, and latency against budgets, then
    computes a weighted average. Exceeding a budget caps that
    component at 0.0 (not penalized further).

    Args:
        trace: Agent execution trace.
        max_tokens: Token budget.
        max_steps: Step budget.
        max_ms: Time budget in milliseconds.
        token_weight: Weight for token efficiency (default 0.4).
        step_weight: Weight for step efficiency (default 0.3).
        time_weight: Weight for time efficiency (default 0.3).

    Returns:
        Efficiency score 0.0–1.0.
    """
    usage = TokenUsage.from_trace(trace)
    token_eff = max(0.0, 1.0 - usage.total_tokens / max_tokens) if max_tokens > 0 else 1.0
    step_eff = max(0.0, 1.0 - len(trace.steps) / max_steps) if max_steps > 0 else 1.0
    time_eff = max(0.0, 1.0 - trace.total_duration_ms / max_ms) if max_ms > 0 else 1.0

    return (token_weight * token_eff +
            step_weight * step_eff +
            time_weight * time_eff)


@dataclass
class AgentResult:
    """(success_rate, cost) point for Pareto analysis."""
    name: str
    success_rate: float
    cost: float  # dollar cost or token count


def pareto_frontier(results: Sequence[AgentResult]) -> list[AgentResult]:
    """Compute the Pareto frontier (maximize success, minimize cost).

    From HAL (Holistic Agent Leaderboard) / ICLR 2026:
        "HAL emphasizes Pareto frontiers of accuracy vs. inference cost."

    Returns agents that are not dominated by any other agent.
    An agent A dominates B if A.success_rate >= B.success_rate AND
    A.cost <= B.cost (with at least one strict inequality).
    """
    frontier: list[AgentResult] = []
    sorted_results = sorted(results, key=lambda r: (-r.success_rate, r.cost))

    for candidate in sorted_results:
        dominated = False
        for other in sorted_results:
            if other is candidate:
                continue
            if (other.success_rate >= candidate.success_rate and
                other.cost <= candidate.cost and
                (other.success_rate > candidate.success_rate or
                 other.cost < candidate.cost)):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)

    return sorted(frontier, key=lambda r: r.cost)


def is_pareto_optimal(agent: AgentResult,
                      others: Sequence[AgentResult]) -> bool:
    """Check if an agent is on the Pareto frontier."""
    for other in others:
        if other.name == agent.name:
            continue
        if (other.success_rate >= agent.success_rate and
            other.cost <= agent.cost and
            (other.success_rate > agent.success_rate or
             other.cost < agent.cost)):
            return False
    return True


__all__ = [
    "TokenUsage",
    "token_cost",
    "cost_score",
    "AgentResult",
    "pareto_frontier",
    "is_pareto_optimal",
]
