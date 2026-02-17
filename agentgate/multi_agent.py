"""
AgentGate Multi-Agent — Evaluate collaboration and coordination quality.

From MultiAgentBench (ACL 2025, arXiv:2503.01935):

    "We introduce MultiAgentBench, a comprehensive benchmark designed
    to evaluate LLM-based multi-agent systems across diverse, interactive
    scenarios. Our framework measures not only task completion but also
    the quality of collaboration and competition."

    "We evaluate various coordination protocols (star, chain, tree,
    and graph topologies) and strategies such as group discussion
    and cognitive planning."

From ValueFlow (2026, arXiv:2602.08567):

    "How value perturbations propagate through agent interactions
    remains poorly understood. We present ValueFlow, a perturbation-
    based evaluation framework for measuring value drift."

    "β-susceptibility measures an agent's sensitivity to perturbed
    peer signals."

Usage:
    from agentgate.multi_agent import (
        AgentRole, MultiAgentTrace, expect_collaboration_quality,
        expect_no_free_riders, coordination_efficiency,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agentgate.scenario import (
    AgentTrace, AgentStep, StepKind,
    Expectation, ExpectationResult,
)


@dataclass
class AgentRole:
    """Define an agent's role in a multi-agent system.

    From MultiAgentBench: agents have distinct roles in
    star/chain/tree/graph topologies.

    Attributes:
        name: Agent identifier.
        role: Role description (e.g., "planner", "executor").
        expected_contributions: Tools or actions this agent should perform.
    """
    name: str
    role: str = ""
    expected_contributions: list[str] = field(default_factory=list)


@dataclass
class MultiAgentTrace:
    """Trace from a multi-agent execution.

    Extends the single-agent trace with per-agent attribution.

    Attributes:
        traces: Dict mapping agent_name → AgentTrace.
        coordination_protocol: The topology used (star/chain/tree/graph).
        total_messages: Total inter-agent messages exchanged.
    """
    traces: dict[str, AgentTrace] = field(default_factory=dict)
    coordination_protocol: str = "star"
    total_messages: int = 0

    @property
    def all_steps(self) -> list[tuple[str, AgentStep]]:
        """All steps across all agents, with agent attribution."""
        result = []
        for agent_name, trace in self.traces.items():
            for step in trace.steps:
                result.append((agent_name, step))
        return result

    @property
    def agent_step_counts(self) -> dict[str, int]:
        """Step count per agent."""
        return {name: len(t.steps) for name, t in self.traces.items()}

    @property
    def agent_tool_calls(self) -> dict[str, list[str]]:
        """Tool calls per agent."""
        return {
            name: [s.name for s in t.tool_calls]
            for name, t in self.traces.items()
        }


def collaboration_quality(mat: MultiAgentTrace) -> float:
    """Measure collaboration quality.

    From MultiAgentBench (ACL 2025): "measures the quality of
    collaboration using milestone-based KPIs."

    Heuristic: good collaboration means balanced contribution
    and minimal redundant work.

    Returns:
        Score 0.0-1.0 where 1.0 = perfect collaboration.
    """
    counts = mat.agent_step_counts
    if not counts:
        return 0.0

    values = list(counts.values())
    if len(values) < 2:
        return 1.0

    # Balance: coefficient of variation (lower = more balanced)
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    cv = (variance ** 0.5) / mean

    # Score: 1.0 when perfectly balanced, decreasing with imbalance
    balance_score = max(0.0, 1.0 - cv)

    # Redundancy: check for duplicate tool calls across agents
    all_tools = mat.agent_tool_calls
    all_tool_names = []
    for tools in all_tools.values():
        all_tool_names.extend(tools)
    total = len(all_tool_names)
    unique = len(set(all_tool_names))
    redundancy = 1.0 - (unique / total) if total > 0 else 0.0
    redundancy_penalty = max(0.0, 1.0 - redundancy)

    return (balance_score + redundancy_penalty) / 2


def coordination_efficiency(mat: MultiAgentTrace) -> float:
    """Measure coordination overhead efficiency.

    Lower message-to-step ratio = more efficient coordination.

    Returns:
        Score 0.0-1.0 where 1.0 = minimal coordination overhead.
    """
    total_steps = sum(mat.agent_step_counts.values())
    if total_steps == 0:
        return 0.0

    # Ideal: 1 message per productive step
    ratio = mat.total_messages / total_steps if total_steps > 0 else float("inf")

    # Score: 1.0 at ratio ≤ 1.0, decreasing as overhead grows
    return max(0.0, min(1.0, 2.0 - ratio))


class ExpectCollaborationQuality(Expectation):
    """Verify multi-agent collaboration meets quality threshold.

    From MultiAgentBench (ACL 2025): "measures the quality of
    collaboration and competition."
    """

    def __init__(self, min_quality: float = 0.5,
                 multi_trace: Optional[MultiAgentTrace] = None):
        self.min_quality = min_quality
        self.multi_trace = multi_trace

    def check(self, trace: AgentTrace) -> ExpectationResult:
        if self.multi_trace is None:
            return ExpectationResult(
                True, "collaboration_quality",
                "no multi-agent trace provided (single agent)",
            )

        quality = collaboration_quality(self.multi_trace)
        if quality >= self.min_quality:
            return ExpectationResult(
                True, "collaboration_quality",
                f"quality={quality:.2f} ≥ {self.min_quality}",
            )
        return ExpectationResult(
            False, "collaboration_quality",
            f"quality={quality:.2f} < {self.min_quality}",
        )


class ExpectNoFreeRiders(Expectation):
    """Verify all agents contributed meaningfully.

    A "free rider" is an agent that consumes coordination
    messages but produces no useful work.

    From MultiAgentBench: milestone-based evaluation ensures
    each agent contributes to progress.
    """

    def __init__(self, roles: list[AgentRole],
                 multi_trace: Optional[MultiAgentTrace] = None):
        self.roles = roles
        self.multi_trace = multi_trace

    def check(self, trace: AgentTrace) -> ExpectationResult:
        if self.multi_trace is None:
            return ExpectationResult(True, "no_free_riders")

        free_riders = []
        agent_tools = self.multi_trace.agent_tool_calls

        for role in self.roles:
            tools = agent_tools.get(role.name, [])
            if not tools and role.expected_contributions:
                free_riders.append(role.name)
            elif role.expected_contributions:
                contributed = any(
                    t in tools for t in role.expected_contributions
                )
                if not contributed:
                    free_riders.append(role.name)

        if free_riders:
            return ExpectationResult(
                False, "no_free_riders",
                f"free riders: {free_riders}",
            )
        return ExpectationResult(True, "no_free_riders")


def expect_collaboration_quality(
    min_quality: float = 0.5,
    multi_trace: Optional[MultiAgentTrace] = None,
) -> ExpectCollaborationQuality:
    """Create a collaboration quality expectation."""
    return ExpectCollaborationQuality(min_quality, multi_trace)


def expect_no_free_riders(
    roles: list[AgentRole],
    multi_trace: Optional[MultiAgentTrace] = None,
) -> ExpectNoFreeRiders:
    """Create a no-free-riders expectation."""
    return ExpectNoFreeRiders(roles, multi_trace)


__all__ = [
    "AgentRole",
    "MultiAgentTrace",
    "collaboration_quality",
    "coordination_efficiency",
    "ExpectCollaborationQuality",
    "ExpectNoFreeRiders",
    "expect_collaboration_quality",
    "expect_no_free_riders",
]
