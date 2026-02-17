"""
AgentGate Trajectory Analysis — Graph-based and reward-propagation metrics.

Implements ideas from:

- **WebGraphEval** (Qian et al., 2025, NeurIPS 2025 Workshop):
    "Abstracts trajectories into a unified, weighted action graph...
    identifies critical decision points overlooked by outcome-based metrics."

- **RewardFlow** (ICLR 2026 submission):
    "Propagates terminal rewards from successful states to all visited
    states using graph propagation... produces dense, state-wise,
    task-centric reward signals."

Key concepts:
- Critical decision points: steps where success/failure diverges
- Reward propagation: assign credit to each step based on outcome
- Trajectory redundancy: detect wasted or unnecessary steps

Usage:
    from agentgate.trajectory import (
        critical_steps, step_credit, trajectory_redundancy,
    )
"""

from __future__ import annotations

from agentgate.scenario import AgentTrace, StepKind


def step_credit(trace: AgentTrace,
                expected_tools: list[str]) -> list[float]:
    """Assign credit to each step based on alignment with expected trajectory.

    Inspired by RewardFlow (ICLR 2026):
        "Produces dense, state-wise, task-centric reward signals that
        indicate whether actions move the agent closer to or farther
        from success."

    Each step gets a credit score:
    - +1.0: step matches expected tool at the right position
    - +0.5: step uses a correct tool but out of order
    - 0.0: step is neutral (LLM call, non-tool)
    - -0.5: step uses an unexpected tool
    - -1.0: step uses a tool that exists in the trace but causes backtracking

    Args:
        trace: Agent execution trace.
        expected_tools: Expected tool call sequence.

    Returns:
        List of credit scores, one per step.
    """
    expected_set = set(expected_tools)
    credits: list[float] = []
    tool_idx = 0  # pointer into expected_tools

    for step in trace.steps:
        if step.kind != StepKind.TOOL_CALL:
            credits.append(0.0)
            continue

        name = step.name

        if tool_idx < len(expected_tools) and name == expected_tools[tool_idx]:
            # Perfect: right tool at right time
            credits.append(1.0)
            tool_idx += 1
        elif name in expected_set:
            # Correct tool but out of order
            credits.append(0.5)
            # Advance pointer if possible
            try:
                future_idx = expected_tools.index(name, tool_idx)
                tool_idx = future_idx + 1
            except ValueError:
                pass
        elif step.error:
            # Error step
            credits.append(-1.0)
        else:
            # Unexpected tool
            credits.append(-0.5)

    return credits


def critical_steps(trace: AgentTrace,
                   expected_tools: list[str]) -> list[int]:
    """Identify critical decision points in the trajectory.

    From WebGraphEval (Qian et al., 2025, NeurIPS 2025):
        "Identifies critical decision points overlooked by
        outcome-based metrics."

    A step is "critical" if:
    1. It's a tool call that deviates from the expected sequence, OR
    2. It's the first correct step after a deviation (recovery point), OR
    3. It's a mutating action (state_change)

    Args:
        trace: Agent execution trace.
        expected_tools: Expected tool sequence.

    Returns:
        List of 0-based step indices that are critical decision points.
    """
    credits = step_credit(trace, expected_tools)
    critical: list[int] = []
    prev_was_deviation = False

    for i, (step, credit) in enumerate(zip(trace.steps, credits)):
        is_critical = False

        # Deviation from expected
        if credit < 0:
            is_critical = True
            prev_was_deviation = True

        # Recovery after deviation
        elif credit > 0 and prev_was_deviation:
            is_critical = True
            prev_was_deviation = False

        # State changes are always critical
        elif step.kind == StepKind.STATE_CHANGE:
            is_critical = True

        else:
            prev_was_deviation = False

        if is_critical:
            critical.append(i)

    return critical


def trajectory_redundancy(trace: AgentTrace,
                          expected_tools: list[str]) -> float:
    """Measure trajectory redundancy (fraction of wasted steps).

    From WebGraphEval (Qian et al., 2025):
        "Highlights redundancy and inefficiency" in agent trajectories.

    A step is redundant if:
    - It's a tool call not in the expected set
    - It's a repeated call (same tool+output as previous)
    - It's an error that was retried

    Args:
        trace: Agent execution trace.
        expected_tools: Expected tool sequence.

    Returns:
        Fraction of redundant steps (0.0–1.0). Lower is better.
    """
    if not trace.steps:
        return 0.0

    expected_set = set(expected_tools)
    redundant = 0
    prev_call = None

    for step in trace.steps:
        if step.kind != StepKind.TOOL_CALL:
            continue

        call_sig = (step.name, str(step.output))

        # Repeated call
        if call_sig == prev_call:
            redundant += 1
        # Unexpected tool
        elif step.name not in expected_set:
            redundant += 1
        # Error
        elif step.error:
            redundant += 1

        prev_call = call_sig

    total_tool_calls = len(trace.tool_calls)
    return redundant / total_tool_calls if total_tool_calls > 0 else 0.0


def trajectory_efficiency(trace: AgentTrace,
                          expected_tools: list[str]) -> float:
    """Compute trajectory efficiency: how close to optimal path.

    Combines credit assignment with path length comparison.
    Efficiency = (sum of positive credits) / len(expected_tools)
    Capped at 1.0 (can't be more efficient than optimal).

    Args:
        trace: Agent execution trace.
        expected_tools: Expected tool sequence.

    Returns:
        Efficiency score 0.0–1.0.
    """
    if not expected_tools:
        return 1.0 if not trace.tool_calls else 0.0

    credits = step_credit(trace, expected_tools)
    positive_credit = sum(c for c in credits if c > 0)
    return min(1.0, positive_credit / len(expected_tools))


__all__ = [
    "step_credit",
    "critical_steps",
    "trajectory_redundancy",
    "trajectory_efficiency",
]
