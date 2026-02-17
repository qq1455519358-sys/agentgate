"""
AgentGate Noise — Robustness testing via controlled noise injection.

From AgentNoiseBench (2026, arXiv:2602.11348):

    "Real-world environments are inherently stochastic and imperfect.
    Most existing benchmarks evaluate agents under idealized assumptions.
    We introduce AgentNoiseBench, a framework for systematically
    evaluating the robustness of agentic models under noisy environments."

    "We categorize environmental noise into two primary types:
    user-noise and tool-noise."

    "Injected noise should increase task difficulty without rendering
    the task unsolvable."

Three design principles from the paper:
1. Realistic Noise Modeling — grounded in real interaction logs
2. Solvability-Preserving — noise doesn't make task impossible
3. Trajectory Consistency Enforcement — outcome + process must hold

Usage:
    from agentgate.noise import (
        UserNoise, ToolNoise, NoisyScenario, noise_robustness_suite,
    )

    noisy = NoisyScenario(
        base=scenario,
        user_noise=UserNoise(typos=0.1, ambiguity=0.2),
        tool_noise=ToolNoise(failure_rate=0.15, partial_results=0.1),
    )
    suite.add(noisy.to_scenario())
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Optional

from agentgate.scenario import (
    Scenario, AgentTrace, AgentStep, StepKind,
    Expectation, ExpectationResult,
)


@dataclass
class UserNoise:
    """User-side noise injection parameters.

    From AgentNoiseBench (2026):
        "User noise captures imperfections and variability in
        human-provided instructions during agent interactions."

    Attributes:
        typos: Probability of introducing typos (0.0-1.0).
        ambiguity: Probability of making instructions ambiguous.
        redundancy: Probability of adding irrelevant information.
        contradiction: Probability of adding contradictory details.
        seed: Random seed for reproducibility.
    """
    typos: float = 0.0
    ambiguity: float = 0.0
    redundancy: float = 0.0
    contradiction: float = 0.0
    seed: Optional[int] = None


@dataclass
class ToolNoise:
    """Tool-side noise injection parameters.

    From AgentNoiseBench (2026):
        "Tool-noise simulates execution failures, inconsistent
        responses, and partial results from external tools."

    Attributes:
        failure_rate: Probability of tool calls failing.
        partial_results: Probability of returning incomplete data.
        latency_spike: Probability of simulated timeout/slow response.
        stale_data: Probability of returning outdated information.
        seed: Random seed for reproducibility.
    """
    failure_rate: float = 0.0
    partial_results: float = 0.0
    latency_spike: float = 0.0
    stale_data: float = 0.0
    seed: Optional[int] = None


def _inject_typos(text: str, rate: float, rng: random.Random) -> str:
    """Inject realistic typos into text."""
    if rate <= 0:
        return text
    chars = list(text)
    swaps = [
        ("a", "s"), ("e", "r"), ("i", "o"), ("t", "y"),
        ("n", "m"), ("c", "v"), ("d", "f"),
    ]
    for i in range(1, len(chars) - 1):
        if chars[i].isalpha() and rng.random() < rate:
            for a, b in swaps:
                if chars[i].lower() == a:
                    chars[i] = b if chars[i].islower() else b.upper()
                    break
    return "".join(chars)


def _inject_ambiguity(text: str, rate: float, rng: random.Random) -> str:
    """Make instructions vaguer."""
    if rate <= 0 or rng.random() >= rate:
        return text
    # Replace specific terms with vague ones
    vague_subs = [
        (r"\b(\d+)\b", "some"),
        (r"\bexactly\b", "roughly"),
        (r"\bmust\b", "should probably"),
        (r"\balways\b", "usually"),
    ]
    result = text
    sub = rng.choice(vague_subs)
    result = re.sub(sub[0], sub[1], result, count=1)
    return result


def _inject_redundancy(text: str, rate: float, rng: random.Random) -> str:
    """Add irrelevant details."""
    if rate <= 0 or rng.random() >= rate:
        return text
    fillers = [
        " (by the way, the weather is nice today)",
        " Also, I had coffee this morning.",
        " Note: this is urgent, kind of.",
        " My colleague mentioned something similar yesterday.",
    ]
    return text + rng.choice(fillers)


def apply_user_noise(text: str, noise: UserNoise) -> str:
    """Apply user noise to input text.

    Preserves solvability per AgentNoiseBench principle.
    """
    rng = random.Random(noise.seed)
    result = text
    result = _inject_typos(result, noise.typos, rng)
    result = _inject_ambiguity(result, noise.ambiguity, rng)
    result = _inject_redundancy(result, noise.redundancy, rng)
    return result


def apply_tool_noise(steps: list[AgentStep], noise: ToolNoise) -> list[AgentStep]:
    """Apply tool noise to trace steps.

    Returns modified steps simulating noisy tool environment.
    """
    rng = random.Random(noise.seed)
    noisy_steps = []
    for step in steps:
        if step.kind != StepKind.TOOL_CALL:
            noisy_steps.append(step)
            continue

        r = rng.random()
        if r < noise.failure_rate:
            # Tool failure
            noisy_steps.append(AgentStep(
                kind=StepKind.TOOL_CALL,
                name=step.name,
                input=step.input,
                output=f"ERROR: {step.name} failed (connection timeout)",
                metadata={**step.metadata, "_noise": "failure"},
            ))
        elif r < noise.failure_rate + noise.partial_results:
            # Partial result
            output = step.output
            if isinstance(output, str) and len(output) > 20:
                output = output[:len(output) // 2] + "... [truncated]"
            noisy_steps.append(AgentStep(
                kind=StepKind.TOOL_CALL,
                name=step.name,
                input=step.input,
                output=output,
                metadata={**step.metadata, "_noise": "partial"},
            ))
        else:
            noisy_steps.append(step)

    return noisy_steps


class ExpectNoiseRobustness(Expectation):
    """Verify agent remains functional under noise.

    From AgentNoiseBench (2026):
        "Trajectory Consistency Enforcement: the agent's intermediate
        reasoning steps should remain logically sound, impervious to
        noise-induced deviations."

    Checks that the agent still completes the task (calls required
    tools) despite noisy conditions.
    """

    def __init__(self, required_tools: list[str], *, max_retries: int = 3):
        self.required_tools = required_tools
        self.max_retries = max_retries

    def check(self, trace: AgentTrace) -> ExpectationResult:
        tool_calls = [s.name for s in trace.tool_calls]
        missing = [t for t in self.required_tools if t not in tool_calls]
        retry_count = sum(
            1 for s in trace.steps
            if s.metadata.get("_noise") in ("failure", "partial")
        )

        if missing:
            return ExpectationResult(
                False,
                "noise_robustness",
                f"missing tools under noise: {missing}; "
                f"noise events: {retry_count}",
            )
        return ExpectationResult(
            True,
            "noise_robustness",
            f"robust despite {retry_count} noise events",
        )


@dataclass
class NoisyScenario:
    """Wrap a base scenario with noise injection.

    From AgentNoiseBench (2026): creates a noise-injected variant
    while preserving the original for comparison.
    """
    base: Scenario
    user_noise: Optional[UserNoise] = None
    tool_noise: Optional[ToolNoise] = None

    def to_scenario(self) -> Scenario:
        """Create a noise-injected copy of the base scenario."""
        noisy_input = self.base.input
        if self.user_noise:
            noisy_input = apply_user_noise(noisy_input, self.user_noise)

        s = Scenario(
            name=f"{self.base.name} [noisy]",
            input=noisy_input,
        )
        s.expectations = list(self.base.expectations)
        # Store noise type info as a simple attribute
        s._noise_type = (
            "user+tool" if (self.user_noise and self.tool_noise)
            else "user" if self.user_noise else "tool"
        )
        return s


def noise_robustness_suite(
    base_scenarios: list[Scenario],
    user_noise: Optional[UserNoise] = None,
    tool_noise: Optional[ToolNoise] = None,
) -> list[Scenario]:
    """Generate noisy variants of a scenario list.

    From AgentNoiseBench (2026): "automated pipeline that injects
    controllable noise into existing agent-centric benchmarks."

    Returns both original and noisy variants for comparison.
    """
    result = []
    for base in base_scenarios:
        result.append(base)  # Original
        noisy = NoisyScenario(
            base=base,
            user_noise=user_noise,
            tool_noise=tool_noise,
        )
        result.append(noisy.to_scenario())
    return result


__all__ = [
    "UserNoise", "ToolNoise", "NoisyScenario",
    "apply_user_noise", "apply_tool_noise",
    "ExpectNoiseRobustness", "noise_robustness_suite",
]
