"""
AgentGate Memory Evaluation — Test agent memory across multi-turn interactions.

From MemoryAgentBench (ICLR 2026, arXiv:2507.05257):

    "Recent benchmarks primarily focus on reasoning, planning, and
    execution capabilities, while another critical component — memory —
    is overlooked."

    Four Core Competencies:
    1. Accurate Retrieval (AR) — can the agent recall stored info?
    2. Test-Time Learning (TTL) — can it learn new facts during interaction?
    3. Long-Range Understanding (LRU) — does it maintain context over time?
    4. Conflict Resolution (CR) — can it handle contradictory information?

Also from the Unified Framework paper (2026, arXiv:2602.03238):

    "Evaluation outcomes are influenced not only by the LLM, but also
    by memory mechanisms... a unified evaluation standard is necessary."

Usage:
    from agentgate.memory_eval import (
        MemoryProbe, memory_consistency_suite,
        expect_accurate_retrieval, expect_conflict_resolution,
    )

    s = Scenario("Memory test", input="What did I say earlier?")
    s.expectations.append(expect_accurate_retrieval(
        stored_fact="user prefers dark mode",
        query="what are my preferences?",
    ))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agentgate.scenario import (
    Scenario, AgentTrace, AgentStep, StepKind,
    Expectation, ExpectationResult,
)


@dataclass
class MemoryProbe:
    """A memory probe: inject a fact, then query for it later.

    From MemoryAgentBench (ICLR 2026):
        "inject once, query multiple times" design philosophy.

    Attributes:
        fact: The fact to inject into agent memory.
        queries: Questions that should be answerable from the fact.
        expected_answers: Expected answer keywords for each query.
        turns_between: Number of turns between injection and query.
        competency: Which competency this tests (AR/TTL/LRU/CR).
    """
    fact: str
    queries: list[str] = field(default_factory=list)
    expected_answers: list[str] = field(default_factory=list)
    turns_between: int = 0
    competency: str = "AR"  # AR, TTL, LRU, CR


class ExpectAccurateRetrieval(Expectation):
    """Verify the agent can retrieve previously stored information.

    MemoryAgentBench competency: Accurate Retrieval (AR)
    "Can the agent recall stored info?"
    """

    def __init__(self, stored_fact: str, query: str,
                 expected_keywords: Optional[list[str]] = None):
        self.stored_fact = stored_fact
        self.query = query
        self.expected_keywords = expected_keywords or []

    def check(self, trace: AgentTrace) -> ExpectationResult:
        # Look for memory recall in any agent step
        for step in trace.steps:
            output = str(step.output).lower()
            if any(kw.lower() in output for kw in self.expected_keywords):
                return ExpectationResult(True, "accurate_retrieval")

        # Check final output / metadata
        final_output = ""
        if trace.steps:
            final_output = str(trace.steps[-1].output).lower()
        if "memory_response" in trace.metadata:
            final_output = str(trace.metadata["memory_response"]).lower()

        if self.expected_keywords:
            found = [kw for kw in self.expected_keywords
                     if kw.lower() in final_output]
            if found:
                return ExpectationResult(
                    True, "accurate_retrieval",
                    f"found: {found}",
                )
            return ExpectationResult(
                False, "accurate_retrieval",
                f"expected {self.expected_keywords}, not found in output",
            )

        return ExpectationResult(True, "accurate_retrieval")


class ExpectConflictResolution(Expectation):
    """Verify the agent handles contradictory information correctly.

    MemoryAgentBench competency: Conflict Resolution (CR)
    "Can it handle contradictory information?"

    When given conflicting facts, the agent should either:
    - Use the most recent fact (recency)
    - Explicitly acknowledge the conflict
    - Ask for clarification
    """

    def __init__(self, old_fact: str, new_fact: str,
                 expected_resolution: str = "recency"):
        self.old_fact = old_fact
        self.new_fact = new_fact
        self.expected_resolution = expected_resolution

    def check(self, trace: AgentTrace) -> ExpectationResult:
        all_output = " ".join(str(s.output) for s in trace.steps).lower()

        conflict_signals = [
            "conflict", "contradict", "previously", "updated",
            "changed", "however", "but earlier", "clarif",
        ]
        acknowledged = any(sig in all_output for sig in conflict_signals)

        if self.expected_resolution == "recency":
            # New fact should be used
            if self.new_fact.lower() in all_output:
                return ExpectationResult(True, "conflict_resolution",
                                         "used most recent fact")
            elif acknowledged:
                return ExpectationResult(True, "conflict_resolution",
                                         "acknowledged conflict")
            return ExpectationResult(
                False, "conflict_resolution",
                f"neither used new fact nor acknowledged conflict",
            )
        elif self.expected_resolution == "acknowledge":
            if acknowledged:
                return ExpectationResult(True, "conflict_resolution")
            return ExpectationResult(
                False, "conflict_resolution",
                "did not acknowledge conflicting information",
            )

        return ExpectationResult(True, "conflict_resolution")


class ExpectMemoryConsistency(Expectation):
    """Verify agent maintains consistent answers across queries.

    MemoryAgentBench competency: Long-Range Understanding (LRU)
    "Does it maintain context over time?"

    Given multiple queries about the same fact, answers should
    not contradict each other.
    """

    def __init__(self, consistency_key: str):
        self.consistency_key = consistency_key

    def check(self, trace: AgentTrace) -> ExpectationResult:
        # Collect all responses related to the key
        responses = []
        for step in trace.steps:
            output = str(step.output).lower()
            if self.consistency_key.lower() in output:
                responses.append(output)

        if len(responses) < 2:
            return ExpectationResult(
                True, "memory_consistency",
                "insufficient data points (need ≥2 mentions)",
            )

        # Check for contradictions between first and last mention
        # Simple heuristic: check for negation patterns
        first = responses[0]
        last = responses[-1]

        contradiction_pairs = [
            ("yes", "no"), ("true", "false"), ("correct", "incorrect"),
            ("can", "cannot"), ("is", "is not"), ("will", "won't"),
        ]
        for pos, neg in contradiction_pairs:
            if (pos in first and neg in last) or (neg in first and pos in last):
                return ExpectationResult(
                    False, "memory_consistency",
                    f"contradiction detected: '{pos}'/'{neg}' "
                    f"across {len(responses)} mentions",
                )

        return ExpectationResult(
            True, "memory_consistency",
            f"consistent across {len(responses)} mentions",
        )


def expect_accurate_retrieval(
    stored_fact: str,
    query: str,
    expected_keywords: Optional[list[str]] = None,
) -> ExpectAccurateRetrieval:
    """Create an accurate retrieval expectation."""
    return ExpectAccurateRetrieval(stored_fact, query, expected_keywords)


def expect_conflict_resolution(
    old_fact: str,
    new_fact: str,
    expected_resolution: str = "recency",
) -> ExpectConflictResolution:
    """Create a conflict resolution expectation."""
    return ExpectConflictResolution(old_fact, new_fact, expected_resolution)


def expect_memory_consistency(key: str) -> ExpectMemoryConsistency:
    """Create a memory consistency expectation."""
    return ExpectMemoryConsistency(key)


def memory_consistency_suite(
    probes: list[MemoryProbe],
) -> list[Scenario]:
    """Generate a memory evaluation suite from probes.

    From MemoryAgentBench (ICLR 2026):
        "inject once, query multiple times" — one fact
        corresponds to multiple questions.

    Returns scenarios testing each probe's competency.
    """
    scenarios = []
    for i, probe in enumerate(probes):
        for j, query in enumerate(probe.queries):
            s = Scenario(
                name=f"Memory {probe.competency} #{i+1}.{j+1}",
                input=query,
            )
            # Store probe metadata as attributes
            s._memory_meta = {
                "injected_fact": probe.fact,
                "competency": probe.competency,
                "turns_between": probe.turns_between,
            }
            if probe.expected_answers and j < len(probe.expected_answers):
                s.expectations.append(
                    expect_accurate_retrieval(
                        stored_fact=probe.fact,
                        query=query,
                        expected_keywords=[probe.expected_answers[j]],
                    )
                )
            scenarios.append(s)
    return scenarios


__all__ = [
    "MemoryProbe",
    "ExpectAccurateRetrieval",
    "ExpectConflictResolution",
    "ExpectMemoryConsistency",
    "expect_accurate_retrieval",
    "expect_conflict_resolution",
    "expect_memory_consistency",
    "memory_consistency_suite",
]
