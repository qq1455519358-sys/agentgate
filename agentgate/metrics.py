"""
AgentGate Metrics — Academic-grounded evaluation metrics.

Implements metrics from recent agent evaluation literature:

- **Node F1** (set-based tool selection F1): Precision/recall of which
    tools were called, ignoring order. Inspired by the "Tool F1 Score"
    in Gabriel et al. (2024) "Advancing Agentic Systems" (NeurIPS 2024
    Workshop), arXiv:2410.22457. We use the term "Node F1" following the
    ICLR 2026 "Hitchhiker's Guide to Agent Evaluation" which generalizes
    it to any tool-call graph node matching.
- **Edge F1** (bigram-based tool ordering F1): Precision/recall of
    consecutive tool-call pairs (directed edges). Related to the
    "Structural Similarity Index" (SSI) in Gabriel et al. (2024), but
    simplified to bigram F1 for practical use.
- **Normalized Edit Distance**: Levenshtein distance between actual and
    expected tool sequences. Recommended by ICLR 2026 Agent Eval Guide
    under "Trajectory Quality" metrics.
- **pass@k / pass^k**: Consistency metrics from τ-bench (Yao et al.,
    2024, Sierra Research). Implemented on SuiteResult in scenario.py.

Usage:
    from agentgate.metrics import node_f1, edge_f1, tool_edit_distance

    trace = agent.run("book a flight")
    expected_tools = ["search_flights", "book_flight"]

    print(f"Node F1: {node_f1(trace, expected_tools):.2f}")
    print(f"Edge F1: {edge_f1(trace, expected_tools):.2f}")
    print(f"Edit Distance: {tool_edit_distance(trace, expected_tools):.2f}")
"""

from __future__ import annotations

from agentgate.scenario import AgentTrace


def node_f1(trace: AgentTrace, expected_tools: list[str]) -> float:
    """Compute Node F1: precision/recall of tool selection (set-based).

    Treats tool calls as an unordered set. Measures whether the agent
    called the right tools (regardless of order or count).

    This corresponds to "Tool F1 Score" in Gabriel et al. (2024),
    "Advancing Agentic Systems" (NeurIPS 2024 Workshop on Open-World
    Agents, arXiv:2410.22457). We adopt the name "Node F1" from the
    ICLR 2026 Agent Evaluation Guide, which uses it to describe
    set-based tool-call matching.

    Args:
        trace: Agent execution trace.
        expected_tools: List of expected tool names.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    actual_set = set(trace.tool_names)
    expected_set = set(expected_tools)

    if not actual_set and not expected_set:
        return 1.0
    if not actual_set or not expected_set:
        return 0.0

    true_positives = len(actual_set & expected_set)
    precision = true_positives / len(actual_set) if actual_set else 0.0
    recall = true_positives / len(expected_set) if expected_set else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def edge_f1(trace: AgentTrace, expected_tools: list[str]) -> float:
    """Compute Edge F1: precision/recall of tool ordering (bigram edges).

    Treats consecutive tool-call pairs as directed edges in a graph.
    Measures whether the agent followed the expected tool ordering.

    Related to the "Structural Similarity Index" (SSI) in Gabriel et al.
    (2024), arXiv:2410.22457, which also measures graph structure fidelity.
    Our simplified version uses bigram F1 rather than full graph isomorphism,
    making it practical for linear tool sequences.

    Also recommended in the ICLR 2026 Agent Eval Guide under
    "Tool-Call Analysis" as a trajectory quality metric.

    Args:
        trace: Agent execution trace.
        expected_tools: Expected tool call sequence (ordered).

    Returns:
        F1 score between 0.0 and 1.0.
    """
    actual = trace.tool_names
    actual_edges = set(zip(actual[:-1], actual[1:])) if len(actual) >= 2 else set()
    expected_edges = set(zip(expected_tools[:-1], expected_tools[1:])) if len(expected_tools) >= 2 else set()

    if not actual_edges and not expected_edges:
        return 1.0
    if not actual_edges or not expected_edges:
        return 0.0

    true_positives = len(actual_edges & expected_edges)
    precision = true_positives / len(actual_edges) if actual_edges else 0.0
    recall = true_positives / len(expected_edges) if expected_edges else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def tool_edit_distance(trace: AgentTrace, expected_tools: list[str]) -> float:
    """Compute normalized edit distance between actual and expected tool sequences.

    Lower is better. 0.0 = exact match, 1.0 = completely different.

    Reference: ICLR 2026 Blog, "A Hitchhiker's Guide to Agent Evaluation"

    Args:
        trace: Agent execution trace.
        expected_tools: Expected tool call sequence.

    Returns:
        Normalized edit distance between 0.0 and 1.0.
    """
    actual = trace.tool_names
    n, m = len(actual), len(expected_tools)

    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1.0

    # Standard Levenshtein
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if actual[i - 1] == expected_tools[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[n][m] / max(n, m)


def side_effect_rate(trace: AgentTrace,
                     allowed_tools: list[str],
                     mutating_tools: list[str] | None = None) -> float:
    """Fraction of tool calls that are unintended side effects.

    From AgentRewardBench (Lù et al., 2025, McGill/Mila):
        "Each trajectory is reviewed by an expert, who answers questions
        pertaining to the success, side effects, and repetitiveness."

    Side effects are tool calls NOT in the allowed list that modify
    state (mutating). If ``mutating_tools`` is provided, only those
    are considered potential side effects. Otherwise, any non-allowed
    tool call is counted.

    Args:
        trace: Agent execution trace.
        allowed_tools: Tools that are expected/allowed for this task.
        mutating_tools: Optional list of tools known to mutate state.
            If provided, only unexpected calls to these tools count as
            side effects (non-mutating extras are ignored).

    Returns:
        Fraction 0.0–1.0. Lower is better. 0.0 = no side effects.
    """
    calls = trace.tool_names
    if not calls:
        return 0.0

    allowed_set = set(allowed_tools)
    unexpected = [t for t in calls if t not in allowed_set]

    if mutating_tools is not None:
        mutating_set = set(mutating_tools)
        side_effects = [t for t in unexpected if t in mutating_set]
    else:
        side_effects = unexpected

    return len(side_effects) / len(calls)


def repetition_rate(trace: AgentTrace) -> float:
    """Detect repetitive action cycles in a trajectory.

    From AgentRewardBench (Lù et al., 2025):
        Expert annotators evaluate "repetitiveness of the agent" —
        whether the agent enters cycles of identical actions.

    Detects consecutive duplicate (tool_name, output) pairs.

    Returns:
        Fraction 0.0–1.0 of steps that are repetitions. Lower is better.
    """
    if len(trace.tool_calls) < 2:
        return 0.0

    calls = [(s.name, str(s.output)) for s in trace.tool_calls]
    repeats = sum(1 for i in range(1, len(calls)) if calls[i] == calls[i - 1])
    return repeats / len(calls)


def decisive_deviation_score(trace: AgentTrace,
                             expected_tools: list[str],
                             mutating_tools: list[str]) -> float:
    """Measure deviation severity weighted by action mutability.

    From SABER (Cuadron et al., 2025, arXiv:2512.07850):
        "Each additional deviation in a mutating action reduces the odds
        of success by up to 92% on Airline and up to 96% on Retail."

    Non-mutating deviations have little effect; mutating deviations are
    catastrophic. This metric counts deviations from expected tools,
    weighting mutating deviations 5x higher than non-mutating ones.

    Args:
        trace: Agent execution trace.
        expected_tools: Expected tool call sequence.
        mutating_tools: Tools known to change environment state
            (e.g., cancel_booking, update_record, delete_file).

    Returns:
        Deviation score 0.0–1.0. Lower is better. 0.0 = perfect match.
    """
    actual = trace.tool_names
    if not expected_tools and not actual:
        return 0.0

    mutating_set = set(mutating_tools)
    max_len = max(len(actual), len(expected_tools))
    if max_len == 0:
        return 0.0

    # Align sequences using edit operations
    n, m = len(actual), len(expected_tools)
    total_penalty = 0.0
    max_penalty = 0.0

    # For each position, check deviations
    # SABER insight: mutating deviations are catastrophic because they
    # change state irreversibly. The weight reflects whether the ACTUAL
    # action is mutating (doing harm) vs the expected action being
    # mutating (missing a necessary mutation).
    for i in range(max(n, m)):
        a = actual[i] if i < n else None
        e = expected_tools[i] if i < m else None

        # Actual mutating action that's wrong = catastrophic (5x)
        # Expected mutating action that's missed = serious (3x)
        # Non-mutating deviation = minor (1x)
        if a != e:
            if a in mutating_set:
                # Agent did a wrong mutating action (most dangerous)
                weight = 5.0
            elif e in mutating_set:
                # Agent missed a necessary mutating action
                weight = 3.0
            else:
                # Non-mutating deviation
                weight = 1.0
            total_penalty += weight

        # Max penalty uses the higher weight for normalization
        is_mutating = (a in mutating_set) or (e in mutating_set)
        max_penalty += 5.0 if is_mutating else 1.0

    return min(1.0, total_penalty / max_penalty) if max_penalty > 0 else 0.0


__all__ = [
    "node_f1",
    "edge_f1",
    "tool_edit_distance",
    "side_effect_rate",
    "repetition_rate",
    "decisive_deviation_score",
]
