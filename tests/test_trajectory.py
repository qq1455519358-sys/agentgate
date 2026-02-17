"""Test trajectory analysis (WebGraphEval + RewardFlow)."""
from agentgate import (
    AgentTrace, AgentStep, StepKind,
    step_credit, critical_steps, trajectory_redundancy, trajectory_efficiency,
)


def _trace(*tool_names, **kwargs):
    steps = [AgentStep(kind=StepKind.TOOL_CALL, name=n, output="ok")
             for n in tool_names]
    return AgentTrace(input="t", steps=steps, **kwargs)


# ═══════════════════════════════════════════════════════════════
# step_credit (RewardFlow-inspired)
# ═══════════════════════════════════════════════════════════════

def test_credit_perfect_match():
    trace = _trace("search", "select", "book")
    credits = step_credit(trace, ["search", "select", "book"])
    assert credits == [1.0, 1.0, 1.0]


def test_credit_unexpected_tool():
    trace = _trace("search", "random", "book")
    credits = step_credit(trace, ["search", "select", "book"])
    assert credits[0] == 1.0    # correct
    assert credits[1] == -0.5   # unexpected
    assert credits[2] == 0.5    # correct but out of order


def test_credit_empty():
    trace = _trace()
    assert step_credit(trace, ["search"]) == []


def test_credit_extra_steps():
    trace = _trace("search", "log", "log", "book")
    credits = step_credit(trace, ["search", "book"])
    assert credits[0] == 1.0   # correct
    assert credits[1] == -0.5  # unexpected
    assert credits[2] == -0.5  # unexpected
    assert credits[3] == 1.0   # correct (matched after advancing)


def test_credit_non_tool_steps():
    trace = AgentTrace(input="t", steps=[
        AgentStep(kind=StepKind.LLM_CALL, name="think", output="hmm"),
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="ok"),
    ])
    credits = step_credit(trace, ["search"])
    assert credits == [0.0, 1.0]


def test_credit_with_errors():
    trace = AgentTrace(input="t", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="bad_tool", output=None, error="failed"),
    ])
    credits = step_credit(trace, ["search"])
    assert credits == [-1.0]


# ═══════════════════════════════════════════════════════════════
# critical_steps (WebGraphEval-inspired)
# ═══════════════════════════════════════════════════════════════

def test_critical_steps_no_deviations():
    trace = _trace("search", "book")
    critical = critical_steps(trace, ["search", "book"])
    assert critical == []  # no deviations = no critical points


def test_critical_steps_deviation():
    trace = _trace("search", "wrong_tool", "book")
    critical = critical_steps(trace, ["search", "select", "book"])
    # wrong_tool at idx 1 is a deviation, book at idx 2 is recovery
    assert 1 in critical  # deviation
    assert 2 in critical  # recovery


def test_critical_steps_state_change():
    trace = AgentTrace(input="t", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="ok"),
        AgentStep(kind=StepKind.STATE_CHANGE, name="balance", output="$0"),
    ])
    critical = critical_steps(trace, ["search"])
    assert 1 in critical  # state changes are always critical


def test_critical_steps_empty():
    trace = _trace()
    assert critical_steps(trace, []) == []


# ═══════════════════════════════════════════════════════════════
# trajectory_redundancy (WebGraphEval-inspired)
# ═══════════════════════════════════════════════════════════════

def test_redundancy_zero():
    trace = _trace("search", "book")
    assert trajectory_redundancy(trace, ["search", "book"]) == 0.0


def test_redundancy_repeats():
    trace = _trace("search", "search", "book")
    # 1 repeat out of 3
    assert abs(trajectory_redundancy(trace, ["search", "book"]) - 1/3) < 0.01


def test_redundancy_unexpected():
    trace = _trace("search", "random", "book")
    # 1 unexpected out of 3
    assert abs(trajectory_redundancy(trace, ["search", "book"]) - 1/3) < 0.01


def test_redundancy_empty():
    assert trajectory_redundancy(_trace(), ["search"]) == 0.0


def test_redundancy_all_wasted():
    trace = _trace("x", "x", "x")
    # All are unexpected or repeated
    assert trajectory_redundancy(trace, ["search"]) == 1.0


# ═══════════════════════════════════════════════════════════════
# trajectory_efficiency
# ═══════════════════════════════════════════════════════════════

def test_efficiency_perfect():
    trace = _trace("search", "select", "book")
    assert trajectory_efficiency(trace, ["search", "select", "book"]) == 1.0


def test_efficiency_partial():
    trace = _trace("search")
    eff = trajectory_efficiency(trace, ["search", "select", "book"])
    assert abs(eff - 1/3) < 0.01  # 1 correct out of 3 expected


def test_efficiency_none():
    trace = _trace("random")
    assert trajectory_efficiency(trace, ["search", "book"]) == 0.0


def test_efficiency_empty_expected():
    trace = _trace()
    assert trajectory_efficiency(trace, []) == 1.0


def test_efficiency_with_extra_steps():
    """Extra steps don't reduce efficiency, they just don't help."""
    trace = _trace("search", "log", "book")
    eff = trajectory_efficiency(trace, ["search", "book"])
    assert eff == 1.0  # both expected tools were called correctly


def test_efficiency_capped_at_one():
    """Efficiency can't exceed 1.0 even with repeated correct tools."""
    trace = _trace("search", "search", "book", "book")
    eff = trajectory_efficiency(trace, ["search", "book"])
    assert eff <= 1.0
