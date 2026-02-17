"""Test multi-agent collaboration metrics (MultiAgentBench ACL 2025)."""
from agentgate import (
    AgentTrace, AgentStep, StepKind,
    AgentRole, MultiAgentTrace,
    collaboration_quality, coordination_efficiency,
    expect_collaboration_quality, expect_no_free_riders,
)


def _balanced_mat():
    """Balanced multi-agent trace — good collaboration."""
    return MultiAgentTrace(
        traces={
            "planner": AgentTrace(input="plan", steps=[
                AgentStep(kind=StepKind.TOOL_CALL, name="analyze", output="plan"),
                AgentStep(kind=StepKind.TOOL_CALL, name="schedule", output="done"),
            ]),
            "executor": AgentTrace(input="exec", steps=[
                AgentStep(kind=StepKind.TOOL_CALL, name="deploy", output="ok"),
                AgentStep(kind=StepKind.TOOL_CALL, name="verify", output="ok"),
            ]),
        },
        coordination_protocol="star",
        total_messages=3,
    )


def _imbalanced_mat():
    """Imbalanced — one agent does all work."""
    return MultiAgentTrace(
        traces={
            "worker": AgentTrace(input="work", steps=[
                AgentStep(kind=StepKind.TOOL_CALL, name="a", output="1"),
                AgentStep(kind=StepKind.TOOL_CALL, name="b", output="2"),
                AgentStep(kind=StepKind.TOOL_CALL, name="c", output="3"),
                AgentStep(kind=StepKind.TOOL_CALL, name="d", output="4"),
                AgentStep(kind=StepKind.TOOL_CALL, name="e", output="5"),
            ]),
            "slacker": AgentTrace(input="idle", steps=[]),
        },
        total_messages=5,
    )


def _redundant_mat():
    """Redundant — agents duplicate each other's work."""
    return MultiAgentTrace(
        traces={
            "agent_a": AgentTrace(input="q", steps=[
                AgentStep(kind=StepKind.TOOL_CALL, name="search", output="r1"),
                AgentStep(kind=StepKind.TOOL_CALL, name="analyze", output="a1"),
            ]),
            "agent_b": AgentTrace(input="q", steps=[
                AgentStep(kind=StepKind.TOOL_CALL, name="search", output="r2"),
                AgentStep(kind=StepKind.TOOL_CALL, name="analyze", output="a2"),
            ]),
        },
        total_messages=2,
    )


def test_collaboration_quality_balanced():
    """Balanced contribution → high quality."""
    mat = _balanced_mat()
    quality = collaboration_quality(mat)
    assert quality > 0.7


def test_collaboration_quality_imbalanced():
    """One agent does everything → low quality."""
    mat = _imbalanced_mat()
    quality = collaboration_quality(mat)
    assert quality <= 0.5


def test_collaboration_quality_redundant():
    """Duplicated work → reduced quality."""
    mat = _redundant_mat()
    quality = collaboration_quality(mat)
    # Balanced but redundant
    assert 0.3 < quality < 0.8


def test_collaboration_quality_empty():
    """Empty trace → 0."""
    mat = MultiAgentTrace()
    assert collaboration_quality(mat) == 0.0


def test_collaboration_quality_single_agent():
    """Single agent → perfect by default."""
    mat = MultiAgentTrace(
        traces={"solo": AgentTrace(input="q", steps=[
            AgentStep(kind=StepKind.TOOL_CALL, name="do", output="done"),
        ])},
    )
    assert collaboration_quality(mat) == 1.0


def test_coordination_efficiency_good():
    """Few messages for many steps → efficient."""
    mat = _balanced_mat()  # 3 messages, 4 steps
    eff = coordination_efficiency(mat)
    assert eff > 0.5


def test_coordination_efficiency_bad():
    """Many messages, few productive steps."""
    mat = MultiAgentTrace(
        traces={"a": AgentTrace(input="q", steps=[
            AgentStep(kind=StepKind.TOOL_CALL, name="x", output="ok"),
        ])},
        total_messages=10,
    )
    eff = coordination_efficiency(mat)
    assert eff == 0.0  # ratio=10, score=max(0, 2-10)=0


def test_expect_collaboration_quality_pass():
    """Quality above threshold."""
    mat = _balanced_mat()
    exp = expect_collaboration_quality(min_quality=0.5, multi_trace=mat)
    # Check with any trace (multi_trace is pre-set)
    dummy = AgentTrace(input="q")
    result = exp.check(dummy)
    assert result.passed


def test_expect_collaboration_quality_fail():
    """Quality below threshold."""
    mat = _imbalanced_mat()
    exp = expect_collaboration_quality(min_quality=0.8, multi_trace=mat)
    result = exp.check(AgentTrace(input="q"))
    assert not result.passed


def test_expect_no_free_riders_pass():
    """All agents contribute."""
    mat = _balanced_mat()
    roles = [
        AgentRole("planner", "planning", ["analyze"]),
        AgentRole("executor", "executing", ["deploy"]),
    ]
    exp = expect_no_free_riders(roles, multi_trace=mat)
    assert exp.check(AgentTrace(input="q")).passed


def test_expect_no_free_riders_fail():
    """Slacker agent detected."""
    mat = _imbalanced_mat()
    roles = [
        AgentRole("worker", "working", ["a"]),
        AgentRole("slacker", "should help", ["assist"]),
    ]
    exp = expect_no_free_riders(roles, multi_trace=mat)
    result = exp.check(AgentTrace(input="q"))
    assert not result.passed
    assert "slacker" in result.detail


def test_multi_agent_trace_properties():
    """Test MultiAgentTrace utility properties."""
    mat = _balanced_mat()
    assert len(mat.all_steps) == 4
    assert mat.agent_step_counts == {"planner": 2, "executor": 2}
    assert "analyze" in mat.agent_tool_calls["planner"]
    assert "deploy" in mat.agent_tool_calls["executor"]


def test_no_multi_trace_graceful():
    """Without multi_trace, expectations pass gracefully."""
    exp1 = expect_collaboration_quality(0.5)
    exp2 = expect_no_free_riders([AgentRole("a")])
    assert exp1.check(AgentTrace(input="q")).passed
    assert exp2.check(AgentTrace(input="q")).passed
