"""Test side effects, repetition, and SABER deviation metrics."""
from agentgate import (
    Scenario, TestSuite, MockAgent, AgentTrace, AgentStep, StepKind,
    side_effect_rate, repetition_rate, decisive_deviation_score,
)


# ═══════════════════════════════════════════════════════════════
# side_effect_rate (AgentRewardBench, Lù et al. 2025)
# ═══════════════════════════════════════════════════════════════

def test_no_side_effects():
    trace = AgentTrace(input="t", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="ok"),
        AgentStep(kind=StepKind.TOOL_CALL, name="book", output="ok"),
    ])
    assert side_effect_rate(trace, ["search", "book"]) == 0.0


def test_all_side_effects():
    trace = AgentTrace(input="t", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="delete_all", output="done"),
        AgentStep(kind=StepKind.TOOL_CALL, name="format_disk", output="done"),
    ])
    assert side_effect_rate(trace, ["search"]) == 1.0


def test_partial_side_effects():
    trace = AgentTrace(input="t", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="ok"),
        AgentStep(kind=StepKind.TOOL_CALL, name="delete_user", output="oops"),
    ])
    assert abs(side_effect_rate(trace, ["search"]) - 0.5) < 0.01


def test_side_effects_mutating_filter():
    """Only mutating tools count as side effects when filter is set."""
    trace = AgentTrace(input="t", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="ok"),
        AgentStep(kind=StepKind.TOOL_CALL, name="log_event", output="logged"),  # non-mutating extra
        AgentStep(kind=StepKind.TOOL_CALL, name="delete_record", output="deleted"),  # mutating extra
    ])
    # Without filter: 2/3 are unexpected
    assert abs(side_effect_rate(trace, ["search"]) - 2/3) < 0.01
    # With filter: only delete_record counts
    assert abs(side_effect_rate(trace, ["search"],
               mutating_tools=["delete_record", "update_record"]) - 1/3) < 0.01


def test_side_effects_empty():
    trace = AgentTrace(input="t", steps=[])
    assert side_effect_rate(trace, ["search"]) == 0.0


# ═══════════════════════════════════════════════════════════════
# repetition_rate (AgentRewardBench)
# ═══════════════════════════════════════════════════════════════

def test_no_repetition():
    trace = AgentTrace(input="t", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="a", output="1"),
        AgentStep(kind=StepKind.TOOL_CALL, name="b", output="2"),
        AgentStep(kind=StepKind.TOOL_CALL, name="c", output="3"),
    ])
    assert repetition_rate(trace) == 0.0


def test_full_repetition():
    trace = AgentTrace(input="t", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="nothing"),
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="nothing"),
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="nothing"),
    ])
    # 2 repetitions out of 3 calls
    assert abs(repetition_rate(trace) - 2/3) < 0.01


def test_partial_repetition():
    trace = AgentTrace(input="t", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="nothing"),
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="nothing"),
        AgentStep(kind=StepKind.TOOL_CALL, name="book", output="ok"),
        AgentStep(kind=StepKind.TOOL_CALL, name="book", output="ok"),
    ])
    # 2 repetitions out of 4 calls
    assert abs(repetition_rate(trace) - 0.5) < 0.01


def test_same_tool_different_output():
    """Same tool but different output = not a repetition."""
    trace = AgentTrace(input="t", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="result1"),
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="result2"),
    ])
    assert repetition_rate(trace) == 0.0


# ═══════════════════════════════════════════════════════════════
# decisive_deviation_score (SABER, Cuadron et al. 2025)
# ═══════════════════════════════════════════════════════════════

def test_deviation_perfect_match():
    trace = AgentTrace(input="t", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="ok"),
        AgentStep(kind=StepKind.TOOL_CALL, name="book", output="ok"),
    ])
    score = decisive_deviation_score(trace, ["search", "book"],
                                      mutating_tools=["book"])
    assert score == 0.0


def test_deviation_mutating_mismatch():
    """Mutating deviation should be penalized much more heavily."""
    trace_mut = AgentTrace(input="t", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="ok"),
        AgentStep(kind=StepKind.TOOL_CALL, name="cancel", output="done"),  # wrong mutating
    ])
    trace_non = AgentTrace(input="t", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="ok"),
        AgentStep(kind=StepKind.TOOL_CALL, name="log", output="logged"),  # wrong non-mutating
    ])
    expected = ["search", "book"]
    mutating = ["book", "cancel"]

    score_mut = decisive_deviation_score(trace_mut, expected, mutating)
    score_non = decisive_deviation_score(trace_non, expected, mutating)

    # Mutating deviation should score worse
    assert score_mut > score_non


def test_deviation_empty():
    trace = AgentTrace(input="t", steps=[])
    assert decisive_deviation_score(trace, [], []) == 0.0


def test_deviation_all_wrong():
    trace = AgentTrace(input="t", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="delete", output="gone"),
    ])
    score = decisive_deviation_score(trace, ["search"],
                                      mutating_tools=["delete"])
    assert score > 0.5  # should be high penalty


# ═══════════════════════════════════════════════════════════════
# Scenario-level expectations
# ═══════════════════════════════════════════════════════════════

def test_expect_no_side_effects_pass():
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="ok"),
        AgentStep(kind=StepKind.TOOL_CALL, name="book", output="done"),
    ]))

    s = Scenario("Clean", input="q")
    s.expect_no_side_effects(["search", "book"])
    suite = TestSuite("se")
    suite.add(s)
    assert suite.run(mock).passed


def test_expect_no_side_effects_fail():
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="ok"),
        AgentStep(kind=StepKind.TOOL_CALL, name="delete_user", output="oops"),
    ]))

    s = Scenario("Dirty", input="q")
    s.expect_no_side_effects(["search"])
    suite = TestSuite("se")
    suite.add(s)
    result = suite.run(mock)
    assert not result.passed
    assert "delete_user" in result.results[0].failed_expectations[0].detail


def test_expect_no_repetition_pass():
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="a", output="1"),
        AgentStep(kind=StepKind.TOOL_CALL, name="b", output="2"),
    ]))

    s = Scenario("NoLoop", input="q")
    s.expect_no_repetition()
    suite = TestSuite("rep")
    suite.add(s)
    assert suite.run(mock).passed


def test_expect_no_repetition_fail():
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="nothing"),
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="nothing"),
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="nothing"),
    ]))

    s = Scenario("Loop", input="q")
    s.expect_no_repetition()
    suite = TestSuite("rep")
    suite.add(s)
    assert not suite.run(mock).passed


def test_expect_max_tokens_pass():
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(input="q", steps=[], metadata={
        "token_usage": {"input": 500, "output": 200},
    }))

    s = Scenario("Budget", input="q")
    s.expect_max_tokens(1000)
    suite = TestSuite("tok")
    suite.add(s)
    assert suite.run(mock).passed


def test_expect_max_tokens_fail():
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(input="q", steps=[], metadata={
        "token_usage": {"input": 5000, "output": 8000},
    }))

    s = Scenario("Over", input="q")
    s.expect_max_tokens(10000)
    suite = TestSuite("tok")
    suite.add(s)
    result = suite.run(mock)
    assert not result.passed
    assert "13000" in result.results[0].failed_expectations[0].detail
