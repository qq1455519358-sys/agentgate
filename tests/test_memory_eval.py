"""Test memory evaluation (MemoryAgentBench, ICLR 2026)."""
from agentgate import (
    Scenario, MockAgent, AgentTrace, AgentStep, StepKind,
    MemoryProbe,
    expect_accurate_retrieval,
    expect_conflict_resolution,
    expect_memory_consistency,
    memory_consistency_suite,
)


def test_accurate_retrieval_pass():
    """Agent recalls stored fact."""
    mock = MockAgent()
    mock.add_trace("prefs", AgentTrace(input="prefs", steps=[
        AgentStep(kind=StepKind.LLM_CALL, name="recall",
                  output="User prefers dark mode and vim keybindings"),
    ]))

    exp = expect_accurate_retrieval(
        stored_fact="user prefers dark mode",
        query="what are my preferences?",
        expected_keywords=["dark mode"],
    )
    trace = mock.run("prefs")
    result = exp.check(trace)
    assert result.passed


def test_accurate_retrieval_fail():
    """Agent can't recall stored fact."""
    mock = MockAgent()
    mock.add_trace("prefs", AgentTrace(input="prefs", steps=[
        AgentStep(kind=StepKind.LLM_CALL, name="recall",
                  output="I don't have any preference information"),
    ]))

    exp = expect_accurate_retrieval(
        stored_fact="user prefers dark mode",
        query="what are my preferences?",
        expected_keywords=["dark mode"],
    )
    result = exp.check(mock.run("prefs"))
    assert not result.passed
    assert "dark mode" in result.detail


def test_accurate_retrieval_in_metadata():
    """Retrieval found in metadata response."""
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(
        input="q", steps=[],
        metadata={"memory_response": "Your preference is dark mode"},
    ))

    exp = expect_accurate_retrieval(
        stored_fact="dark mode",
        query="q",
        expected_keywords=["dark mode"],
    )
    assert exp.check(mock.run("q")).passed


def test_conflict_resolution_recency():
    """Agent uses the most recent conflicting fact."""
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.LLM_CALL, name="resolve",
                  output="The meeting was previously at 2pm but updated to 3pm"),
    ]))

    exp = expect_conflict_resolution(
        old_fact="meeting at 2pm",
        new_fact="3pm",
        expected_resolution="recency",
    )
    assert exp.check(mock.run("q")).passed


def test_conflict_resolution_acknowledge():
    """Agent acknowledges the conflict."""
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.LLM_CALL, name="think",
                  output="There's a contradiction in the information"),
    ]))

    exp = expect_conflict_resolution(
        old_fact="budget is $1000",
        new_fact="budget is $500",
        expected_resolution="acknowledge",
    )
    assert exp.check(mock.run("q")).passed


def test_conflict_resolution_fail():
    """Agent ignores conflict entirely."""
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.LLM_CALL, name="think",
                  output="The budget looks fine, proceeding with the task"),
    ]))

    exp = expect_conflict_resolution(
        old_fact="budget is $1000",
        new_fact="$500",
        expected_resolution="recency",
    )
    result = exp.check(mock.run("q"))
    assert not result.passed


def test_memory_consistency_pass():
    """Agent gives consistent answers about same topic."""
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.LLM_CALL, name="r1",
                  output="The project deadline is March 15"),
        AgentStep(kind=StepKind.TOOL_CALL, name="check", output="ok"),
        AgentStep(kind=StepKind.LLM_CALL, name="r2",
                  output="As mentioned, the project deadline is in March"),
    ]))

    exp = expect_memory_consistency("project deadline")
    assert exp.check(mock.run("q")).passed


def test_memory_consistency_fail():
    """Agent contradicts itself."""
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.LLM_CALL, name="r1",
                  output="The task is correct and the deadline is set"),
        AgentStep(kind=StepKind.LLM_CALL, name="r2",
                  output="Actually the task is incorrect and needs revision"),
    ]))

    exp = expect_memory_consistency("task")
    result = exp.check(mock.run("q"))
    assert not result.passed
    assert "contradiction" in result.detail


def test_memory_consistency_insufficient_data():
    """Not enough data points — passes with note."""
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.LLM_CALL, name="r1",
                  output="The task is complete"),
    ]))

    exp = expect_memory_consistency("deadline")
    result = exp.check(mock.run("q"))
    assert result.passed  # Not enough data to detect contradiction
    assert "insufficient" in result.detail


def test_memory_probe_suite():
    """Memory probes generate correct scenario suite."""
    probes = [
        MemoryProbe(
            fact="User's birthday is March 5",
            queries=["When is my birthday?", "What month was I born?"],
            expected_answers=["march 5", "march"],
            competency="AR",
        ),
        MemoryProbe(
            fact="Project uses Python 3.12",
            queries=["What Python version?"],
            expected_answers=["3.12"],
            competency="TTL",
        ),
    ]
    scenarios = memory_consistency_suite(probes)
    assert len(scenarios) == 3  # 2 from first probe + 1 from second
    assert scenarios[0].name == "Memory AR #1.1"
    assert scenarios[2].name == "Memory TTL #2.1"
    assert scenarios[0]._memory_meta["competency"] == "AR"
    assert scenarios[2]._memory_meta["injected_fact"] == "Project uses Python 3.12"


def test_memory_probe_with_turns_between():
    """Probe preserves turns_between metadata."""
    probe = MemoryProbe(
        fact="API key is abc123",
        queries=["What's the API key?"],
        expected_answers=["abc123"],
        turns_between=10,
        competency="LRU",
    )
    scenarios = memory_consistency_suite([probe])
    assert scenarios[0]._memory_meta["turns_between"] == 10
    assert scenarios[0]._memory_meta["competency"] == "LRU"
