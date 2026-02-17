"""Test Agent-as-a-Judge (ICLR 2026: LLM-based graders)."""
from agentgate import (
    Scenario, TestSuite, MockAgent, AgentTrace, AgentStep, StepKind,
    ExpectLLMJudge,
)


def _make_trace(tools, output=None):
    return AgentTrace(
        input="test query",
        steps=[AgentStep(kind=StepKind.TOOL_CALL, name=t, output="ok") for t in tools],
        output=output,
    )


# --- Judge function variants ---

def always_pass(criteria, trace_text):
    """Simple judge: always passes."""
    return True


def always_fail(criteria, trace_text):
    """Simple judge: always fails."""
    return False


def fail_with_reason(criteria, trace_text):
    """Judge that returns (bool, reason) tuple."""
    return (False, "Agent was rude to the user")


def fail_with_dict(criteria, trace_text):
    """Judge that returns a dict."""
    return {"passed": False, "reasoning": "Violated safety guidelines"}


def pass_with_dict(criteria, trace_text):
    """Judge that returns a passing dict."""
    return {"passed": True, "reasoning": "All criteria met"}


def keyword_judge(criteria, trace_text):
    """Judge that checks for specific tools in trace."""
    return "search" in trace_text.lower()


def error_judge(criteria, trace_text):
    """Judge that raises an exception."""
    raise ValueError("LLM API unavailable")


# --- Tests ---

def test_llm_judge_pass():
    """LLM judge that passes."""
    mock = MockAgent()
    mock.add_trace("q", _make_trace(["search"]))

    s = Scenario("Judge test", input="q")
    s.expect_llm_judge("Be helpful", judge_fn=always_pass)

    suite = TestSuite("judge")
    suite.add(s)
    result = suite.run(mock)
    assert result.passed


def test_llm_judge_fail():
    """LLM judge that fails."""
    mock = MockAgent()
    mock.add_trace("q", _make_trace(["search"]))

    s = Scenario("Judge test", input="q")
    s.expect_llm_judge("Be polite", judge_fn=always_fail)

    suite = TestSuite("judge")
    suite.add(s)
    result = suite.run(mock)
    assert not result.passed


def test_llm_judge_tuple_return():
    """Judge returning (bool, reason) tuple."""
    mock = MockAgent()
    mock.add_trace("q", _make_trace(["search"]))

    s = Scenario("Politeness", input="q")
    s.expect_llm_judge("Be polite", judge_fn=fail_with_reason)

    suite = TestSuite("judge")
    suite.add(s)
    result = suite.run(mock)
    assert not result.passed
    failed = result.results[0].failed_expectations
    assert len(failed) == 1
    assert "rude" in failed[0].detail.lower()


def test_llm_judge_dict_return():
    """Judge returning a dict with passed + reasoning."""
    mock = MockAgent()
    mock.add_trace("q", _make_trace(["search"]))

    s = Scenario("Safety", input="q")
    s.expect_llm_judge("Follow safety", judge_fn=fail_with_dict)

    suite = TestSuite("judge")
    suite.add(s)
    result = suite.run(mock)
    assert not result.passed
    assert "safety" in result.results[0].failed_expectations[0].detail.lower()


def test_llm_judge_dict_pass():
    """Judge returning a passing dict."""
    mock = MockAgent()
    mock.add_trace("q", _make_trace(["search"]))

    s = Scenario("Check", input="q")
    s.expect_llm_judge("All good", judge_fn=pass_with_dict)

    suite = TestSuite("judge")
    suite.add(s)
    result = suite.run(mock)
    assert result.passed


def test_llm_judge_with_trace_content():
    """Judge that actually inspects trace content."""
    mock = MockAgent()
    mock.add_trace("q", _make_trace(["search_flights", "book_flight"]))

    s = Scenario("Search check", input="q")
    s.expect_llm_judge("Agent must search before booking", judge_fn=keyword_judge)

    suite = TestSuite("judge")
    suite.add(s)
    result = suite.run(mock)
    assert result.passed  # "search" is in the trace


def test_llm_judge_error_handling():
    """Judge that raises an exception → fails gracefully."""
    mock = MockAgent()
    mock.add_trace("q", _make_trace(["search"]))

    s = Scenario("Error test", input="q")
    s.expect_llm_judge("Check", judge_fn=error_judge)

    suite = TestSuite("judge")
    suite.add(s)
    result = suite.run(mock)
    assert not result.passed
    failed = result.results[0].failed_expectations
    assert "API unavailable" in failed[0].detail


def test_llm_judge_no_fn():
    """No judge_fn provided → fails with helpful message."""
    mock = MockAgent()
    mock.add_trace("q", _make_trace(["search"]))

    s = Scenario("No fn", input="q")
    s.expect_llm_judge("Check something")  # no judge_fn

    suite = TestSuite("judge")
    suite.add(s)
    result = suite.run(mock)
    assert not result.passed
    assert "No judge_fn" in result.results[0].failed_expectations[0].detail


def test_llm_judge_combined_with_deterministic():
    """LLM judge + deterministic expectations together."""
    mock = MockAgent()
    mock.add_trace("q", _make_trace(["search", "book"], output="Flight booked"))

    s = Scenario("Combined", input="q")
    s.expect_tool_call("search")
    s.expect_tool_call("book")
    s.expect_llm_judge("Response is professional", judge_fn=always_pass)

    suite = TestSuite("combined")
    suite.add(s)
    result = suite.run(mock)
    assert result.passed


def test_trace_to_text_format():
    """Verify trace_to_text produces readable output."""
    trace = AgentTrace(
        input="book a flight",
        steps=[
            AgentStep(kind=StepKind.TOOL_CALL, name="search",
                      input={"dest": "Tokyo"}, output="3 flights"),
            AgentStep(kind=StepKind.LLM_CALL, name="agent", output="I found 3 flights"),
            AgentStep(kind=StepKind.TOOL_CALL, name="book",
                      input={"flight": "UA1"}, output="confirmed"),
        ],
        output="Your flight UA1 is booked!",
    )

    text = ExpectLLMJudge._trace_to_text(trace)
    assert "book a flight" in text
    assert "search" in text
    assert "dest=Tokyo" in text
    assert "3 flights" in text
    assert "book" in text
    assert "flight=UA1" in text
    assert "confirmed" in text
    assert "Your flight UA1 is booked!" in text


def test_multiple_judges():
    """Multiple LLM judges on the same scenario."""
    mock = MockAgent()
    mock.add_trace("q", _make_trace(["search"]))

    s = Scenario("Multi judge", input="q")
    s.expect_llm_judge("Is polite", judge_fn=always_pass)
    s.expect_llm_judge("Is safe", judge_fn=always_pass)
    s.expect_llm_judge("Is efficient", judge_fn=always_fail)

    suite = TestSuite("multi")
    suite.add(s)
    result = suite.run(mock)
    assert not result.passed  # one judge failed
    assert len(result.results[0].failed_expectations) == 1
