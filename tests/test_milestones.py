"""Test partial credit / milestone scoring (ICLR 2026: Milestones and Subgoals)."""
from agentgate import (
    Scenario, TestSuite, MockAgent, AgentTrace, AgentStep, StepKind,
)


def test_all_milestones_reached():
    """Full credit when all milestones are reached."""
    mock = MockAgent()
    mock.add_trace("book flight", AgentTrace(
        input="book flight",
        steps=[
            AgentStep(kind=StepKind.TOOL_CALL, name="search_flights", output="3 results"),
            AgentStep(kind=StepKind.TOOL_CALL, name="select_flight", output="UA123"),
            AgentStep(kind=StepKind.TOOL_CALL, name="book_flight", output="confirmed"),
        ],
        output="Flight booked: UA123",
    ))

    s = Scenario("Book flight", input="book flight")
    s.expect_milestone("Found flights", tool="search_flights")
    s.expect_milestone("Selected flight", tool="select_flight")
    s.expect_milestone("Booking confirmed", tool="book_flight")

    suite = TestSuite("milestones")
    suite.add(s)
    result = suite.run(mock)
    assert result.results[0].score == 1.0
    assert result.average_score == 1.0


def test_partial_milestones():
    """Partial credit when only some milestones are reached."""
    mock = MockAgent()
    mock.add_trace("book flight", AgentTrace(
        input="book flight",
        steps=[
            AgentStep(kind=StepKind.TOOL_CALL, name="search_flights", output="3 results"),
            # Agent got stuck — never selected or booked
        ],
    ))

    s = Scenario("Book flight", input="book flight")
    s.expect_milestone("Found flights", tool="search_flights", weight=1.0)
    s.expect_milestone("Selected flight", tool="select_flight", weight=1.0)
    s.expect_milestone("Booking confirmed", tool="book_flight", weight=2.0)

    suite = TestSuite("milestones")
    suite.add(s)
    result = suite.run(mock)
    # 1/4 total weight achieved (search=1, total=1+1+2=4)
    assert abs(result.results[0].score - 0.25) < 0.01


def test_no_milestones_reached():
    """Zero credit when no milestones are reached."""
    mock = MockAgent()
    mock.add_trace("book flight", AgentTrace(
        input="book flight",
        steps=[
            AgentStep(kind=StepKind.TOOL_CALL, name="random_tool", output="nothing"),
        ],
    ))

    s = Scenario("Book flight", input="book flight")
    s.expect_milestone("Found flights", tool="search_flights")
    s.expect_milestone("Booking confirmed", tool="book_flight")

    suite = TestSuite("milestones")
    suite.add(s)
    result = suite.run(mock)
    assert result.results[0].score == 0.0


def test_weighted_milestones():
    """Weighted milestones give proportional credit."""
    mock = MockAgent()
    mock.add_trace("process", AgentTrace(
        input="process",
        steps=[
            AgentStep(kind=StepKind.TOOL_CALL, name="validate", output="ok"),
            AgentStep(kind=StepKind.TOOL_CALL, name="transform", output="done"),
            # Missing: "store" (weight=3)
        ],
    ))

    s = Scenario("Process data", input="process")
    s.expect_milestone("Validated", tool="validate", weight=1.0)
    s.expect_milestone("Transformed", tool="transform", weight=1.0)
    s.expect_milestone("Stored", tool="store", weight=3.0)

    suite = TestSuite("weighted")
    suite.add(s)
    result = suite.run(mock)
    # 2/5 weight achieved
    assert abs(result.results[0].score - 0.4) < 0.01


def test_milestones_with_regular_expectations():
    """Milestones + regular expectations combine scoring."""
    mock = MockAgent()
    mock.add_trace("check", AgentTrace(
        input="check",
        steps=[
            AgentStep(kind=StepKind.TOOL_CALL, name="check_booking", output="ok"),
        ],
    ))

    s = Scenario("Check booking", input="check")
    s.expect_milestone("Checked", tool="check_booking")
    s.expect_no_tool_call("cancel_booking")  # regular expectation

    suite = TestSuite("mixed")
    suite.add(s)
    result = suite.run(mock)
    # milestone: 1/1 = 1.0, regular: 1/1 = 1.0
    # combined: 0.7 * 1.0 + 0.3 * 1.0 = 1.0
    assert result.results[0].score == 1.0


def test_output_milestone():
    """Milestone based on output content."""
    mock = MockAgent()
    mock.add_trace("ask", AgentTrace(
        input="ask",
        steps=[],
        output="Your booking is confirmed. Reference: BK123",
    ))

    s = Scenario("Confirm", input="ask")
    s.expect_milestone("Got confirmation", output_contains="confirmed")
    s.expect_milestone("Got reference", output_contains="Reference:")

    suite = TestSuite("output")
    suite.add(s)
    result = suite.run(mock)
    assert result.results[0].score == 1.0


def test_suite_average_score():
    """Suite average_score averages across scenarios."""
    mock = MockAgent()
    mock.add_trace("good", AgentTrace(
        input="good",
        steps=[AgentStep(kind=StepKind.TOOL_CALL, name="a", output="ok")],
    ))
    mock.add_trace("bad", AgentTrace(
        input="bad",
        steps=[AgentStep(kind=StepKind.TOOL_CALL, name="x", output="nope")],
    ))

    s1 = Scenario("Good", input="good")
    s1.expect_milestone("Step A", tool="a")

    s2 = Scenario("Bad", input="bad")
    s2.expect_milestone("Step A", tool="a")  # not reached (tool is "x")

    suite = TestSuite("avg")
    suite.add(s1)
    suite.add(s2)
    result = suite.run(mock)
    # s1: score=1.0, s2: score=0.0, avg=0.5
    assert abs(result.average_score - 0.5) < 0.01


def test_summary_shows_score():
    """Summary should show score when milestones are used."""
    mock = MockAgent()
    mock.add_trace("x", AgentTrace(
        input="x",
        steps=[AgentStep(kind=StepKind.TOOL_CALL, name="a", output="ok")],
    ))

    s = Scenario("Test", input="x")
    s.expect_milestone("Step", tool="a")

    suite = TestSuite("summary")
    suite.add(s)
    result = suite.run(mock)
    summary = result.summary()
    assert "score:" in summary
