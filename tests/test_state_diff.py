"""Test state-diff evaluation (Agent-Diff, 2026)."""
from agentgate import (
    Scenario, TestSuite, MockAgent, AgentTrace, AgentStep, StepKind,
    StateDiff, expect_state_diff,
)


def test_expected_changes_pass():
    """State matches expected changes."""
    mock = MockAgent()
    mock.add_trace("book", AgentTrace(
        input="book",
        steps=[AgentStep(kind=StepKind.TOOL_CALL, name="book", output="ok")],
        metadata={"state": {"booking": {"status": "confirmed", "id": "BK123"}}},
    ))

    s = Scenario("Book", input="book")
    diff = StateDiff(expected_changes={"booking": {"status": "confirmed"}})
    s.expectations.append(expect_state_diff(diff))

    suite = TestSuite("sd")
    suite.add(s)
    assert suite.run(mock).passed


def test_expected_changes_fail():
    """State doesn't match expected."""
    mock = MockAgent()
    mock.add_trace("book", AgentTrace(
        input="book",
        steps=[],
        metadata={"state": {"booking": {"status": "pending"}}},
    ))

    s = Scenario("Book", input="book")
    diff = StateDiff(expected_changes={"booking": {"status": "confirmed"}})
    s.expectations.append(expect_state_diff(diff))

    suite = TestSuite("sd")
    suite.add(s)
    result = suite.run(mock)
    assert not result.passed
    assert "confirmed" in result.results[0].failed_expectations[0].detail


def test_expected_key_missing():
    """Expected key not in state."""
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(input="q", metadata={"state": {}}))

    s = Scenario("Missing", input="q")
    diff = StateDiff(expected_changes={"order": "placed"})
    s.expectations.append(expect_state_diff(diff))

    suite = TestSuite("sd")
    suite.add(s)
    assert not suite.run(mock).passed


def test_forbidden_changes_pass():
    """No forbidden changes occurred."""
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(
        input="q",
        metadata={
            "pre_state": {"user_balance": 100},
            "state": {"user_balance": 100, "booking": "confirmed"},
        },
    ))

    s = Scenario("Safe", input="q")
    diff = StateDiff(
        expected_changes={"booking": "confirmed"},
        forbidden_changes=["user_balance"],
    )
    s.expectations.append(expect_state_diff(diff))

    suite = TestSuite("sd")
    suite.add(s)
    assert suite.run(mock).passed


def test_forbidden_changes_fail():
    """Forbidden state was modified — side effect detected."""
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(
        input="q",
        metadata={
            "pre_state": {"user_balance": 100, "booking": "none"},
            "state": {"user_balance": 50, "booking": "confirmed"},
        },
    ))

    s = Scenario("Side effect", input="q")
    diff = StateDiff(
        expected_changes={"booking": "confirmed"},
        forbidden_changes=["user_balance"],
    )
    s.expectations.append(expect_state_diff(diff))

    suite = TestSuite("sd")
    suite.add(s)
    result = suite.run(mock)
    assert not result.passed
    assert "user_balance" in result.results[0].failed_expectations[0].detail


def test_forbidden_key_created():
    """Forbidden key didn't exist before but was created."""
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(
        input="q",
        metadata={
            "pre_state": {},
            "state": {"admin_flag": True},
        },
    ))

    s = Scenario("Created", input="q")
    diff = StateDiff(forbidden_changes=["admin_flag"])
    s.expectations.append(expect_state_diff(diff))

    suite = TestSuite("sd")
    suite.add(s)
    assert not suite.run(mock).passed


def test_required_unchanged_pass():
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(
        input="q",
        metadata={"state": {"version": "1.0", "data": "new"}},
    ))

    s = Scenario("Unchanged", input="q")
    diff = StateDiff(
        expected_changes={"data": "new"},
        required_unchanged={"version": "1.0"},
    )
    s.expectations.append(expect_state_diff(diff))

    suite = TestSuite("sd")
    suite.add(s)
    assert suite.run(mock).passed


def test_required_unchanged_fail():
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(
        input="q",
        metadata={"state": {"version": "2.0"}},
    ))

    s = Scenario("Changed", input="q")
    diff = StateDiff(required_unchanged={"version": "1.0"})
    s.expectations.append(expect_state_diff(diff))

    suite = TestSuite("sd")
    suite.add(s)
    assert not suite.run(mock).passed


def test_state_from_state_change_steps():
    """State extracted from state_change steps when no metadata."""
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(
        input="q",
        steps=[
            AgentStep(kind=StepKind.STATE_CHANGE, name="booking_status", output="confirmed"),
            AgentStep(kind=StepKind.STATE_CHANGE, name="balance", output=50),
        ],
    ))

    s = Scenario("Steps", input="q")
    diff = StateDiff(expected_changes={"booking_status": "confirmed"})
    s.expectations.append(expect_state_diff(diff))

    suite = TestSuite("sd")
    suite.add(s)
    assert suite.run(mock).passed


def test_nested_dict_matching():
    """Nested dict partial matching."""
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(
        input="q",
        metadata={"state": {
            "order": {"status": "shipped", "tracking": "TRK123", "items": 3}
        }},
    ))

    s = Scenario("Nested", input="q")
    diff = StateDiff(expected_changes={"order": {"status": "shipped"}})
    s.expectations.append(expect_state_diff(diff))

    suite = TestSuite("sd")
    suite.add(s)
    assert suite.run(mock).passed  # partial match: only checks status
