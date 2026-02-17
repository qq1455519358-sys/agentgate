"""
AgentGate pytest plugin — run E2E agent scenarios as pytest tests.

Usage:
    # conftest.py
    pytest_plugins = ["agentgate"]

    # test_my_agent.py
    from agentgate import Scenario, MockAgent

    def test_booking(agentgate):
        agent = MockAgent()
        agent.add_trace("book flight", ...)
        s = Scenario("Book flight", input="book a flight to Tokyo")
        s.expect_tool_call("search_flights")
        result = agentgate.run(agent, s)
        assert result.passed, result.summary()

Or simply use the `@scenario` decorator:

    from agentgate import scenario_test, Scenario, MockAgent

    @scenario_test
    def test_booking():
        agent = MockAgent()
        ...
        s = Scenario("Book flight", input="book a flight to Tokyo")
        s.expect_tool_call("search_flights")
        return agent, s  # plugin runs and asserts
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Optional

from agentgate.scenario import (
    Scenario,
    ScenarioResult,
    AgentAdapter,
    CallableAgentAdapter,
    _run_scenario_with_timeout,
)


@dataclass
class AgentGateFixture:
    """pytest fixture for running AgentGate scenarios inline."""

    timeout_seconds: float = 60.0

    def run(self, agent, scenario: Scenario, *,
            timeout_seconds: Optional[float] = None) -> ScenarioResult:
        """Run a single scenario and return the result.

        Does NOT auto-assert — you control the assertion.
        """
        if callable(agent) and not isinstance(agent, AgentAdapter):
            agent = CallableAgentAdapter(agent)

        effective_timeout = timeout_seconds or self.timeout_seconds
        return _run_scenario_with_timeout(agent, scenario, effective_timeout)

    def assert_pass(self, agent, scenario: Scenario, *,
                    timeout_seconds: Optional[float] = None) -> ScenarioResult:
        """Run a scenario and assert it passes. Returns result for inspection."""
        result = self.run(agent, scenario, timeout_seconds=timeout_seconds)
        if not result.passed:
            pytest.fail(result.summary())
        return result

    def assert_fail(self, agent, scenario: Scenario, *,
                    timeout_seconds: Optional[float] = None) -> ScenarioResult:
        """Run a scenario and assert it FAILS (useful for adversarial tests)."""
        result = self.run(agent, scenario, timeout_seconds=timeout_seconds)
        if result.passed:
            pytest.fail(
                f"Expected scenario '{scenario.name}' to FAIL but it passed.\n"
                f"{result.summary()}"
            )
        return result


@pytest.fixture
def agentgate():
    """Fixture for running AgentGate scenarios in pytest tests."""
    return AgentGateFixture()
