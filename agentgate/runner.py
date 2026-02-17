"""
AgentGate Runner — TestSuite statistical running engine.

TestSuite is the primary public API name for running scenarios.
It wraps ScenarioSuite from scenario.py with the user-facing name.

Usage:
    from agentgate import TestSuite, Scenario

    suite = TestSuite("my-agent-tests")
    suite.add(Scenario("Book a flight", input="Book me a flight to Tokyo"))

    result = suite.run(agent, runs=5, min_pass_rate=0.8, timeout_seconds=30)
    print(result.summary())
    assert result.passed
"""

from __future__ import annotations

from agentgate.scenario import ScenarioSuite, SuiteResult, ScenarioResult, SingleRunResult


class TestSuite(ScenarioSuite):
    """
    A collection of E2E scenarios to run against an agent.

    This is the primary user-facing name. Wraps ScenarioSuite with
    identical functionality.

    Supports:
    - Statistical multi-run execution (runs=N, min_pass_rate)
    - Per-scenario and suite-level timeouts
    - Rich failure diagnostics with full tool-call traces

    Example::

        suite = TestSuite("booking-agent")
        suite.add(scenario_book)
        suite.add(scenario_cancel)

        # Single run
        result = suite.run(agent)

        # Statistical: 10 runs, require 80% pass rate
        result = suite.run(agent, runs=10, min_pass_rate=0.8)

        print(result.summary())
        assert result.passed
    """
    pass


__all__ = ["TestSuite", "SuiteResult", "ScenarioResult", "SingleRunResult"]
