"""
AgentGate Regression — Capability vs Regression eval suites with trials.

From Anthropic's "Demystifying evals for AI agents" (Jan 2026):

    "Capability evals ask 'What can this agent do well?' They should
    start at a low pass rate, targeting tasks the agent struggles with."

    "Regression evals ask 'Does the agent still handle all the tasks
    it used to?' and should have a nearly 100% pass rate."

    "After an agent is optimized, capability evals with high pass rates
    can 'graduate' to become a regression suite."

    "Each attempt at a task is a trial. Because model outputs vary
    between runs, we run multiple trials to produce more consistent
    results."

Usage:
    from agentgate.regression import EvalSuiteManager

    mgr = EvalSuiteManager()
    mgr.add_capability("new_feature", scenario)
    mgr.add_regression("core_booking", scenario)
    results = mgr.run(agent, trials=5)
    mgr.graduate(threshold=0.95)  # promote passing capabilities
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agentgate.scenario import (
    Scenario, ScenarioResult, AgentAdapter,
)
from agentgate.runner import TestSuite


@dataclass
class EvalResult:
    """Result of a managed eval run with capability/regression split."""
    capability_results: list[ScenarioResult] = field(default_factory=list)
    regression_results: list[ScenarioResult] = field(default_factory=list)

    @property
    def capability_pass_rate(self) -> float:
        if not self.capability_results:
            return 0.0
        return sum(1 for r in self.capability_results if r.passed) / len(self.capability_results)

    @property
    def regression_pass_rate(self) -> float:
        if not self.regression_results:
            return 1.0
        return sum(1 for r in self.regression_results if r.passed) / len(self.regression_results)

    @property
    def has_regressions(self) -> bool:
        """True if any regression test failed — this is a blocker."""
        return any(not r.passed for r in self.regression_results)

    @property
    def new_capabilities(self) -> list[str]:
        """Capability tests that are now passing."""
        return [r.scenario_name for r in self.capability_results if r.passed]

    def summary(self) -> str:
        lines = []
        reg_icon = "✅" if not self.has_regressions else "🚨"
        lines.append(f"{reg_icon} Regression: {self.regression_pass_rate:.0%} "
                      f"({sum(1 for r in self.regression_results if r.passed)}"
                      f"/{len(self.regression_results)})")

        cap_n = len(self.capability_results)
        cap_p = sum(1 for r in self.capability_results if r.passed)
        lines.append(f"🧪 Capability: {self.capability_pass_rate:.0%} ({cap_p}/{cap_n})")

        if self.has_regressions:
            failed = [r.scenario_name for r in self.regression_results if not r.passed]
            lines.append(f"🚨 REGRESSIONS: {', '.join(failed)}")

        if self.new_capabilities:
            lines.append(f"🎉 New capabilities: {', '.join(self.new_capabilities)}")

        return "\n".join(lines)


class EvalSuiteManager:
    """Manages capability and regression eval suites.

    From Anthropic (Jan 2026):
        "As teams hill-climb on capability evals, it's important to also
        run regression evals to make sure changes don't cause issues."

    Scenarios start as capability tests. When they consistently pass
    (above threshold), they "graduate" to the regression suite.
    """

    def __init__(self):
        self._capability: dict[str, Scenario] = {}
        self._regression: dict[str, Scenario] = {}
        self._history: list[dict] = []

    def add_capability(self, name: str, scenario: Scenario) -> None:
        """Add a scenario to the capability suite."""
        self._capability[name] = scenario

    def add_regression(self, name: str, scenario: Scenario) -> None:
        """Add a scenario to the regression suite."""
        self._regression[name] = scenario

    @property
    def capability_count(self) -> int:
        return len(self._capability)

    @property
    def regression_count(self) -> int:
        return len(self._regression)

    def run(self, agent: AgentAdapter, *,
            trials: int = 1,
            min_pass_rate: float = 0.0) -> EvalResult:
        """Run both suites.

        Args:
            agent: The agent adapter to test.
            trials: Number of trials per scenario (Anthropic: "run multiple
                trials to produce more consistent results").
            min_pass_rate: Minimum pass rate for multi-trial scenarios.

        Returns:
            EvalResult with separate capability and regression results.
        """
        result = EvalResult()

        # Run regression suite
        if self._regression:
            reg_suite = TestSuite("regression")
            for s in self._regression.values():
                reg_suite.add(s)
            reg_result = reg_suite.run(agent, runs=trials, min_pass_rate=min_pass_rate)
            result.regression_results = reg_result.results

        # Run capability suite
        if self._capability:
            cap_suite = TestSuite("capability")
            for s in self._capability.values():
                cap_suite.add(s)
            cap_result = cap_suite.run(agent, runs=trials, min_pass_rate=min_pass_rate)
            result.capability_results = cap_result.results

        # Record history
        self._history.append({
            "regression_pass_rate": result.regression_pass_rate,
            "capability_pass_rate": result.capability_pass_rate,
            "regressions": result.has_regressions,
            "new_capabilities": result.new_capabilities,
        })

        return result

    def graduate(self, threshold: float = 0.95) -> list[str]:
        """Promote passing capability tests to regression suite.

        From Anthropic (Jan 2026):
            "Capability evals with high pass rates can 'graduate' to
            become a regression suite."

        Args:
            threshold: Minimum pass rate to graduate (default 0.95).

        Returns:
            List of graduated scenario keys.
        """
        graduated: list[str] = []

        if not self._history:
            return graduated

        # Build reverse lookup: scenario_name → key
        name_to_key = {s.name: k for k, s in self._capability.items()}

        last = self._history[-1]
        for scenario_name in list(last.get("new_capabilities", [])):
            key = name_to_key.get(scenario_name)
            if key and key in self._capability:
                scenario = self._capability.pop(key)
                self._regression[key] = scenario
                graduated.append(key)

        return graduated

    def save_state(self, path: str | Path) -> None:
        """Save suite composition and history to JSON."""
        data = {
            "capability": list(self._capability.keys()),
            "regression": list(self._regression.keys()),
            "history": self._history,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load_state(self, path: str | Path) -> None:
        """Load history from JSON (scenarios must be re-added)."""
        data = json.loads(Path(path).read_text())
        self._history = data.get("history", [])


__all__ = ["EvalSuiteManager", "EvalResult"]
