"""
AgentGate Scenario — Scenario definition and suite runner.

This module contains:
- Scenario: top-level E2E test definition
- ScenarioSuite: collection runner with statistical multi-run support
- AgentAdapter / CallableAgentAdapter: framework bridge

Types (AgentTrace, StepKind, etc.) live in agentgate.types.
Expectations (ExpectToolCall, etc.) live in agentgate.expectations.
Both are re-exported here for full backward compatibility.
"""

from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# Re-export types for backward compatibility
from agentgate.types import (
    StepKind,
    AgentStep,
    AgentTrace,
    ExpectationResult,
    SingleRunResult,
    ScenarioResult,
    SuiteResult,
)

# Re-export expectations for backward compatibility
from agentgate.expectations import (
    Expectation,
    ExpectToolCall,
    ExpectNoToolCall,
    ExpectToolOrder,
    ExpectState,
    ExpectNoError,
    ExpectOnToolFailure,
    ExpectOutput,
    ExpectMaxSteps,
    ExpectMaxDuration,
    ExpectNoSideEffects,
    ExpectNoRepetition,
    ExpectMaxTokens,
    ExpectMilestone,
    ExpectLLMJudge,
    _build_full_step_trace,
)


# ---------------------------------------------------------------------------
# Agent adapter
# ---------------------------------------------------------------------------

class AgentAdapter:
    """Base class for framework-specific agent adapters.

    Subclass to hook into LangGraph, CrewAI, AutoGen, OpenAI, etc.
    """

    def run(self, input_text: str) -> AgentTrace:
        raise NotImplementedError("Subclass must implement run()")


class CallableAgentAdapter(AgentAdapter):
    """Adapter for a simple callable agent function."""

    def __init__(self, agent_fn: Callable, *, trace_extractor: Optional[Callable] = None):
        self.agent_fn = agent_fn
        self.trace_extractor = trace_extractor

    def run(self, input_text: str) -> AgentTrace:
        start = time.time()
        result = self.agent_fn(input_text)
        elapsed = (time.time() - start) * 1000

        if self.trace_extractor:
            trace = self.trace_extractor(result)
            trace.input = input_text
            trace.total_duration_ms = elapsed
            return trace

        if isinstance(result, AgentTrace):
            result.input = input_text
            result.total_duration_ms = elapsed
            return result

        return AgentTrace(input=input_text, output=result, total_duration_ms=elapsed)


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------

class Scenario:
    """An end-to-end behavioral test for an AI agent.

    A Scenario defines:
    - An input to send to the agent
    - A set of expectations about the agent's behavior
    - Optional resource limits (timeout, max steps)
    """

    def __init__(self, name: str, *, input: Optional[str] = None,
                 timeout_seconds: float = 60.0,
                 max_steps: Optional[int] = None,
                 max_llm_calls: Optional[int] = None):
        self.name = name
        self.input = input or name
        self.timeout_seconds = timeout_seconds
        self.max_steps = max_steps
        self.max_llm_calls = max_llm_calls
        self.expectations: list[Expectation] = []

    # --- Fluent expectation builders ---

    def expect_tool_call(self, tool_name: str, **kwargs) -> "Scenario":
        self.expectations.append(ExpectToolCall(tool_name, **kwargs))
        return self

    def expect_no_tool_call(self, tool_name: str) -> "Scenario":
        self.expectations.append(ExpectNoToolCall(tool_name))
        return self

    def expect_tool_order(self, tool_sequence: list[str]) -> "Scenario":
        self.expectations.append(ExpectToolOrder(tool_sequence))
        return self

    def expect_state(self, state_key: str, **kwargs) -> "Scenario":
        self.expectations.append(ExpectState(state_key, **kwargs))
        return self

    def expect_no_error(self) -> "Scenario":
        self.expectations.append(ExpectNoError())
        return self

    def on_tool_failure(self, tool_name: str, *, expect: str) -> "Scenario":
        self.expectations.append(ExpectOnToolFailure(tool_name, expect=expect))
        return self

    def expect_output(self, **kwargs) -> "Scenario":
        self.expectations.append(ExpectOutput(**kwargs))
        return self

    def expect_max_steps(self, max_steps: int) -> "Scenario":
        self.expectations.append(ExpectMaxSteps(max_steps))
        return self

    def expect_max_duration(self, max_ms: float) -> "Scenario":
        self.expectations.append(ExpectMaxDuration(max_ms))
        return self

    def expect_no_side_effects(self, allowed_tools: list[str],
                                mutating_tools: Optional[list[str]] = None) -> "Scenario":
        self.expectations.append(ExpectNoSideEffects(allowed_tools, mutating_tools))
        return self

    def expect_no_repetition(self, max_rate: float = 0.0) -> "Scenario":
        self.expectations.append(ExpectNoRepetition(max_rate))
        return self

    def expect_max_tokens(self, max_tokens: int) -> "Scenario":
        self.expectations.append(ExpectMaxTokens(max_tokens))
        return self

    def expect_milestone(self, name: str, **kwargs) -> "Scenario":
        self.expectations.append(ExpectMilestone(name, **kwargs))
        return self

    def expect_llm_judge(self, criteria: str, **kwargs) -> "Scenario":
        self.expectations.append(ExpectLLMJudge(criteria, **kwargs))
        return self

    def expect_policy(self, policy_name: str, *,
                      forbidden_tools: list[str] | None = None,
                      forbidden_outputs: list[str] | None = None,
                      required_tools: list[str] | None = None) -> "Scenario":
        if forbidden_tools:
            for t in forbidden_tools:
                self.expectations.append(ExpectNoToolCall(t))
        if forbidden_outputs:
            for pattern in forbidden_outputs:
                self.expectations.append(ExpectOutput(not_contains=pattern))
        if required_tools:
            for t in required_tools:
                self.expectations.append(ExpectToolCall(t))
        return self

    def check(self, trace: AgentTrace) -> ScenarioResult:
        """Run all expectations against a trace."""
        if not self.expectations and self.max_steps is None and self.max_llm_calls is None:
            import warnings
            warnings.warn(f"Scenario '{self.name}' has no expectations.", UserWarning, stacklevel=2)

        results: list[ExpectationResult] = [exp.check(trace) for exp in self.expectations]

        if self.max_steps is not None and len(trace.steps) > self.max_steps:
            results.append(ExpectationResult(
                False, f"scenario.max_steps={self.max_steps}",
                f"Agent took {len(trace.steps)} steps, limit was {self.max_steps}",
                trace_context=_build_full_step_trace(trace),
            ))

        if self.max_llm_calls is not None:
            llm_count = sum(1 for s in trace.steps if s.kind == StepKind.LLM_CALL)
            if llm_count > self.max_llm_calls:
                results.append(ExpectationResult(
                    False, f"scenario.max_llm_calls={self.max_llm_calls}",
                    f"Agent made {llm_count} LLM calls, limit was {self.max_llm_calls}",
                    trace_context=_build_full_step_trace(trace),
                ))

        passed = all(r.passed for r in results)
        return ScenarioResult(
            scenario_name=self.name, input=self.input, passed=passed,
            expectations=results, trace=trace, duration_ms=trace.total_duration_ms,
        )


# ---------------------------------------------------------------------------
# Timeout-protected execution
# ---------------------------------------------------------------------------

def _run_scenario_with_timeout(
    agent: AgentAdapter, scenario: Scenario, timeout_seconds: float,
) -> ScenarioResult:
    if timeout_seconds <= 0:
        trace = agent.run(scenario.input)
        return scenario.check(trace)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(agent.run, scenario.input)
        try:
            trace = future.result(timeout=timeout_seconds)
            return scenario.check(trace)
        except FuturesTimeoutError:
            future.cancel()
            empty_trace = AgentTrace(input=scenario.input)
            detail = f"Scenario '{scenario.name}' timed out after {timeout_seconds:.1f}s"
            return ScenarioResult(
                scenario_name=scenario.name, input=scenario.input, passed=False,
                expectations=[ExpectationResult(False, "timeout", detail,
                    trace_context=[f"⏰ Exceeded {timeout_seconds:.1f}s"])],
                trace=empty_trace, timed_out=True, timeout_detail=detail,
            )
        except Exception as exc:
            empty_trace = AgentTrace(input=scenario.input)
            return ScenarioResult(
                scenario_name=scenario.name, input=scenario.input, passed=False,
                expectations=[ExpectationResult(False, "runtime_error",
                    f"Agent raised {type(exc).__name__}: {exc}",
                    trace_context=traceback.format_exc().splitlines())],
                trace=empty_trace,
            )


# ---------------------------------------------------------------------------
# ScenarioSuite
# ---------------------------------------------------------------------------

class ScenarioSuite:
    """Collection of E2E scenarios with statistical multi-run support."""

    def __init__(self, name: str):
        self.name = name
        self.scenarios: list[Scenario] = []

    def add(self, scenario: Scenario) -> "ScenarioSuite":
        self.scenarios.append(scenario)
        return self

    def run(self, agent: AgentAdapter | Callable, *,
            runs: int = 1,
            min_pass_rate: float = 1.0,
            timeout_seconds: float = 300.0,
            scenario_timeout: float = 60.0,
            pass_k: Optional[int] = None,
            pass_threshold: Optional[float] = None,
            ) -> SuiteResult:

        if pass_k is not None and runs == 1:
            runs = pass_k
        if pass_threshold is not None and min_pass_rate == 1.0:
            min_pass_rate = pass_threshold

        if callable(agent) and not isinstance(agent, AgentAdapter):
            agent = CallableAgentAdapter(agent)

        suite_start = time.time()
        results: list[ScenarioResult] = []

        for scenario in self.scenarios:
            elapsed_s = time.time() - suite_start
            if elapsed_s >= timeout_seconds:
                empty_trace = AgentTrace(input=scenario.input)
                results.append(ScenarioResult(
                    scenario_name=scenario.name, input=scenario.input, passed=False,
                    expectations=[ExpectationResult(False, "suite_timeout",
                        f"Suite timeout ({timeout_seconds:.1f}s) exceeded",
                        trace_context=[f"⏰ Suite elapsed: {elapsed_s:.1f}s"])],
                    trace=empty_trace, timed_out=True,
                    timeout_detail=f"Suite timeout exceeded",
                ))
                continue

            effective_timeout = min(
                scenario.timeout_seconds, scenario_timeout,
                timeout_seconds - elapsed_s,
            )

            if runs <= 1:
                result = _run_scenario_with_timeout(agent, scenario, effective_timeout)
                result.pass_rate = 1.0 if result.passed else 0.0
                result.min_pass_rate = min_pass_rate
                results.append(result)
            else:
                run_results: list[SingleRunResult] = []
                scenario_start = time.time()

                for run_idx in range(runs):
                    run_elapsed = time.time() - scenario_start
                    remaining = effective_timeout - run_elapsed
                    if remaining <= 0:
                        run_results.append(SingleRunResult(
                            run_index=run_idx, passed=False,
                            expectations=[ExpectationResult(False, "timeout",
                                f"Run {run_idx + 1} skipped — timeout")],
                            trace=None, error=f"Timeout after {effective_timeout:.1f}s",
                        ))
                        continue

                    per_run_timeout = min(remaining, effective_timeout / max(runs - run_idx, 1))
                    run_start = time.time()
                    single_result = _run_scenario_with_timeout(agent, scenario, per_run_timeout)
                    run_duration = (time.time() - run_start) * 1000

                    run_results.append(SingleRunResult(
                        run_index=run_idx, passed=single_result.passed,
                        expectations=single_result.expectations,
                        trace=single_result.trace, duration_ms=run_duration,
                        error=single_result.timeout_detail if single_result.timed_out else None,
                    ))

                pass_count = sum(1 for rr in run_results if rr.passed)
                actual_pass_rate = pass_count / len(run_results) if run_results else 0.0
                overall_passed = actual_pass_rate >= min_pass_rate

                last_run = run_results[-1]
                total_duration = (time.time() - scenario_start) * 1000

                result = ScenarioResult(
                    scenario_name=scenario.name, input=scenario.input,
                    passed=overall_passed,
                    expectations=last_run.expectations,
                    trace=last_run.trace or AgentTrace(input=scenario.input),
                    duration_ms=total_duration, run_results=run_results,
                    pass_rate=actual_pass_rate, min_pass_rate=min_pass_rate,
                )
                results.append(result)

        elapsed_ms = (time.time() - suite_start) * 1000
        return SuiteResult(suite_name=self.name, results=results, duration_ms=elapsed_ms)

    def to_jsonl(self, path: str | Path) -> None:
        with open(path, "w") as f:
            for s in self.scenarios:
                record = {"name": s.name, "input": s.input,
                          "expectations": [type(e).__name__ for e in s.expectations]}
                f.write(json.dumps(record) + "\n")

    @classmethod
    def from_jsonl(cls, name: str, path: str | Path) -> "ScenarioSuite":
        suite = cls(name)
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                suite.add(Scenario(data["name"], input=data.get("input")))
        return suite
