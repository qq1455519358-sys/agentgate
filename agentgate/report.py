"""
AgentGate Report — Terminal output formatting.

Provides rich, human-readable terminal output for scenario results,
including full tool-call traces and failure diagnostics.

Usage:
    from agentgate.report import format_result, format_suite_result

    result = suite.run(agent)
    print(format_suite_result(result))
"""

from __future__ import annotations

from agentgate.scenario import (
    ScenarioResult,
    SuiteResult,
    ExpectationResult,
    AgentTrace,
)


def format_expectation(exp: ExpectationResult, indent: str = "  ") -> str:
    """Format a single expectation result with trace context."""
    icon = "✅" if exp.passed else "❌"
    lines = [f"{indent}{icon} {exp.description}"]
    if exp.detail:
        lines[0] += f" — {exp.detail}"
    if not exp.passed and exp.trace_context:
        lines.append(f"{indent}  Trace:")
        for line in exp.trace_context:
            lines.append(f"{indent}    {line}")
    return "\n".join(lines)


def format_result(result: ScenarioResult) -> str:
    """Format a single scenario result."""
    total = len(result.expectations)
    passed = sum(1 for e in result.expectations if e.passed)
    icon = "✅" if result.passed else "❌"

    lines: list[str] = []

    if result.run_results:
        lines.append(f"{icon} Scenario: {result.scenario_name} — {result.statistical_summary}")
    else:
        lines.append(f"{icon} Scenario: {result.scenario_name} ({passed}/{total} expectations passed)")

    if result.timed_out:
        lines.append(f"  ⏰ TIMEOUT: {result.timeout_detail}")

    for e in result.expectations:
        lines.append(format_expectation(e))

    if result.run_results and not result.passed:
        lines.append("  --- Per-run breakdown ---")
        for rr in result.run_results:
            run_icon = "✅" if rr.passed else "❌"
            suffix = f" (error: {rr.error})" if rr.error else ""
            lines.append(f"  Run {rr.run_index + 1}: {run_icon}{suffix}")
            if not rr.passed:
                for e in rr.expectations:
                    if not e.passed:
                        lines.append(format_expectation(e, indent="    "))

    return "\n".join(lines)


def format_suite_result(result: SuiteResult) -> str:
    """Format a full suite result for terminal output."""
    total = len(result.results)
    passed = sum(1 for r in result.results if r.passed)
    icon = "✅" if result.passed else "❌"

    lines = [
        f"\n{'=' * 60}",
        f"{icon} Suite: {result.suite_name} — {passed}/{total} scenarios passed ({result.pass_rate:.0%})",
        f"{'=' * 60}",
    ]
    for r in result.results:
        lines.append(format_result(r))
        lines.append("")

    return "\n".join(lines)


def print_suite_result(result: SuiteResult) -> None:
    """Print suite result to stdout."""
    print(format_suite_result(result))


def diff_results(baseline: SuiteResult, current: SuiteResult) -> str:
    """Compare two suite results and show regressions/improvements.

    Useful for detecting behavioral drift across model updates, prompt changes,
    or agent code changes. Inspired by Anthropic's recommendation to run
    multiple trials and track regressions over time.

    Args:
        baseline: Previous suite result (e.g., from last release).
        current: New suite result (e.g., from current commit).

    Returns:
        Human-readable diff string showing regressions (🔴), improvements (🟢),
        and unchanged scenarios (⚪).
    """
    baseline_map: dict[str, ScenarioResult] = {r.scenario_name: r for r in baseline.results}
    current_map: dict[str, ScenarioResult] = {r.scenario_name: r for r in current.results}

    all_names = list(dict.fromkeys(
        [r.scenario_name for r in baseline.results] +
        [r.scenario_name for r in current.results]
    ))

    lines: list[str] = [
        f"\n{'=' * 60}",
        f"📊 Regression Report: {baseline.suite_name}",
        f"{'=' * 60}",
    ]

    regressions = 0
    improvements = 0

    for name in all_names:
        b = baseline_map.get(name)
        c = current_map.get(name)

        if b and c:
            b_pass = b.passed
            c_pass = c.passed
            if b_pass and not c_pass:
                lines.append(f"  🔴 REGRESSION: {name}")
                lines.append(f"      was: ✅ pass → now: ❌ fail")
                regressions += 1
            elif not b_pass and c_pass:
                lines.append(f"  🟢 FIXED: {name}")
                lines.append(f"      was: ❌ fail → now: ✅ pass")
                improvements += 1
            else:
                icon = "✅" if c_pass else "❌"
                lines.append(f"  ⚪ unchanged: {name} ({icon})")

            # Show pass_rate diff for statistical runs
            if b.pass_rate is not None and c.pass_rate is not None:
                if abs(b.pass_rate - c.pass_rate) > 0.05:
                    lines.append(f"      pass_rate: {b.pass_rate:.0%} → {c.pass_rate:.0%}")

            # Show tool call sequence diff
            b_tools = b.trace.tool_names if b.trace else []
            c_tools = c.trace.tool_names if c.trace else []
            if b_tools != c_tools:
                lines.append(f"      tools: {b_tools} → {c_tools}")

        elif c and not b:
            icon = "✅" if c.passed else "❌"
            lines.append(f"  🆕 NEW: {name} ({icon})")
        elif b and not c:
            lines.append(f"  🗑️  REMOVED: {name}")

    lines.append(f"\n  Summary: {regressions} regressions, {improvements} improvements")
    if regressions > 0:
        lines.append(f"  ⚠️  REGRESSIONS DETECTED — review before deploying")

    return "\n".join(lines)


__all__ = [
    "format_expectation",
    "format_result",
    "format_suite_result",
    "print_suite_result",
    "diff_results",
]
