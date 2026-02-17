"""AgentGate CI/CD integration.

Provides:
  - run_regression()  — execute a full regression suite, return pass/fail
  - JUnit XML output  — compatible with GitHub Actions, GitLab, Jenkins
  - Console reporter  — human-readable terminal output
"""

from __future__ import annotations

import sys
import time
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path
from typing import TextIO

from .suite import TestSuite, AgentFn
from .types import SuiteResult, TestResult, Verdict


# ---------------------------------------------------------------------------
# JUnit XML generation
# ---------------------------------------------------------------------------

def _build_junit_xml(suite_result: SuiteResult) -> ET.Element:
    """Convert a SuiteResult into JUnit XML format.

    Produces a ``<testsuites>`` root with a single ``<testsuite>`` containing
    one ``<testcase>`` per test.  Failed tests get a ``<failure>`` child,
    errored tests get an ``<error>`` child.
    """
    testsuites = ET.Element("testsuites")

    ts = ET.SubElement(testsuites, "testsuite")
    ts.set("name", suite_result.suite_name)
    ts.set("tests", str(suite_result.total))
    ts.set("failures", str(suite_result.failed_count))
    ts.set("errors", str(sum(1 for r in suite_result.results if r.error)))
    ts.set("time", f"{suite_result.duration_ms / 1000:.3f}")
    ts.set("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))

    for result in suite_result.results:
        tc = ET.SubElement(ts, "testcase")
        tc.set("name", result.test_case.name or result.test_case.id)
        tc.set("classname", suite_result.suite_name)
        tc.set("time", f"{result.duration_ms / 1000:.3f}")

        if result.error:
            err = ET.SubElement(tc, "error")
            err.set("message", result.error)
            err.text = result.error
        elif result.verdict == Verdict.FAIL:
            details = []
            for er in result.eval_results:
                details.append(
                    f"[{er.kind.value}] score={er.score:.4f} — {er.detail}"
                )
            msg = "; ".join(details)
            fail = ET.SubElement(tc, "failure")
            fail.set("message", msg)
            fail.text = (
                f"Input: {result.test_case.input}\n"
                f"Expected: {result.test_case.expected}\n"
                f"Actual: {result.actual_output}\n"
                f"Details: {msg}"
            )
        elif result.verdict == Verdict.SKIP:
            ET.SubElement(tc, "skipped")

        # System output with eval details
        stdout = ET.SubElement(tc, "system-out")
        lines = [f"actual_output: {result.actual_output}"]
        for er in result.eval_results:
            lines.append(f"{er.kind.value}: {er.score:.4f} — {er.detail}")
        stdout.text = "\n".join(lines)

    return testsuites


def write_junit_xml(suite_result: SuiteResult, path: str | Path) -> Path:
    """Write JUnit XML report to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(_build_junit_xml(suite_result))
    ET.indent(tree, space="  ")
    tree.write(str(path), encoding="unicode", xml_declaration=True)
    return path


def junit_xml_string(suite_result: SuiteResult) -> str:
    """Return JUnit XML as a string."""
    root = _build_junit_xml(suite_result)
    ET.indent(root, space="  ")
    buf = StringIO()
    tree = ET.ElementTree(root)
    tree.write(buf, encoding="unicode", xml_declaration=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Console reporter
# ---------------------------------------------------------------------------

_VERDICT_EMOJI = {
    Verdict.PASS: "✅",
    Verdict.FAIL: "❌",
    Verdict.WARN: "⚠️",
    Verdict.SKIP: "⏭️",
}


def print_report(suite_result: SuiteResult, file: TextIO = sys.stdout) -> None:
    """Print a human-readable test report to the terminal."""
    w = file.write

    w(f"\n{'=' * 60}\n")
    w(f"  AgentGate — {suite_result.suite_name}\n")
    w(f"{'=' * 60}\n\n")

    for result in suite_result.results:
        emoji = _VERDICT_EMOJI.get(result.verdict, "?")
        name = result.test_case.name or result.test_case.id
        w(f"  {emoji} {name}")
        if result.duration_ms > 0:
            w(f"  ({result.duration_ms:.0f}ms)")
        w("\n")

        if result.error:
            w(f"     ERROR: {result.error}\n")
        else:
            for er in result.eval_results:
                w(f"     [{er.kind.value}] {er.score:.4f} — {er.detail}\n")
        w("\n")

    w(f"{'-' * 60}\n")
    w(
        f"  Total: {suite_result.total}  "
        f"Passed: {suite_result.passed_count}  "
        f"Failed: {suite_result.failed_count}  "
        f"Rate: {suite_result.pass_rate:.1%}\n"
    )
    w(f"  Duration: {suite_result.duration_ms:.0f}ms\n")
    w(f"{'=' * 60}\n\n")


# ---------------------------------------------------------------------------
# run_regression — main entry point for CI
# ---------------------------------------------------------------------------

def run_regression(
    suite: TestSuite,
    agent_fn: AgentFn | None = None,
    *,
    junit_path: str | Path | None = None,
    tags: list[str] | None = None,
    pass_k: int | None = None,
    verbose: bool = True,
    fail_fast: bool = False,
) -> SuiteResult:
    """Run a full regression suite and optionally write JUnit XML.

    This is the primary CI integration point.  Use it in your pipeline::

        result = run_regression(suite, my_agent, junit_path="report.xml")
        sys.exit(0 if result.all_passed else 1)

    Args:
        suite: The test suite to run.
        agent_fn: Agent callable for data-driven tests.
        junit_path: If set, write JUnit XML here.
        tags: Filter tests by tag.
        pass_k: Use PassK strategy with this K value.
        verbose: Print console report.
        fail_fast: (reserved for future use)

    Returns:
        SuiteResult with all test outcomes.
    """
    result = suite.run(agent_fn, tags=tags, pass_k=pass_k)

    if verbose:
        print_report(result)

    if junit_path:
        p = write_junit_xml(result, junit_path)
        if verbose:
            print(f"📄 JUnit XML written to {p}")

    return result


def ci_main(
    suite: TestSuite,
    agent_fn: AgentFn | None = None,
    *,
    junit_path: str | Path = "agentgate-report.xml",
    tags: list[str] | None = None,
    pass_k: int | None = None,
) -> None:
    """Run regression and exit with appropriate code.

    Exit 0 = all passed (deploy OK).
    Exit 1 = failures detected (block deployment).
    """
    result = run_regression(
        suite,
        agent_fn,
        junit_path=junit_path,
        tags=tags,
        pass_k=pass_k,
        verbose=True,
    )
    sys.exit(0 if result.all_passed else 1)
