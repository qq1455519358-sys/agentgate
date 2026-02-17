"""Test reproducibility (Same Prompt Different Outcomes, 2026)."""
from agentgate import (
    VarianceReport, variance_report, reproducibility_score,
    expect_reproducible,
)
from agentgate.scenario import ScenarioResult, ExpectationResult, AgentTrace


def _make_results(pass_fail: list[bool]) -> list[ScenarioResult]:
    """Helper to create scenario results with given outcomes."""
    results = []
    for i, passed in enumerate(pass_fail):
        r = ScenarioResult(
            scenario_name="test_scenario",
            input="test",
            passed=passed,
            expectations=[] if passed else [
                ExpectationResult(False, "check", "failed")
            ],
            trace=AgentTrace(input="test"),
        )
        results.append(r)
    return results


def test_variance_report_all_pass():
    """All runs pass → high consistency."""
    results = _make_results([True, True, True, True, True])
    report = variance_report(results)
    assert report.total_runs == 5
    assert report.pass_count == 5
    assert report.consistency_rate == 1.0


def test_variance_report_all_fail():
    """All runs fail → also high consistency (consistently failing)."""
    results = _make_results([False, False, False])
    report = variance_report(results)
    assert report.consistency_rate == 1.0
    assert report.pass_count == 0


def test_variance_report_mixed():
    """Mixed results → low consistency."""
    results = _make_results([True, False, True, False, True])
    report = variance_report(results)
    assert report.consistency_rate == 0.6  # 3/5 majority
    assert report.pass_count == 3


def test_variance_report_empty():
    """Empty results."""
    report = variance_report([])
    assert report.total_runs == 0
    assert report.consistency_rate == 0.0


def test_variance_report_summary():
    results = _make_results([True, True, False])
    report = variance_report(results)
    summary = report.summary()
    assert "runs=3" in summary
    assert "consistency=" in summary


def test_reproducibility_score_perfect():
    """All same outcome → high score."""
    results = _make_results([True] * 10)
    score = reproducibility_score(results)
    assert score >= 0.9


def test_reproducibility_score_random():
    """50/50 split → low score."""
    results = _make_results([True, False, True, False])
    score = reproducibility_score(results)
    assert score < 0.8


def test_reproducibility_score_empty():
    assert reproducibility_score([]) == 0.0


def test_expect_reproducible_pass():
    """High consistency → pass."""
    results = _make_results([True] * 8 + [False] * 2)
    exp = expect_reproducible(
        min_consistency=0.7,
        multi_results=results,
    )
    from agentgate import AgentTrace
    result = exp.check(AgentTrace(input="q"))
    assert result.passed
    assert "80.0%" in result.detail


def test_expect_reproducible_fail():
    """Low consistency → fail."""
    results = _make_results([True, False, True, False, True])
    exp = expect_reproducible(
        min_consistency=0.8,
        multi_results=results,
    )
    from agentgate import AgentTrace
    result = exp.check(AgentTrace(input="q"))
    assert not result.passed
    assert "60.0%" in result.detail


def test_expect_reproducible_no_multi():
    """Without multi results, passes with note."""
    exp = expect_reproducible()
    from agentgate import AgentTrace
    result = exp.check(AgentTrace(input="q"))
    assert result.passed
    assert "single run" in result.detail
