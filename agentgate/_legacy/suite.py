"""AgentGate test suite engine.

Provides:
  - TestSuite       — manage and run collections of test cases
  - GoldenDataset   — load expected I/O pairs from JSONL
  - PassK           — run non-deterministic agents K times, average scores
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Awaitable

from .core import Evaluator, SemScore, evaluate as _quick_evaluate
from .types import (
    EvalResult,
    ScoreKind,
    SuiteResult,
    TestCase,
    TestResult,
    Verdict,
)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AgentFn = Callable[[str], str]
AsyncAgentFn = Callable[[str], Awaitable[str]]


# ---------------------------------------------------------------------------
# GoldenDataset
# ---------------------------------------------------------------------------

class GoldenDataset:
    """Load test cases from a JSONL file.

    Each line must be a JSON object with at least ``input`` and ``expected``
    fields.  Optional: ``name``, ``tags``, ``meta``.

    Example JSONL::

        {"input": "hi", "expected": "hello", "name": "greeting"}
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cases: list[TestCase] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Golden dataset not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {lineno} of {self.path}: {exc}"
                    ) from exc
                if "input" not in obj or "expected" not in obj:
                    raise ValueError(
                        f"Line {lineno}: each entry must have 'input' and 'expected'"
                    )
                self._cases.append(
                    TestCase(
                        input=obj["input"],
                        expected=obj["expected"],
                        name=obj.get("name", f"golden_{lineno}"),
                        tags=obj.get("tags", []),
                        meta=obj.get("meta", {}),
                    )
                )

    @property
    def cases(self) -> list[TestCase]:
        return list(self._cases)

    def __len__(self) -> int:
        return len(self._cases)

    def __iter__(self):
        return iter(self._cases)


# ---------------------------------------------------------------------------
# PassK strategy
# ---------------------------------------------------------------------------

class PassK:
    """Run an agent K times per input and average the evaluation scores.

    Implements the Pass^K evaluation strategy for non-deterministic AI agents.
    Because LLM outputs vary across invocations (temperature > 0, tool use,
    retrieval randomness), a single sample is insufficient to characterize
    agent quality.  Pass^K runs K independent trials and aggregates scores.

    Reference:
        Cresta AI (2025). "Pass^K: A Methodology for Evaluating
        Non-Deterministic AI Agent Outputs." Cresta Engineering Blog.

    The method extends the pass@k metric from code generation (Chen et al.,
    2021, "Evaluating Large Language Models Trained on Code",
    arXiv:2107.03374) to general agent evaluation by computing the mean and
    standard deviation of scores across K independent runs, providing both
    a central tendency estimate and a variance signal for flaky agents.

    Args:
        k: Number of independent runs per input.  Higher K gives more
           stable estimates but costs more compute.  Recommended: 5–20.
        evaluator: The scoring evaluator to use per run (default: SemScore).

    Usage::

        pk = PassK(k=10)
        result = pk.run(my_agent, "summarize this", expected_summary)
        print(result.score)          # mean score across 10 runs
        print(result.meta["std"])    # standard deviation
    """

    def __init__(
        self,
        k: int = 5,
        evaluator: Evaluator | None = None,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.evaluator = evaluator or SemScore()

    def run(
        self,
        agent_fn: AgentFn,
        input_text: str,
        expected: str,
    ) -> EvalResult:
        """Run agent_fn k times, evaluate each, return averaged result."""
        scores: list[float] = []
        for _ in range(self.k):
            output = agent_fn(input_text)
            result = self.evaluator.evaluate(output, expected)
            scores.append(result.score)

        mean_score = round(statistics.mean(scores), 4)
        std_dev = round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0
        # Use the evaluator's threshold if available, else 0.5
        threshold = getattr(self.evaluator, "threshold", 0.5)
        verdict = Verdict.PASS if mean_score >= threshold else Verdict.FAIL
        return EvalResult(
            score=mean_score,
            verdict=verdict,
            kind=ScoreKind.COMPOSITE,
            detail=f"pass@{self.k}: mean={mean_score}, std={std_dev}",
            meta={"k": self.k, "scores": scores, "std": std_dev},
        )

    async def arun(
        self,
        agent_fn: AsyncAgentFn,
        input_text: str,
        expected: str,
    ) -> EvalResult:
        """Async variant — runs all K calls concurrently."""
        tasks = [agent_fn(input_text) for _ in range(self.k)]
        outputs = await asyncio.gather(*tasks)
        scores: list[float] = []
        for output in outputs:
            result = self.evaluator.evaluate(output, expected)
            scores.append(result.score)

        mean_score = round(statistics.mean(scores), 4)
        std_dev = round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0
        threshold = getattr(self.evaluator, "threshold", 0.5)
        verdict = Verdict.PASS if mean_score >= threshold else Verdict.FAIL
        return EvalResult(
            score=mean_score,
            verdict=verdict,
            kind=ScoreKind.COMPOSITE,
            detail=f"pass@{self.k}: mean={mean_score}, std={std_dev}",
            meta={"k": self.k, "scores": scores, "std": std_dev},
        )


# ---------------------------------------------------------------------------
# TestSuite
# ---------------------------------------------------------------------------

class TestSuite:
    """Declarative test suite for agent evaluation.

    Usage::

        suite = TestSuite("my-agent-tests")

        @suite.test(tags=["greeting"])
        def test_hello():
            actual = my_agent("say hello")
            return actual, "Hello! How can I help you?"

        results = suite.run()
    """

    def __init__(
        self,
        name: str = "default",
        evaluators: list[Evaluator] | None = None,
        threshold: float = 0.75,
    ) -> None:
        self.name = name
        self.evaluators = evaluators or [SemScore(threshold=threshold)]
        self.threshold = threshold
        self._tests: list[dict[str, Any]] = []
        self._cases: list[TestCase] = []

    # -- Decorator API -----------------------------------------------------

    def test(
        self,
        *,
        name: str = "",
        tags: list[str] | None = None,
        evaluators: list[Evaluator] | None = None,
    ) -> Callable:
        """Decorator to register a test function.

        The decorated function must return ``(actual_output, expected_output)``
        as a tuple of two strings.
        """

        def decorator(fn: Callable[[], tuple[str, str]]) -> Callable[[], tuple[str, str]]:
            self._tests.append({
                "fn": fn,
                "name": name or fn.__name__,
                "tags": tags or [],
                "evaluators": evaluators,
            })
            return fn

        return decorator

    # -- Data-driven API ---------------------------------------------------

    def add_case(self, case: TestCase) -> None:
        self._cases.append(case)

    def add_cases(self, cases: list[TestCase]) -> None:
        self._cases.extend(cases)

    def load_golden(self, path: str | Path) -> None:
        """Load test cases from a golden JSONL dataset."""
        dataset = GoldenDataset(path)
        self._cases.extend(dataset.cases)

    # -- Run ---------------------------------------------------------------

    def run(
        self,
        agent_fn: AgentFn | None = None,
        *,
        tags: list[str] | None = None,
        pass_k: int | None = None,
    ) -> SuiteResult:
        """Execute all registered tests and data-driven cases.

        Args:
            agent_fn: Agent callable for data-driven cases (input → output).
            tags: If given, only run tests matching at least one tag.
            pass_k: If set, use PassK strategy with this K value.
        """
        suite_start = time.perf_counter()
        results: list[TestResult] = []

        # 1. Decorator-based tests
        for entry in self._tests:
            if tags and not set(tags) & set(entry["tags"]):
                continue
            test_case = TestCase(
                input="(decorator test)",
                expected="",
                name=entry["name"],
                tags=entry["tags"],
            )
            t0 = time.perf_counter()
            try:
                actual, expected = entry["fn"]()
                test_case.expected = expected
                evals = entry["evaluators"] or self.evaluators
                eval_results = [e.evaluate(actual, expected) for e in evals]
                duration = (time.perf_counter() - t0) * 1000
                results.append(
                    TestResult(
                        test_case=test_case,
                        eval_results=eval_results,
                        actual_output=actual,
                        duration_ms=round(duration, 2),
                    )
                )
            except Exception as exc:
                duration = (time.perf_counter() - t0) * 1000
                results.append(
                    TestResult(
                        test_case=test_case,
                        error=str(exc),
                        duration_ms=round(duration, 2),
                    )
                )

        # 2. Data-driven cases
        for case in self._cases:
            if tags and not set(tags) & set(case.tags):
                continue
            if agent_fn is None:
                results.append(
                    TestResult(
                        test_case=case,
                        error="No agent_fn provided for data-driven case",
                    )
                )
                continue

            t0 = time.perf_counter()
            try:
                if pass_k and pass_k > 1:
                    pk = PassK(k=pass_k, evaluator=self.evaluators[0])
                    eval_result = pk.run(agent_fn, case.input, case.expected)
                    actual_output = "(pass@k aggregated)"
                    eval_results = [eval_result]
                else:
                    actual_output = agent_fn(case.input)
                    eval_results = [
                        e.evaluate(actual_output, case.expected) for e in self.evaluators
                    ]
                duration = (time.perf_counter() - t0) * 1000
                results.append(
                    TestResult(
                        test_case=case,
                        eval_results=eval_results,
                        actual_output=actual_output,
                        duration_ms=round(duration, 2),
                    )
                )
            except Exception as exc:
                duration = (time.perf_counter() - t0) * 1000
                results.append(
                    TestResult(
                        test_case=case,
                        error=str(exc),
                        duration_ms=round(duration, 2),
                    )
                )

        total_ms = (time.perf_counter() - suite_start) * 1000
        return SuiteResult(
            suite_name=self.name,
            results=results,
            duration_ms=round(total_ms, 2),
        )

    # -- Async run ---------------------------------------------------------

    async def arun(
        self,
        agent_fn: AsyncAgentFn | None = None,
        *,
        tags: list[str] | None = None,
        pass_k: int | None = None,
    ) -> SuiteResult:
        """Async variant of :meth:`run`.

        Data-driven cases run concurrently when using an async agent.
        """
        suite_start = time.perf_counter()
        results: list[TestResult] = []

        # Decorator tests (run sequentially — they are sync)
        for entry in self._tests:
            if tags and not set(tags) & set(entry["tags"]):
                continue
            test_case = TestCase(
                input="(decorator test)",
                expected="",
                name=entry["name"],
                tags=entry["tags"],
            )
            t0 = time.perf_counter()
            try:
                actual, expected = entry["fn"]()
                test_case.expected = expected
                evals = entry["evaluators"] or self.evaluators
                eval_results = [e.evaluate(actual, expected) for e in evals]
                duration = (time.perf_counter() - t0) * 1000
                results.append(
                    TestResult(
                        test_case=test_case,
                        eval_results=eval_results,
                        actual_output=actual,
                        duration_ms=round(duration, 2),
                    )
                )
            except Exception as exc:
                duration = (time.perf_counter() - t0) * 1000
                results.append(
                    TestResult(
                        test_case=test_case,
                        error=str(exc),
                        duration_ms=round(duration, 2),
                    )
                )

        # Data-driven cases (async, concurrent)
        async def _run_case(case: TestCase) -> TestResult:
            if agent_fn is None:
                return TestResult(
                    test_case=case,
                    error="No agent_fn provided for data-driven case",
                )
            t0 = time.perf_counter()
            try:
                if pass_k and pass_k > 1:
                    pk = PassK(k=pass_k, evaluator=self.evaluators[0])
                    eval_result = await pk.arun(agent_fn, case.input, case.expected)
                    actual_output = "(pass@k aggregated)"
                    eval_results = [eval_result]
                else:
                    actual_output = await agent_fn(case.input)
                    eval_results = [
                        e.evaluate(actual_output, case.expected) for e in self.evaluators
                    ]
                duration = (time.perf_counter() - t0) * 1000
                return TestResult(
                    test_case=case,
                    eval_results=eval_results,
                    actual_output=actual_output,
                    duration_ms=round(duration, 2),
                )
            except Exception as exc:
                duration = (time.perf_counter() - t0) * 1000
                return TestResult(
                    test_case=case,
                    error=str(exc),
                    duration_ms=round(duration, 2),
                )

        filtered = [
            c for c in self._cases
            if not tags or set(tags) & set(c.tags)
        ]
        case_results = await asyncio.gather(*[_run_case(c) for c in filtered])
        results.extend(case_results)

        total_ms = (time.perf_counter() - suite_start) * 1000
        return SuiteResult(
            suite_name=self.name,
            results=results,
            duration_ms=round(total_ms, 2),
        )
