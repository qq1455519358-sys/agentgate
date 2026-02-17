"""AgentGate evaluation engine.

Five evaluators with a unified interface:
  - SemScore              — semantic similarity via sentence-transformers
  - RuleCheck             — regex / keyword matching
  - LLMJudge              — GPT-4o-mini as an async judge (optional dependency)
  - HallucinationDetector — semantic entropy-based hallucination detection
  - DriftDetector         — Population Stability Index for distribution drift
"""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

from .types import EvalResult, ScoreKind, Verdict


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Evaluator(ABC):
    """Base class every evaluator must implement."""

    @abstractmethod
    def evaluate(self, actual: str, expected: str, **kwargs: Any) -> EvalResult:
        ...


# ---------------------------------------------------------------------------
# SemScore – Semantic similarity
# ---------------------------------------------------------------------------

class SemScore(Evaluator):
    """Semantic similarity score using sentence-transformers embeddings.

    Computes cosine similarity between the actual and expected outputs in
    embedding space.  Based on the SemScore methodology for automated
    evaluation of instruction-tuned language models.

    Reference:
        Aynetdinov, A. & Akbik, A. (2024). "SemScore: Automated Evaluation
        of Instruction-Tuned LLMs based on Semantic Textual Similarity."
        arXiv:2401.17072.  https://arxiv.org/abs/2401.17072

    The paper demonstrates that STS-based evaluation via pre-trained
    sentence embeddings achieves the highest correlation with human judgement
    among reference-based metrics, outperforming BLEU, ROUGE, and BERTScore
    for instruction-following tasks.

    Model is loaded lazily on first call and cached for the process lifetime.
    Default model: ``all-MiniLM-L6-v2``.  Default threshold: 0.75.
    """

    _model: Any = None  # class-level cache

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.75,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold

    # Lazy singleton -------------------------------------------------------
    def _get_model(self) -> Any:
        if SemScore._model is None:
            from sentence_transformers import SentenceTransformer

            SemScore._model = SentenceTransformer(self.model_name)
        return SemScore._model

    # Core API -------------------------------------------------------------
    def evaluate(self, actual: str, expected: str, **kwargs: Any) -> EvalResult:
        model = self._get_model()
        embeddings = model.encode([actual, expected], normalize_embeddings=True)
        similarity: float = float(np.dot(embeddings[0], embeddings[1]))
        # Clamp to [0, 1] – cosine can be slightly negative for unrelated text
        similarity = max(0.0, min(1.0, similarity))
        verdict = Verdict.PASS if similarity >= self.threshold else Verdict.FAIL
        return EvalResult(
            score=round(similarity, 4),
            verdict=verdict,
            kind=ScoreKind.SEMANTIC,
            detail=f"cosine={similarity:.4f} (threshold={self.threshold})",
        )

    def score(self, actual: str, expected: str) -> float:
        """Convenience: return raw float score."""
        return self.evaluate(actual, expected).score


# ---------------------------------------------------------------------------
# RuleCheck – regex / keyword rules
# ---------------------------------------------------------------------------

class _Rule:
    """Internal representation of a single rule."""

    __slots__ = ("pattern", "is_regex", "negate", "label")

    def __init__(
        self,
        pattern: str,
        *,
        is_regex: bool = False,
        negate: bool = False,
        label: str = "",
    ) -> None:
        self.pattern = pattern
        self.is_regex = is_regex
        self.negate = negate
        self.label = label or pattern

    def check(self, text: str) -> bool:
        """Return True if the rule passes."""
        if self.is_regex:
            found = bool(re.search(self.pattern, text, re.IGNORECASE))
        else:
            found = self.pattern.lower() in text.lower()
        return (not found) if self.negate else found


class RuleCheck(Evaluator):
    """Keyword / regex rule-based evaluator.

    Usage::

        rc = RuleCheck()
        rc.must_contain("hello")
        rc.must_not_contain_regex(r"\\bcrap\\b")
        result = rc.evaluate(actual, expected)
    """

    def __init__(self) -> None:
        self._rules: list[_Rule] = []

    # Builder API ----------------------------------------------------------
    def must_contain(self, keyword: str, *, label: str = "") -> "RuleCheck":
        self._rules.append(_Rule(keyword, label=label or f"must_contain({keyword!r})"))
        return self

    def must_not_contain(self, keyword: str, *, label: str = "") -> "RuleCheck":
        self._rules.append(
            _Rule(keyword, negate=True, label=label or f"must_not_contain({keyword!r})")
        )
        return self

    def must_match_regex(self, pattern: str, *, label: str = "") -> "RuleCheck":
        self._rules.append(
            _Rule(pattern, is_regex=True, label=label or f"must_match({pattern!r})")
        )
        return self

    def must_not_contain_regex(self, pattern: str, *, label: str = "") -> "RuleCheck":
        self._rules.append(
            _Rule(
                pattern,
                is_regex=True,
                negate=True,
                label=label or f"must_not_match({pattern!r})",
            )
        )
        return self

    # Core API -------------------------------------------------------------
    def evaluate(self, actual: str, expected: str = "", **kwargs: Any) -> EvalResult:
        if not self._rules:
            return EvalResult(
                score=1.0,
                verdict=Verdict.PASS,
                kind=ScoreKind.RULE,
                detail="no rules defined",
            )

        passed: list[str] = []
        failed: list[str] = []
        for rule in self._rules:
            if rule.check(actual):
                passed.append(rule.label)
            else:
                failed.append(rule.label)

        total = len(self._rules)
        score = len(passed) / total
        verdict = Verdict.PASS if not failed else Verdict.FAIL
        detail_parts: list[str] = []
        if failed:
            detail_parts.append(f"failed: {', '.join(failed)}")
        if passed:
            detail_parts.append(f"passed: {', '.join(passed)}")
        return EvalResult(
            score=round(score, 4),
            verdict=verdict,
            kind=ScoreKind.RULE,
            detail="; ".join(detail_parts),
        )


# ---------------------------------------------------------------------------
# LLMJudge – GPT-4o-mini as evaluator (async, optional)
# ---------------------------------------------------------------------------

_DEFAULT_JUDGE_PROMPT = """\
You are an evaluation judge. Compare the ACTUAL output against the EXPECTED output.

EXPECTED:
{expected}

ACTUAL:
{actual}

Rate how well the actual output matches the expected output.
Respond with ONLY a JSON object (no markdown):
{{"score": <float 0-1>, "reasoning": "<brief explanation>"}}
"""


class LLMJudge(Evaluator):
    """Use an OpenAI chat model (default: gpt-4o-mini) as a judge.

    Implements the "LLM-as-a-Judge" paradigm where a frontier model scores
    the quality of agent outputs against expected references.

    Reference:
        Zheng, L. et al. (2023). "Judging LLM-as-a-Judge with MT-Bench
        and Chatbot Arena." arXiv:2306.05685.
        https://arxiv.org/abs/2306.05685

    Requires the ``openai`` package and a valid ``OPENAI_API_KEY`` env var.
    Supports both sync ``evaluate()`` and async ``aevaluate()``.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        threshold: float = 0.7,
        prompt_template: str = _DEFAULT_JUDGE_PROMPT,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.prompt_template = prompt_template
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    def _build_client(self) -> Any:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "LLMJudge requires the 'openai' package. "
                "Install it with: pip install openai"
            ) from exc

        kwargs: dict[str, Any] = {"timeout": self.timeout}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return openai.OpenAI(**kwargs)

    def _build_async_client(self) -> Any:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "LLMJudge requires the 'openai' package. "
                "Install it with: pip install openai"
            ) from exc

        kwargs: dict[str, Any] = {"timeout": self.timeout}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return openai.AsyncOpenAI(**kwargs)

    @staticmethod
    def _parse_response(text: str) -> tuple[float, str]:
        """Extract score and reasoning from LLM JSON response."""
        import json

        text = text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        data = json.loads(text)
        score = float(data.get("score", 0.0))
        reasoning = str(data.get("reasoning", ""))
        return max(0.0, min(1.0, score)), reasoning

    def evaluate(self, actual: str, expected: str, **kwargs: Any) -> EvalResult:
        """Synchronous evaluation via OpenAI chat completion."""
        client = self._build_client()
        prompt = self.prompt_template.format(actual=actual, expected=expected)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content or ""
        try:
            score, reasoning = self._parse_response(raw)
        except Exception:
            return EvalResult(
                score=0.0,
                verdict=Verdict.FAIL,
                kind=ScoreKind.LLM_JUDGE,
                detail=f"Failed to parse judge response: {raw[:200]}",
            )
        verdict = Verdict.PASS if score >= self.threshold else Verdict.FAIL
        return EvalResult(
            score=round(score, 4),
            verdict=verdict,
            kind=ScoreKind.LLM_JUDGE,
            detail=reasoning,
            meta={"model": self.model, "raw_response": raw},
        )

    async def aevaluate(self, actual: str, expected: str, **kwargs: Any) -> EvalResult:
        """Async evaluation via OpenAI async client."""
        client = self._build_async_client()
        prompt = self.prompt_template.format(actual=actual, expected=expected)
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            raw = response.choices[0].message.content or ""
            score, reasoning = self._parse_response(raw)
        except Exception as exc:
            return EvalResult(
                score=0.0,
                verdict=Verdict.FAIL,
                kind=ScoreKind.LLM_JUDGE,
                detail=f"LLM judge error: {exc}",
            )
        finally:
            await client.close()
        verdict = Verdict.PASS if score >= self.threshold else Verdict.FAIL
        return EvalResult(
            score=round(score, 4),
            verdict=verdict,
            kind=ScoreKind.LLM_JUDGE,
            detail=reasoning,
            meta={"model": self.model, "raw_response": raw},
        )


# ---------------------------------------------------------------------------
# HallucinationDetector – semantic entropy
# ---------------------------------------------------------------------------

class HallucinationDetector(Evaluator):
    """Detect hallucinations via semantic entropy over multiple samples.

    Generates K outputs for the same input using the provided agent function,
    clusters them by semantic similarity, and computes the entropy of the
    cluster distribution.  High entropy → the model is uncertain and likely
    hallucinating.  Low entropy → outputs are semantically consistent.

    Reference:
        Farquhar, S., Kossen, J., Kuhn, L. & Gal, Y. (2024). "Detecting
        hallucinations in large language models using semantic entropy."
        Nature, 630, 625–630.  https://doi.org/10.1038/s41586-024-07421-0

    The paper introduces *semantic entropy* — entropy computed over meaning
    clusters rather than raw token sequences — as a principled, unsupervised
    method for hallucination detection that outperforms token-level
    probability methods across question-answering benchmarks.

    Algorithm:
        1. Sample K responses for the same prompt.
        2. Embed all responses with a sentence-transformer model.
        3. Cluster embeddings via agglomerative clustering (cosine distance,
           threshold = ``cluster_threshold``).
        4. Compute Shannon entropy over the cluster size distribution.
        5. Normalize to [0, 1] by dividing by log(K).
        6. Score = 1 - normalized_entropy (high = consistent = good).

    Usage::

        detector = HallucinationDetector(k=10)
        result = detector.evaluate_agent(my_agent, "What is the capital of France?")
        print(result.score)   # close to 1.0 = consistent, no hallucination
        print(result.detail)  # "semantic_entropy=0.12, clusters=2, k=10"
    """

    _model: Any = None

    def __init__(
        self,
        k: int = 10,
        cluster_threshold: float = 0.3,
        entropy_threshold: float = 0.5,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        if k < 2:
            raise ValueError("k must be >= 2 for entropy computation")
        self.k = k
        self.cluster_threshold = cluster_threshold
        self.entropy_threshold = entropy_threshold
        self.model_name = model_name

    def _get_model(self) -> Any:
        if HallucinationDetector._model is None:
            from sentence_transformers import SentenceTransformer

            HallucinationDetector._model = SentenceTransformer(self.model_name)
        return HallucinationDetector._model

    @staticmethod
    def _shannon_entropy(counts: list[int]) -> float:
        """Compute Shannon entropy from cluster counts."""
        total = sum(counts)
        if total == 0:
            return 0.0
        probs = [c / total for c in counts if c > 0]
        return -sum(p * np.log(p) for p in probs)

    def _compute_semantic_entropy(self, responses: list[str]) -> tuple[float, int]:
        """Embed, cluster, and compute entropy.

        Returns (normalized_entropy, num_clusters).
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        model = self._get_model()
        embeddings = model.encode(responses, normalize_embeddings=True)

        if len(responses) == 1:
            return 0.0, 1

        # Pairwise cosine distance
        distances = pdist(embeddings, metric="cosine")

        # Agglomerative clustering
        Z = linkage(distances, method="average")
        labels = fcluster(Z, t=self.cluster_threshold, criterion="distance")

        # Cluster sizes
        from collections import Counter

        cluster_counts = list(Counter(labels).values())
        num_clusters = len(cluster_counts)

        # Shannon entropy, normalized by log(K) so it's in [0, 1]
        raw_entropy = self._shannon_entropy(cluster_counts)
        max_entropy = np.log(len(responses))
        normalized = float(raw_entropy / max_entropy) if max_entropy > 0 else 0.0

        return normalized, num_clusters

    def evaluate(self, actual: str, expected: str = "", **kwargs: Any) -> EvalResult:
        """Evaluate a single response (limited utility — use evaluate_agent).

        When called with a single string, wraps it in a list and computes
        trivial entropy.  For meaningful results, use :meth:`evaluate_agent`
        or pass ``responses`` in kwargs.
        """
        responses: list[str] = kwargs.get("responses", [actual])
        if len(responses) < 2:
            return EvalResult(
                score=1.0,
                verdict=Verdict.PASS,
                kind=ScoreKind.HALLUCINATION,
                detail="single response — entropy undefined, assuming consistent",
            )
        return self._evaluate_responses(responses)

    def _evaluate_responses(self, responses: list[str]) -> EvalResult:
        normalized_entropy, num_clusters = self._compute_semantic_entropy(responses)
        consistency_score = round(max(0.0, min(1.0, 1.0 - normalized_entropy)), 4)
        verdict = Verdict.PASS if consistency_score >= self.entropy_threshold else Verdict.FAIL
        return EvalResult(
            score=consistency_score,
            verdict=verdict,
            kind=ScoreKind.HALLUCINATION,
            detail=(
                f"semantic_entropy={normalized_entropy:.4f}, "
                f"clusters={num_clusters}, k={len(responses)}"
            ),
            meta={
                "normalized_entropy": normalized_entropy,
                "num_clusters": num_clusters,
                "k": len(responses),
            },
        )

    def evaluate_agent(
        self,
        agent_fn: "Callable[[str], str]",
        input_text: str,
    ) -> EvalResult:
        """Sample K responses from the agent and evaluate hallucination risk.

        This is the primary API.  Calls ``agent_fn(input_text)`` K times,
        collects responses, and computes semantic entropy.
        """
        responses = [agent_fn(input_text) for _ in range(self.k)]
        return self._evaluate_responses(responses)

    async def aevaluate_agent(
        self,
        agent_fn: "Callable[[str], Awaitable[str]]",
        input_text: str,
    ) -> EvalResult:
        """Async variant — samples K responses concurrently."""
        tasks = [agent_fn(input_text) for _ in range(self.k)]
        responses = await asyncio.gather(*tasks)
        return self._evaluate_responses(list(responses))


# ---------------------------------------------------------------------------
# DriftDetector – Population Stability Index
# ---------------------------------------------------------------------------

class DriftDetector:
    """Detect distribution drift in agent outputs using PSI.

    Compares the embedding distribution of a *reference* set (e.g., golden
    outputs from last release) against a *current* set (today's agent
    outputs).  Uses the Population Stability Index (PSI) to quantify how
    much the distribution has shifted.

    Reference:
        The Population Stability Index was introduced in credit scoring
        literature (Siddiqi, N., 2006. "Credit Risk Scorecards: Developing
        and Implementing Intelligent Credit Scoring." Wiley) and is widely
        used in model monitoring.

        Standard interpretation thresholds:
          - PSI < 0.10 → no significant drift
          - 0.10 ≤ PSI < 0.25 → moderate drift, investigate
          - PSI ≥ 0.25 → significant drift, action required

    Algorithm:
        1. Embed both reference and current response sets.
        2. Project embeddings onto principal components (PCA, n=10).
        3. For each component, bin values into ``n_bins`` quantile buckets
           (defined by the reference distribution).
        4. Compute per-bin PSI: Σ (p_current - p_ref) × ln(p_current / p_ref).
        5. Average across components for final PSI score.
        6. Convert to a 0–1 quality score: score = exp(-PSI).

    Usage::

        detector = DriftDetector()
        result = detector.compare(
            reference=["expected output 1", "expected output 2", ...],
            current=["actual output 1", "actual output 2", ...],
        )
        print(result.score)   # 0.95 = minimal drift
        print(result.detail)  # "psi=0.051, interpretation=no significant drift"
    """

    _model: Any = None

    def __init__(
        self,
        n_bins: int = 10,
        psi_threshold: float = 0.25,
        model_name: str = "all-MiniLM-L6-v2",
        n_components: int = 10,
    ) -> None:
        self.n_bins = n_bins
        self.psi_threshold = psi_threshold
        self.model_name = model_name
        self.n_components = n_components

    def _get_model(self) -> Any:
        if DriftDetector._model is None:
            from sentence_transformers import SentenceTransformer

            DriftDetector._model = SentenceTransformer(self.model_name)
        return DriftDetector._model

    @staticmethod
    def _psi_single(ref_bins: np.ndarray, cur_bins: np.ndarray) -> float:
        """Compute PSI for a single feature's binned proportions."""
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        ref_p = np.clip(ref_bins, eps, None)
        cur_p = np.clip(cur_bins, eps, None)
        return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))

    def compare(
        self,
        reference: list[str],
        current: list[str],
    ) -> EvalResult:
        """Compare reference and current output distributions.

        Args:
            reference: List of expected/golden outputs (the baseline).
            current: List of current agent outputs to check for drift.

        Returns:
            EvalResult with score = exp(-PSI), where 1.0 = no drift.
        """
        if len(reference) < 2 or len(current) < 2:
            return EvalResult(
                score=1.0,
                verdict=Verdict.WARN,
                kind=ScoreKind.DRIFT,
                detail="Need >= 2 samples in both reference and current sets",
            )

        model = self._get_model()
        ref_emb = model.encode(reference, normalize_embeddings=True)
        cur_emb = model.encode(current, normalize_embeddings=True)

        # PCA to reduce dimensionality
        from sklearn.decomposition import PCA

        n_comp = min(self.n_components, ref_emb.shape[1], len(reference) - 1)
        pca = PCA(n_components=n_comp)
        ref_proj = pca.fit_transform(ref_emb)
        cur_proj = pca.transform(cur_emb)

        # Compute PSI per component
        psi_values: list[float] = []
        n_bins = min(self.n_bins, len(reference))

        for dim in range(n_comp):
            ref_col = ref_proj[:, dim]
            cur_col = cur_proj[:, dim]

            # Quantile-based bin edges from the reference distribution
            edges = np.percentile(ref_col, np.linspace(0, 100, n_bins + 1))
            edges[0] = -np.inf
            edges[-1] = np.inf

            ref_counts = np.histogram(ref_col, bins=edges)[0].astype(float)
            cur_counts = np.histogram(cur_col, bins=edges)[0].astype(float)

            # Normalize to proportions
            ref_p = ref_counts / ref_counts.sum()
            cur_p = cur_counts / cur_counts.sum()

            psi_values.append(self._psi_single(ref_p, cur_p))

        avg_psi = float(np.mean(psi_values))

        # Interpretation
        if avg_psi < 0.10:
            interpretation = "no significant drift"
        elif avg_psi < 0.25:
            interpretation = "moderate drift — investigate"
        else:
            interpretation = "significant drift — action required"

        # Convert to 0–1 score (exponential decay)
        quality_score = round(float(np.exp(-avg_psi)), 4)
        verdict = Verdict.PASS if avg_psi < self.psi_threshold else Verdict.FAIL

        return EvalResult(
            score=quality_score,
            verdict=verdict,
            kind=ScoreKind.DRIFT,
            detail=f"psi={avg_psi:.4f}, interpretation={interpretation}",
            meta={
                "psi": round(avg_psi, 6),
                "psi_per_component": [round(v, 6) for v in psi_values],
                "n_components": n_comp,
                "n_reference": len(reference),
                "n_current": len(current),
                "interpretation": interpretation,
            },
        )


# ---------------------------------------------------------------------------
# Composite evaluator
# ---------------------------------------------------------------------------

class CompositeEvaluator(Evaluator):
    """Run multiple evaluators and aggregate scores with weights.

    Example::

        combo = CompositeEvaluator([
            (SemScore(threshold=0.8), 0.6),
            (RuleCheck().must_contain("hello"), 0.4),
        ])
        result = combo.evaluate(actual, expected)
    """

    def __init__(
        self,
        evaluators: Sequence[tuple[Evaluator, float]],
        threshold: float = 0.7,
    ) -> None:
        total_weight = sum(w for _, w in evaluators)
        if total_weight <= 0:
            raise ValueError("Total weight must be > 0")
        self._evaluators = [(e, w / total_weight) for e, w in evaluators]
        self.threshold = threshold

    def evaluate(self, actual: str, expected: str, **kwargs: Any) -> EvalResult:
        weighted_sum = 0.0
        details: list[str] = []
        sub_results: list[dict[str, Any]] = []

        for evaluator, weight in self._evaluators:
            result = evaluator.evaluate(actual, expected, **kwargs)
            weighted_sum += result.score * weight
            name = type(evaluator).__name__
            details.append(f"{name}={result.score:.3f}×{weight:.2f}")
            sub_results.append({
                "evaluator": name,
                "score": result.score,
                "weight": weight,
                "verdict": result.verdict.value,
            })

        score = round(weighted_sum, 4)
        verdict = Verdict.PASS if score >= self.threshold else Verdict.FAIL
        return EvalResult(
            score=score,
            verdict=verdict,
            kind=ScoreKind.COMPOSITE,
            detail=" + ".join(details) + f" = {score:.4f}",
            meta={"sub_results": sub_results},
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def evaluate(
    actual: str,
    expected: str,
    *,
    semantic: bool = True,
    rules: RuleCheck | None = None,
    llm_judge: bool = False,
    sem_threshold: float = 0.75,
    llm_threshold: float = 0.7,
    llm_model: str = "gpt-4o-mini",
) -> list[EvalResult]:
    """One-shot evaluation with sensible defaults.

    Returns a list of :class:`EvalResult`, one per active evaluator.
    """
    results: list[EvalResult] = []

    if semantic:
        results.append(SemScore(threshold=sem_threshold).evaluate(actual, expected))

    if rules is not None:
        results.append(rules.evaluate(actual, expected))

    if llm_judge:
        judge = LLMJudge(model=llm_model, threshold=llm_threshold)
        results.append(judge.evaluate(actual, expected))

    return results
