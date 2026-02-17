"""
AgentGate Mock Mode — Test scenarios without real LLM calls.

Provides MockAgent that replays pre-recorded traces, enabling:
- Free, instant scenario testing during development
- Deterministic CI runs (no API cost, no flakiness)
- Trace recording from real runs for replay in tests

Usage:
    # Record a trace from a real agent
    trace = real_adapter.run("Book a flight")
    MockAgent.save_trace(trace, "traces/book_flight.json")

    # Replay in tests (free, instant, deterministic)
    mock = MockAgent.from_traces("traces/")
    result = suite.run(mock)

    # Or define traces inline for quick testing
    mock = MockAgent({
        "book flight": AgentTrace(input="book flight", steps=[...], output="Done"),
    })
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from agentgate.scenario import AgentAdapter, AgentTrace, AgentStep, StepKind

# Re-export TraceRecorder for backward compatibility
from agentgate.trace import TraceRecorder


class MockAgent(AgentAdapter):
    """
    A mock agent that returns pre-defined traces for given inputs.

    Enables testing without real LLM calls:
    - Development: fast iteration, zero cost
    - CI: deterministic, reproducible, no API keys needed
    - Demo: guaranteed consistent output

    Matching is fuzzy: input is lowercased and matched against trace keys
    using substring matching, so "Book me a flight to Tokyo" matches
    a trace keyed as "book flight".
    """

    def __init__(self, traces: dict[str, AgentTrace] | None = None,
                 default_trace: AgentTrace | None = None):
        """
        Args:
            traces: Dict mapping input patterns to AgentTrace objects.
                    Keys are lowercased and used for fuzzy matching.
            default_trace: Fallback trace returned for any unmatched input.
                    Useful for adversarial testing where you want the agent
                    to refuse all unexpected inputs uniformly.
        """
        self._traces: dict[str, AgentTrace] = {}
        self.default_trace = default_trace
        if traces:
            for key, trace in traces.items():
                self._traces[key.lower().strip()] = trace

    def add_trace(self, input_pattern: str, trace: AgentTrace) -> "MockAgent":
        """Register a trace for an input pattern."""
        self._traces[input_pattern.lower().strip()] = trace
        return self

    def run(self, input_text: str) -> AgentTrace:
        """Find the best matching trace for the input."""
        normalized = input_text.lower().strip()

        # Exact match first
        if normalized in self._traces:
            trace = self._traces[normalized]
            trace.input = input_text
            return trace

        # Substring match
        for key, trace in self._traces.items():
            if key in normalized or normalized in key:
                trace.input = input_text
                return trace

        # Regex match
        for key, trace in self._traces.items():
            try:
                if re.search(key, normalized):
                    trace.input = input_text
                    return trace
            except re.error:
                pass

        # Default trace fallback (for adversarial testing, etc.)
        if self.default_trace is not None:
            import copy
            trace = copy.deepcopy(self.default_trace)
            trace.input = input_text
            return trace

        # No match — return error trace
        return AgentTrace(
            input=input_text,
            output=f"[MockAgent] No trace registered for: {input_text}",
            steps=[
                AgentStep(
                    kind=StepKind.ERROR,
                    name="mock_no_match",
                    error=f"No mock trace found for input: {input_text}. "
                          f"Registered patterns: {list(self._traces.keys())}",
                )
            ],
        )

    # ---- Serialization ----

    @staticmethod
    def save_trace(trace: AgentTrace, path: str | Path) -> None:
        """Save a trace to JSON for later replay."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "input": trace.input,
            "output": trace.output,
            "total_duration_ms": trace.total_duration_ms,
            "metadata": trace.metadata,
            "steps": [
                {
                    "kind": step.kind.value,
                    "name": step.name,
                    "input": step.input,
                    "output": step.output,
                    "duration_ms": step.duration_ms,
                    "metadata": step.metadata,
                    "error": step.error,
                }
                for step in trace.steps
            ],
        }
        p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def load_trace(path: str | Path) -> AgentTrace:
        """Load a trace from JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        steps = [
            AgentStep(
                kind=StepKind(s["kind"]),
                name=s["name"],
                input=s.get("input", {}),
                output=s.get("output"),
                duration_ms=s.get("duration_ms", 0),
                metadata=s.get("metadata", {}),
                error=s.get("error"),
            )
            for s in data.get("steps", [])
        ]
        return AgentTrace(
            input=data["input"],
            output=data.get("output"),
            steps=steps,
            total_duration_ms=data.get("total_duration_ms", 0),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_traces(cls, directory: str | Path) -> "MockAgent":
        """
        Load all traces from a directory.

        Matching strategy (in order):
        1. Use the trace's stored 'input' field as the primary key (most reliable)
        2. Fall back to cleaned filename as secondary key

        This means traces saved by TraceRecorder (which stores full input in JSON)
        will match correctly regardless of filename truncation.
        """
        mock = cls()
        dir_path = Path(directory)
        for f in sorted(dir_path.glob("*.json")):
            trace = cls.load_trace(f)
            # Primary key: the original input stored in the trace (full, untruncated)
            if trace.input:
                mock._traces[trace.input.lower().strip()] = trace
            # Secondary key: cleaned filename (for manually created trace files)
            # Strip leading counter prefix like "0001_"
            stem = f.stem
            stem = re.sub(r'^\d+_', '', stem)  # remove "0001_" prefix
            key = stem.replace("_", " ").replace("-", " ").lower().strip()
            if key and key not in mock._traces:
                mock._traces[key] = trace
        return mock
