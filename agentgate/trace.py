"""
AgentGate Trace — Trace data structures and recording.

Provides:
- AgentTrace, AgentStep, StepKind — execution trace data structures
- TraceRecorder — record real agent runs for mock replay

These are re-exported from scenario.py (data structures) and extended
with recording capabilities.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agentgate.scenario import AgentAdapter, AgentTrace, AgentStep, StepKind


class TraceRecorder:
    """
    Records traces from real agent runs for later mock replay.

    Usage:
        recorder = TraceRecorder("traces/")

        # Wrap your real adapter
        recorded_adapter = recorder.wrap(real_adapter)

        # Run scenarios — traces are automatically saved
        suite.run(recorded_adapter)

        # Later, replay without API calls
        from agentgate import MockAgent
        mock = MockAgent.from_traces("traces/")
        suite.run(mock)
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def wrap(self, adapter: AgentAdapter) -> "_RecordingAdapter":
        """Wrap an adapter to record all traces."""
        return _RecordingAdapter(adapter, self)

    def _save(self, trace: AgentTrace) -> Path:
        """Save a trace and return the file path."""
        from agentgate.mock import MockAgent

        self._count += 1
        safe_name = re.sub(r'[^\w\s-]', '', trace.input[:50]).strip().replace(' ', '_')
        filename = f"{self._count:04d}_{safe_name}.json"
        path = self.output_dir / filename
        MockAgent.save_trace(trace, path)
        return path


class _RecordingAdapter(AgentAdapter):
    """Internal: wraps an adapter to record traces."""

    def __init__(self, inner: AgentAdapter, recorder: TraceRecorder):
        self._inner = inner
        self._recorder = recorder

    def run(self, input_text: str) -> AgentTrace:
        trace = self._inner.run(input_text)
        self._recorder._save(trace)
        return trace
