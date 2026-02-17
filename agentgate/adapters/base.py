"""
AgentGate Adapter Base — Framework-agnostic abstract interface.

This defines the contract that all framework adapters must implement.
The interface uses generic concepts only — no LangGraph, CrewAI, or
AutoGen-specific terminology.

To add a new framework:
    1. Subclass AgentAdapter
    2. Implement run() to execute the agent and return an AgentTrace
    3. Use capture_tool_call() and capture_step() helpers to build the trace

Example:
    class MyFrameworkAdapter(AgentAdapter):
        def __init__(self, app):
            self.app = app

        def run(self, input_text: str) -> AgentTrace:
            trace = AgentTrace(input=input_text)
            # ... run your framework, capture steps ...
            return trace
"""

from __future__ import annotations

import time
from typing import Any, Optional

from agentgate.scenario import AgentAdapter, AgentTrace, AgentStep, StepKind


class BaseAdapter(AgentAdapter):
    """
    Abstract base for framework-specific adapters.

    Provides helper methods for building traces from framework events.
    Subclasses must implement run().

    The interface is deliberately framework-agnostic:
    - capture_tool_call() — record a tool invocation
    - capture_step() — record any execution step
    - get_trace() — finalize and return the trace
    """

    def capture_tool_call(
        self,
        trace: AgentTrace,
        name: str,
        *,
        input: dict[str, Any] | None = None,
        output: Any = None,
        error: str | None = None,
        duration_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> AgentStep:
        """Record a tool call in the trace.

        Args:
            trace: The trace being built.
            name: Tool name.
            input: Arguments passed to the tool.
            output: Tool return value.
            error: Error message if the tool failed.
            duration_ms: How long the tool took.
            metadata: Extra metadata.

        Returns:
            The created AgentStep.
        """
        step = AgentStep(
            kind=StepKind.TOOL_CALL,
            name=name,
            input=input or {},
            output=output,
            error=error,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        trace.steps.append(step)
        return step

    def capture_step(
        self,
        trace: AgentTrace,
        kind: StepKind,
        name: str,
        *,
        input: dict[str, Any] | None = None,
        output: Any = None,
        error: str | None = None,
        duration_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> AgentStep:
        """Record any execution step in the trace.

        Args:
            trace: The trace being built.
            kind: Type of step (tool_call, llm_call, state_change, etc.).
            name: Step identifier.
            input: Input data.
            output: Output data.
            error: Error message if the step failed.
            duration_ms: Duration in milliseconds.
            metadata: Extra metadata.

        Returns:
            The created AgentStep.
        """
        step = AgentStep(
            kind=kind,
            name=name,
            input=input or {},
            output=output,
            error=error,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        trace.steps.append(step)
        return step

    def get_trace(
        self,
        trace: AgentTrace,
        *,
        output: Any = None,
        total_duration_ms: float = 0.0,
    ) -> AgentTrace:
        """Finalize a trace with output and timing.

        Args:
            trace: The trace to finalize.
            output: Final agent output.
            total_duration_ms: Total wall-clock time.

        Returns:
            The finalized trace.
        """
        if output is not None:
            trace.output = output
        if total_duration_ms > 0:
            trace.total_duration_ms = total_duration_ms
        return trace

    def run(self, input_text: str) -> AgentTrace:
        """Execute the agent and return a trace. Must be implemented by subclasses."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement run(). "
            "Use capture_tool_call() and capture_step() to build the trace."
        )


__all__ = ["BaseAdapter", "AgentAdapter", "AgentTrace", "AgentStep", "StepKind"]
