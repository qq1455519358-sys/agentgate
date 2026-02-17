"""
AgentGate Generic Adapters — Wrap any agent framework.

Provides adapters for common patterns:
- FunctionAdapter: wrap any function that returns structured output
- DictAdapter: wrap functions that return dicts with tool_calls
- StdoutAdapter: capture tool calls from stdout/logs

Usage:
    from agentgate.adapters.generic import FunctionAdapter

    def my_agent(query):
        # ... your agent logic ...
        return {
            "output": "Done!",
            "tool_calls": [
                {"name": "search", "args": {"q": "flights"}, "result": "found 3"},
                {"name": "book", "args": {"id": "FL1"}, "result": "confirmed"},
            ]
        }

    adapter = FunctionAdapter(my_agent)
    trace = adapter.run("Book a flight")
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from agentgate.types import AgentTrace, AgentStep, StepKind
from agentgate.adapters.base import AgentAdapter


class FunctionAdapter(AgentAdapter):
    """Wrap any function that returns a dict with output and tool_calls.

    Expected return format:
        {
            "output": "final text",
            "tool_calls": [
                {"name": "tool", "args": {...}, "result": "...", "error": None},
            ],
            "metadata": {...}  # optional
        }

    Also accepts:
    - A plain string (becomes output with no tool calls)
    - An AgentTrace directly
    - A dict with just "output" (no tool calls)
    """

    def __init__(self, fn: Callable, *, name: str = "agent"):
        self.fn = fn
        self.name = name

    def run(self, input_text: str) -> AgentTrace:
        trace = AgentTrace(input=input_text)
        start = time.time()

        try:
            result = self.fn(input_text)
        except Exception as e:
            trace.steps.append(AgentStep(
                kind=StepKind.ERROR, name="agent_error",
                error=f"{type(e).__name__}: {e}",
            ))
            trace.total_duration_ms = (time.time() - start) * 1000
            return trace

        if isinstance(result, AgentTrace):
            result.total_duration_ms = (time.time() - start) * 1000
            return result

        if isinstance(result, str):
            trace.output = result
            trace.total_duration_ms = (time.time() - start) * 1000
            return trace

        if isinstance(result, dict):
            trace.output = result.get("output")
            trace.metadata.update(result.get("metadata", {}))

            for tc in result.get("tool_calls", []):
                trace.steps.append(AgentStep(
                    kind=StepKind.TOOL_CALL,
                    name=tc.get("name", "unknown"),
                    input=tc.get("args", tc.get("input", {})),
                    output=tc.get("result", tc.get("output")),
                    error=tc.get("error"),
                ))

            for step in result.get("steps", []):
                kind_str = step.get("kind", "tool_call")
                kind = StepKind(kind_str) if kind_str in [k.value for k in StepKind] else StepKind.TOOL_CALL
                trace.steps.append(AgentStep(
                    kind=kind,
                    name=step.get("name", "unknown"),
                    input=step.get("input", {}),
                    output=step.get("output"),
                    error=step.get("error"),
                ))

        trace.total_duration_ms = (time.time() - start) * 1000
        return trace


class CrewAIAdapter(AgentAdapter):
    """Adapter for CrewAI agents.

    Wraps a CrewAI Crew and captures tool calls from the execution.

    Usage:
        from crewai import Crew
        from agentgate.adapters.generic import CrewAIAdapter

        crew = Crew(agents=[...], tasks=[...])
        adapter = CrewAIAdapter(crew)
        trace = adapter.run("Research AI trends")
    """

    def __init__(self, crew: Any, *, verbose: bool = False):
        self.crew = crew
        self.verbose = verbose

    def run(self, input_text: str) -> AgentTrace:
        trace = AgentTrace(input=input_text)
        start = time.time()

        try:
            # CrewAI's kickoff method
            result = self.crew.kickoff(inputs={"input": input_text})

            # Extract output
            if hasattr(result, 'raw'):
                trace.output = result.raw
            elif isinstance(result, str):
                trace.output = result
            else:
                trace.output = str(result)

            # Extract tasks results as steps
            if hasattr(result, 'tasks_output'):
                for i, task_output in enumerate(result.tasks_output):
                    trace.steps.append(AgentStep(
                        kind=StepKind.TOOL_CALL,
                        name=f"task_{i}" if not hasattr(task_output, 'description')
                             else task_output.description[:50],
                        output=task_output.raw if hasattr(task_output, 'raw') else str(task_output),
                    ))

            # Extract token usage if available
            if hasattr(result, 'token_usage'):
                trace.metadata["token_usage"] = {
                    "input": getattr(result.token_usage, 'prompt_tokens', 0),
                    "output": getattr(result.token_usage, 'completion_tokens', 0),
                }

        except Exception as e:
            trace.steps.append(AgentStep(
                kind=StepKind.ERROR, name="crewai_error",
                error=f"{type(e).__name__}: {e}",
            ))

        trace.total_duration_ms = (time.time() - start) * 1000
        return trace


class AutoGenAdapter(AgentAdapter):
    """Adapter for Microsoft AutoGen agents.

    Wraps an AutoGen ConversableAgent or GroupChat.

    Usage:
        from autogen import ConversableAgent
        from agentgate.adapters.generic import AutoGenAdapter

        agent = ConversableAgent(...)
        adapter = AutoGenAdapter(agent)
        trace = adapter.run("Analyze this data")
    """

    def __init__(self, agent: Any, *, human_input_mode: str = "NEVER"):
        self.agent = agent
        self.human_input_mode = human_input_mode

    def run(self, input_text: str) -> AgentTrace:
        trace = AgentTrace(input=input_text)
        start = time.time()

        try:
            # AutoGen's initiate_chat
            result = self.agent.initiate_chat(
                self.agent,
                message=input_text,
                max_turns=10,
            )

            # Extract chat history as steps
            if hasattr(result, 'chat_history'):
                for msg in result.chat_history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")

                    if "tool_calls" in msg:
                        for tc in msg["tool_calls"]:
                            trace.steps.append(AgentStep(
                                kind=StepKind.TOOL_CALL,
                                name=tc.get("function", {}).get("name", "unknown"),
                                input=tc.get("function", {}).get("arguments", {}),
                            ))
                    elif role == "assistant":
                        trace.steps.append(AgentStep(
                            kind=StepKind.LLM_CALL,
                            name="autogen",
                            output=content,
                        ))

                # Last assistant message is the output
                for msg in reversed(result.chat_history):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        trace.output = msg["content"]
                        break

            if hasattr(result, 'cost'):
                trace.metadata["cost"] = result.cost

        except Exception as e:
            trace.steps.append(AgentStep(
                kind=StepKind.ERROR, name="autogen_error",
                error=f"{type(e).__name__}: {e}",
            ))

        trace.total_duration_ms = (time.time() - start) * 1000
        return trace


__all__ = ["FunctionAdapter", "CrewAIAdapter", "AutoGenAdapter"]
