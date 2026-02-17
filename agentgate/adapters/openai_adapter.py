"""
AgentGate OpenAI Adapter — Run OpenAI function-calling agents and capture traces.

Works with the standard OpenAI chat completions API with tools.
Zero framework dependency — just openai SDK.

Usage:
    from openai import OpenAI
    from agentgate.adapters.openai_adapter import OpenAIAdapter

    client = OpenAI()
    tools = [{"type": "function", "function": {...}}]
    tool_fns = {"search": search_fn, "book": book_fn}

    adapter = OpenAIAdapter(client, tools=tools, tool_fns=tool_fns, model="gpt-4o")
    trace = adapter.run("Book a flight to Tokyo")
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional

from agentgate.types import AgentTrace, AgentStep, StepKind
from agentgate.adapters.base import AgentAdapter


class OpenAIAdapter(AgentAdapter):
    """Adapter for OpenAI function-calling agents.

    Runs a tool-use loop: sends messages → model responds with tool calls →
    execute tools → feed results back → repeat until model stops calling tools.

    Args:
        client: An ``openai.OpenAI`` instance.
        tools: List of tool definitions (OpenAI format).
        tool_fns: Dict mapping tool names to Python callables.
        model: Model name (default "gpt-4o").
        system_prompt: Optional system message.
        max_turns: Maximum tool-call rounds (default 10).
    """

    def __init__(
        self,
        client: Any,
        *,
        tools: list[dict],
        tool_fns: dict[str, Callable],
        model: str = "gpt-4o",
        system_prompt: str = "You are a helpful assistant.",
        max_turns: int = 10,
    ):
        self.client = client
        self.tools = tools
        self.tool_fns = tool_fns
        self.model = model
        self.system_prompt = system_prompt
        self.max_turns = max_turns

    def run(self, input_text: str) -> AgentTrace:
        trace = AgentTrace(input=input_text)
        start = time.time()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text},
        ]

        total_input_tokens = 0
        total_output_tokens = 0

        for turn in range(self.max_turns):
            # LLM call
            llm_start = time.time()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools if self.tools else None,
                )
            except Exception as e:
                trace.steps.append(AgentStep(
                    kind=StepKind.ERROR,
                    name="openai_api_error",
                    error=str(e),
                    duration_ms=(time.time() - llm_start) * 1000,
                ))
                break

            llm_duration = (time.time() - llm_start) * 1000

            # Track token usage
            if response.usage:
                total_input_tokens += response.usage.prompt_tokens or 0
                total_output_tokens += response.usage.completion_tokens or 0

            msg = response.choices[0].message

            # Record LLM step
            trace.steps.append(AgentStep(
                kind=StepKind.LLM_CALL,
                name=self.model,
                input={"turn": turn + 1},
                output=msg.content or "(tool calls)",
                duration_ms=llm_duration,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "input": response.usage.prompt_tokens if response.usage else 0,
                        "output": response.usage.completion_tokens if response.usage else 0,
                    },
                },
            ))

            # No tool calls → done
            if not msg.tool_calls:
                trace.output = msg.content
                messages.append({"role": "assistant", "content": msg.content})
                break

            # Process tool calls
            messages.append(msg.model_dump())

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {"raw": tool_call.function.arguments}

                # Execute the tool
                tool_start = time.time()
                tool_error = None
                tool_output = None

                if fn_name in self.tool_fns:
                    try:
                        tool_output = self.tool_fns[fn_name](**fn_args)
                        if not isinstance(tool_output, str):
                            tool_output = json.dumps(tool_output)
                    except Exception as e:
                        tool_error = f"{type(e).__name__}: {e}"
                        tool_output = tool_error
                else:
                    tool_error = f"Unknown tool: {fn_name}"
                    tool_output = tool_error

                tool_duration = (time.time() - tool_start) * 1000

                # Record tool step
                trace.steps.append(AgentStep(
                    kind=StepKind.TOOL_CALL,
                    name=fn_name,
                    input=fn_args,
                    output=tool_output,
                    error=tool_error,
                    duration_ms=tool_duration,
                ))

                # Feed result back
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_output),
                })

        trace.total_duration_ms = (time.time() - start) * 1000
        trace.metadata["token_usage"] = {
            "input": total_input_tokens,
            "output": total_output_tokens,
        }
        trace.metadata["model"] = self.model
        trace.metadata["turns"] = min(turn + 1, self.max_turns) if 'turn' in dir() else 0

        return trace


class AsyncOpenAIAdapter(AgentAdapter):
    """Async version of OpenAIAdapter for use with AsyncOpenAI.

    Usage:
        from openai import AsyncOpenAI
        from agentgate.adapters.openai_adapter import AsyncOpenAIAdapter

        client = AsyncOpenAI()
        adapter = AsyncOpenAIAdapter(client, tools=tools, tool_fns=tool_fns)

        # Use with async suite runner
        trace = await adapter.arun("Book a flight")
    """

    def __init__(
        self,
        client: Any,
        *,
        tools: list[dict],
        tool_fns: dict[str, Callable],
        model: str = "gpt-4o",
        system_prompt: str = "You are a helpful assistant.",
        max_turns: int = 10,
    ):
        self.client = client
        self.tools = tools
        self.tool_fns = tool_fns
        self.model = model
        self.system_prompt = system_prompt
        self.max_turns = max_turns

    async def arun(self, input_text: str) -> AgentTrace:
        """Run the agent asynchronously."""
        import asyncio
        trace = AgentTrace(input=input_text)
        start = time.time()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text},
        ]

        total_input_tokens = 0
        total_output_tokens = 0

        for turn in range(self.max_turns):
            llm_start = time.time()
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools if self.tools else None,
                )
            except Exception as e:
                trace.steps.append(AgentStep(
                    kind=StepKind.ERROR, name="openai_api_error",
                    error=str(e), duration_ms=(time.time() - llm_start) * 1000,
                ))
                break

            llm_duration = (time.time() - llm_start) * 1000

            if response.usage:
                total_input_tokens += response.usage.prompt_tokens or 0
                total_output_tokens += response.usage.completion_tokens or 0

            msg = response.choices[0].message
            trace.steps.append(AgentStep(
                kind=StepKind.LLM_CALL, name=self.model,
                input={"turn": turn + 1},
                output=msg.content or "(tool calls)",
                duration_ms=llm_duration,
            ))

            if not msg.tool_calls:
                trace.output = msg.content
                break

            messages.append(msg.model_dump())

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {"raw": tool_call.function.arguments}

                tool_start = time.time()
                tool_error = None
                tool_output = None

                if fn_name in self.tool_fns:
                    try:
                        result = self.tool_fns[fn_name](**fn_args)
                        # Support async tool functions
                        if asyncio.iscoroutine(result):
                            result = await result
                        tool_output = result if isinstance(result, str) else json.dumps(result)
                    except Exception as e:
                        tool_error = f"{type(e).__name__}: {e}"
                        tool_output = tool_error
                else:
                    tool_error = f"Unknown tool: {fn_name}"
                    tool_output = tool_error

                trace.steps.append(AgentStep(
                    kind=StepKind.TOOL_CALL, name=fn_name,
                    input=fn_args, output=tool_output, error=tool_error,
                    duration_ms=(time.time() - tool_start) * 1000,
                ))

                messages.append({
                    "role": "tool", "tool_call_id": tool_call.id,
                    "content": str(tool_output),
                })

        trace.total_duration_ms = (time.time() - start) * 1000
        trace.metadata["token_usage"] = {"input": total_input_tokens, "output": total_output_tokens}
        trace.metadata["model"] = self.model
        return trace

    def run(self, input_text: str) -> AgentTrace:
        """Sync wrapper — runs the async version in an event loop."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in async context — create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self.arun(input_text)).result()
        return asyncio.run(self.arun(input_text))


__all__ = ["OpenAIAdapter", "AsyncOpenAIAdapter"]
