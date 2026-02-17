"""
AgentGate Anthropic Adapter — Run Claude tool-use agents and capture traces.

Works with the Anthropic Python SDK's tool use API.

Usage:
    from anthropic import Anthropic
    from agentgate.adapters.anthropic_adapter import AnthropicAdapter

    client = Anthropic()
    adapter = AnthropicAdapter(
        client,
        tools=[{"name": "search", "description": "...", "input_schema": {...}}],
        tool_fns={"search": search_fn},
        model="claude-sonnet-4-20250514",
    )
    trace = adapter.run("Book a flight to Tokyo")
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional

from agentgate.types import AgentTrace, AgentStep, StepKind
from agentgate.adapters.base import AgentAdapter


class AnthropicAdapter(AgentAdapter):
    """Adapter for Anthropic Claude tool-use agents.

    Runs a tool-use loop: sends messages → Claude responds with tool_use blocks →
    execute tools → feed tool_result back → repeat until Claude sends text only.

    Args:
        client: An ``anthropic.Anthropic`` instance.
        tools: List of tool definitions (Anthropic format).
        tool_fns: Dict mapping tool names to Python callables.
        model: Model name (default "claude-sonnet-4-20250514").
        system: Optional system message.
        max_turns: Maximum tool-call rounds (default 10).
        max_tokens: Max tokens per response (default 4096).
    """

    def __init__(
        self,
        client: Any,
        *,
        tools: list[dict],
        tool_fns: dict[str, Callable],
        model: str = "claude-sonnet-4-20250514",
        system: str = "You are a helpful assistant.",
        max_turns: int = 10,
        max_tokens: int = 4096,
    ):
        self.client = client
        self.tools = tools
        self.tool_fns = tool_fns
        self.model = model
        self.system = system
        self.max_turns = max_turns
        self.max_tokens = max_tokens

    def run(self, input_text: str) -> AgentTrace:
        trace = AgentTrace(input=input_text)
        start = time.time()

        messages = [{"role": "user", "content": input_text}]

        total_input_tokens = 0
        total_output_tokens = 0

        for turn in range(self.max_turns):
            llm_start = time.time()
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=self.system,
                    messages=messages,
                    tools=self.tools if self.tools else None,
                )
            except Exception as e:
                trace.steps.append(AgentStep(
                    kind=StepKind.ERROR, name="anthropic_api_error",
                    error=str(e), duration_ms=(time.time() - llm_start) * 1000,
                ))
                break

            llm_duration = (time.time() - llm_start) * 1000

            # Track tokens
            if hasattr(response, 'usage') and response.usage:
                total_input_tokens += getattr(response.usage, 'input_tokens', 0)
                total_output_tokens += getattr(response.usage, 'output_tokens', 0)

            # Extract text and tool_use blocks
            text_parts = []
            tool_use_blocks = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_use_blocks.append(block)

            # Record LLM step
            trace.steps.append(AgentStep(
                kind=StepKind.LLM_CALL,
                name=self.model,
                input={"turn": turn + 1},
                output=" ".join(text_parts) if text_parts else "(tool calls)",
                duration_ms=llm_duration,
                metadata={
                    "stop_reason": response.stop_reason,
                    "usage": {
                        "input": getattr(response.usage, 'input_tokens', 0) if response.usage else 0,
                        "output": getattr(response.usage, 'output_tokens', 0) if response.usage else 0,
                    },
                },
            ))

            # No tool use → done
            if response.stop_reason == "end_turn" or not tool_use_blocks:
                trace.output = " ".join(text_parts) if text_parts else None
                break

            # Process tool calls
            # Add assistant message with all content blocks
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tool_block in tool_use_blocks:
                fn_name = tool_block.name
                fn_args = tool_block.input if isinstance(tool_block.input, dict) else {}

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

                trace.steps.append(AgentStep(
                    kind=StepKind.TOOL_CALL,
                    name=fn_name,
                    input=fn_args,
                    output=tool_output,
                    error=tool_error,
                    duration_ms=tool_duration,
                ))

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": str(tool_output),
                    **({"is_error": True} if tool_error else {}),
                })

            messages.append({"role": "user", "content": tool_results})

        trace.total_duration_ms = (time.time() - start) * 1000
        trace.metadata["token_usage"] = {
            "input": total_input_tokens,
            "output": total_output_tokens,
        }
        trace.metadata["model"] = self.model
        return trace


__all__ = ["AnthropicAdapter"]
