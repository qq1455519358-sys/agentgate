"""
AgentGate adapter for LangGraph agents.

Hooks into LangGraph's streaming API to capture the full execution trace
including tool calls, state changes, and LLM node transitions.

Usage:
    from agentgate.adapters.langgraph import LangGraphAdapter

    adapter = LangGraphAdapter(my_langgraph_app)
    trace = adapter.run("Book a flight to Tokyo")

    # Or with TestSuite
    suite.run(adapter)
"""

from __future__ import annotations

import time
from typing import Any, Optional

from agentgate.adapters.base import BaseAdapter
from agentgate.scenario import AgentTrace, AgentStep, StepKind


class LangGraphAdapter(BaseAdapter):
    """
    Adapter for LangGraph CompiledGraph agents.

    Hooks into LangGraph's streaming API to capture the full execution trace
    including tool calls, state changes, and node transitions.
    """

    def __init__(self, graph: Any, *, config: Optional[dict] = None,
                 input_key: str = "messages",
                 output_parser: Optional[callable] = None):
        """
        Args:
            graph: A LangGraph CompiledGraph instance.
            config: Optional LangGraph config dict.
            input_key: Key for input in the graph state (default: "messages").
            output_parser: Optional function to extract final output from state.
        """
        self.graph = graph
        self.config = config or {}
        self.input_key = input_key
        self.output_parser = output_parser

    def run(self, input_text: str) -> AgentTrace:
        """Run the LangGraph agent and capture a complete trace."""
        start = time.time()
        trace = AgentTrace(input=input_text)

        try:
            from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
        except ImportError:
            raise ImportError(
                "LangGraph adapter requires langchain-core. "
                "Install with: pip install langchain-core langgraph"
            )

        input_state = {self.input_key: [HumanMessage(content=input_text)]}

        final_state = None
        try:
            for event in self.graph.stream(input_state, config=self.config,
                                            stream_mode="updates"):
                for node_name, node_output in event.items():
                    if self._is_tool_node(node_name, node_output):
                        self._capture_tool_steps(trace, node_name, node_output)
                    elif self._is_llm_node(node_output):
                        self._capture_llm_step(trace, node_name, node_output)
                    else:
                        self.capture_step(
                            trace,
                            StepKind.STATE_CHANGE,
                            node_name,
                            output=self._safe_serialize(node_output),
                        )
                    final_state = node_output

        except Exception as e:
            self.capture_step(
                trace,
                StepKind.ERROR,
                "execution_error",
                error=str(e),
            )

        elapsed = (time.time() - start) * 1000
        output = self._extract_output(final_state) if final_state else None
        return self.get_trace(trace, output=output, total_duration_ms=elapsed)

    def _is_tool_node(self, node_name: str, output: Any) -> bool:
        if "tool" in node_name.lower():
            return True
        if isinstance(output, dict) and self.input_key in output:
            msgs = output[self.input_key]
            if msgs:
                try:
                    from langchain_core.messages import ToolMessage
                    return any(isinstance(m, ToolMessage) for m in msgs)
                except ImportError:
                    pass
        return False

    def _is_llm_node(self, output: Any) -> bool:
        if isinstance(output, dict) and self.input_key in output:
            msgs = output[self.input_key]
            if msgs:
                try:
                    from langchain_core.messages import AIMessage
                    return any(isinstance(m, AIMessage) for m in msgs)
                except ImportError:
                    pass
        return False

    def _capture_tool_steps(self, trace: AgentTrace, node_name: str, output: Any) -> None:
        """Extract and capture tool call steps from a tool node output."""
        captured = False
        if isinstance(output, dict) and self.input_key in output:
            for msg in output[self.input_key]:
                try:
                    from langchain_core.messages import ToolMessage
                    if isinstance(msg, ToolMessage):
                        error = None
                        if hasattr(msg, 'status') and msg.status == 'error':
                            error = str(msg.content)
                        self.capture_tool_call(
                            trace,
                            msg.name or node_name,
                            input={"tool_call_id": msg.tool_call_id} if hasattr(msg, 'tool_call_id') else {},
                            output=msg.content,
                            error=error,
                        )
                        captured = True
                except ImportError:
                    pass
        if not captured:
            self.capture_tool_call(
                trace,
                node_name,
                output=self._safe_serialize(output),
            )

    def _capture_llm_step(self, trace: AgentTrace, node_name: str, output: Any) -> None:
        """Extract and capture an LLM call step."""
        tool_calls = []
        content = ""
        if isinstance(output, dict) and self.input_key in output:
            for msg in output[self.input_key]:
                try:
                    from langchain_core.messages import AIMessage
                    if isinstance(msg, AIMessage):
                        content = msg.content
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            tool_calls = [
                                {"name": tc["name"], "args": tc.get("args", {})}
                                for tc in msg.tool_calls
                            ]
                except ImportError:
                    pass

        self.capture_step(
            trace,
            StepKind.LLM_CALL,
            node_name,
            output=content,
            metadata={"tool_calls_requested": tool_calls} if tool_calls else {},
        )

    def _extract_output(self, final_state: Any) -> Any:
        if self.output_parser:
            return self.output_parser(final_state)
        if isinstance(final_state, dict) and self.input_key in final_state:
            msgs = final_state[self.input_key]
            if msgs:
                last = msgs[-1]
                if hasattr(last, 'content'):
                    return last.content
        return str(final_state)

    def _safe_serialize(self, obj: Any) -> Any:
        try:
            import json
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)
