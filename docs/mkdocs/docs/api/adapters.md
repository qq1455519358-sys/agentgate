# Adapters API

## OpenAIAdapter

Works with OpenAI, DeepSeek, Azure OpenAI, or any OpenAI-compatible API.

```python
from openai import OpenAI
from agentgate.adapters.openai_adapter import OpenAIAdapter

adapter = OpenAIAdapter(
    client=OpenAI(),
    tools=[...],              # OpenAI tool definitions
    tool_fns={"name": fn},    # name → callable
    model="gpt-4o",
    system_prompt="You are...",
    max_turns=10,
)

trace = adapter.run("Query")
```

### Token Tracking

```python
trace.metadata["token_usage"]  # {"input": 500, "output": 200}
trace.metadata["model"]        # "gpt-4o"
trace.metadata["turns"]        # 3
```

## AsyncOpenAIAdapter

```python
from openai import AsyncOpenAI
from agentgate.adapters.openai_adapter import AsyncOpenAIAdapter

adapter = AsyncOpenAIAdapter(
    client=AsyncOpenAI(),
    tools=[...],
    tool_fns={"name": fn},  # supports async tool functions too
)

trace = await adapter.arun("Query")  # async
trace = adapter.run("Query")          # sync fallback
```

## LangGraphAdapter

```python
from agentgate.adapters.langgraph import LangGraphAdapter

adapter = LangGraphAdapter(
    graph=your_compiled_graph,
    input_key="messages",
    config={"configurable": {"thread_id": "1"}},
)
```

## CallableAgentAdapter

For any function:

```python
from agentgate import CallableAgentAdapter, AgentTrace

# Simple: function returns AgentTrace
def my_agent(input_text: str) -> AgentTrace:
    ...

adapter = CallableAgentAdapter(my_agent)

# With trace extractor
def my_agent(input_text: str) -> dict:
    return {"result": "...", "tool_calls": [...]}

def extract_trace(result: dict) -> AgentTrace:
    trace = AgentTrace(input="")
    # ... build trace from result ...
    return trace

adapter = CallableAgentAdapter(my_agent, trace_extractor=extract_trace)
```

## Writing Custom Adapters

```python
from agentgate.adapters.base import AgentAdapter
from agentgate.types import AgentTrace, AgentStep, StepKind

class MyFrameworkAdapter(AgentAdapter):
    def __init__(self, agent):
        self.agent = agent

    def run(self, input_text: str) -> AgentTrace:
        trace = AgentTrace(input=input_text)

        # Run your agent and capture steps
        result = self.agent.invoke(input_text)
        for action in result.actions:
            trace.steps.append(AgentStep(
                kind=StepKind.TOOL_CALL,
                name=action.tool,
                input=action.args,
                output=action.result,
            ))

        trace.output = result.final_answer
        return trace
```
