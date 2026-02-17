# Core API

## Scenario

```python
from agentgate import Scenario

s = Scenario(
    name="Book a flight",
    input="Book SFO→NRT",
    timeout_seconds=30,
    max_steps=10,
    max_llm_calls=5,
)
```

### Methods

| Method | Description |
|--------|-------------|
| `expect_tool_call(name, **kwargs)` | Assert tool is called |
| `expect_no_tool_call(name)` | Assert tool is NOT called |
| `expect_tool_order([...])` | Assert tool subsequence |
| `expect_state(key, value=...)` | Assert state reached |
| `expect_no_error()` | Assert no errors |
| `expect_output(contains=..., matches=...)` | Assert output content |
| `expect_max_steps(n)` | Assert step limit |
| `expect_max_duration(ms)` | Assert time limit |
| `expect_max_tokens(n)` | Assert token budget |
| `expect_no_side_effects(allowed, mutating)` | Assert no side effects |
| `expect_no_repetition(max_rate)` | Assert no loops |
| `expect_milestone(name, tool=..., weight=...)` | Partial credit |
| `expect_llm_judge(criteria, judge_fn=...)` | LLM evaluation |
| `expect_policy(name, forbidden_tools=...)` | Policy bundle |
| `on_tool_failure(name, expect=...)` | Recovery behavior |
| `check(trace)` | Run expectations → ScenarioResult |

## TestSuite

```python
from agentgate import TestSuite

suite = TestSuite("my-tests")
suite.add(scenario)
result = suite.run(
    adapter,
    runs=5,
    min_pass_rate=0.8,
    timeout_seconds=300,
    scenario_timeout=60,
)
```

## AgentTrace

```python
from agentgate import AgentTrace, AgentStep, StepKind

trace = AgentTrace(input="query")
trace.steps.append(AgentStep(
    kind=StepKind.TOOL_CALL,
    name="search",
    input={"q": "flights"},
    output={"results": [...]},
))

# Properties
trace.tool_calls      # list[AgentStep] — TOOL_CALL steps only
trace.tool_names      # list[str] — tool names in order
trace.errors          # list[AgentStep] — ERROR steps
trace.state_changes   # list[AgentStep] — STATE_CHANGE steps
trace.get_state(key)  # last value of a state key
```

## StepKind

| Kind | Description |
|------|-------------|
| `TOOL_CALL` | Tool/function call |
| `LLM_CALL` | LLM inference |
| `STATE_CHANGE` | Environment state change |
| `HUMAN_HANDOFF` | Escalation to human |
| `ERROR` | Error occurred |

## MockAgent

```python
from agentgate import MockAgent, AgentTrace, AgentStep, StepKind

mock = MockAgent()
mock.add_trace("query pattern", AgentTrace(input="...", steps=[...]))

# From recorded traces
mock = MockAgent.from_traces("traces/")
```
