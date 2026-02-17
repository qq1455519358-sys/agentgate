# Core Concepts

## Scenario

A **Scenario** is an E2E behavioral test. It defines:

- **Input**: What to send to the agent
- **Expectations**: What the agent should (or shouldn't) do
- **Limits**: Resource constraints (timeout, max steps, max tokens)

```python
s = Scenario("Book a flight", input="Book SFO→NRT", timeout_seconds=30)
s.expect_tool_call("search_flights", before="book_flight")
s.expect_no_tool_call("cancel_booking")
s.expect_max_steps(10)
```

## Expectations

Expectations are assertions about agent **behavior**, not output text.

| Expectation | Tests |
|-------------|-------|
| `expect_tool_call("X")` | Tool X was called |
| `expect_no_tool_call("X")` | Tool X was NOT called (safety) |
| `expect_tool_order(["A", "B"])` | A called before B |
| `expect_no_error()` | No errors in trace |
| `expect_max_steps(N)` | Agent finished within N steps |
| `expect_output(contains="...")` | Output contains text |
| `expect_milestone("...", tool="X")` | Partial credit |
| `expect_llm_judge("criteria", judge_fn=fn)` | LLM evaluates trace |

## AgentTrace

The **trace** captures everything the agent did:

```python
trace = adapter.run("Book a flight")
print(trace.tool_calls)     # [AgentStep(name="search"), AgentStep(name="book")]
print(trace.tool_names)     # ["search", "book"]
print(trace.errors)         # []
print(trace.total_duration_ms)  # 3400.0
print(trace.metadata)       # {"token_usage": {"input": 500, "output": 200}}
```

## Adapters

Adapters bridge your agent framework to AgentGate:

- **`OpenAIAdapter`** — OpenAI/DeepSeek function-calling agents
- **`LangGraphAdapter`** — LangGraph apps
- **`CallableAgentAdapter`** — Any function `fn(input) → AgentTrace`
- **`MockAgent`** — Pre-recorded traces (zero API cost)

## TestSuite

A **TestSuite** runs multiple scenarios:

```python
suite = TestSuite("my-agent")
suite.add(scenario_1)
suite.add(scenario_2)
result = suite.run(adapter, runs=5, min_pass_rate=0.8)
```

## Results

```
✅ Suite: my-agent — 2/2 scenarios passed (100%)
  Consistency (τ-bench): pass^1=1.000, pass^2=1.000
✅ Scenario: Book a flight (3/3 expectations passed)
  ✅ expect_tool_call('search_flights')
  ✅ expect_tool_call('book_flight')
  ✅ expect_no_tool_call('cancel_booking')
```
