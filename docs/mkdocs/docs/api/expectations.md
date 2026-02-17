# Expectations API

All expectations live in `agentgate.expectations` and are also available from the top-level `agentgate` import.

## Tool Expectations

### ExpectToolCall

```python
s.expect_tool_call("search",
    order=1,              # absolute position (1-indexed)
    before="book",        # must come before "book"
    after="login",        # must come after "login"
    with_args={"q": "X"}, # argument matching
    times=2,              # exact count
    min_times=1,          # at least N
    max_times=3,          # at most N
)
```

### ExpectNoToolCall

```python
s.expect_no_tool_call("delete_user")
```

### ExpectToolOrder

```python
s.expect_tool_order(["search", "select", "book"])
# Checks subsequence — other tools may be interleaved
```

## State Expectations

```python
s.expect_state("booking_status", value="confirmed", within_steps=5)
```

## Output Expectations

```python
s.expect_output(
    contains="confirmed",
    matches=r"BK\d+",
    not_contains="error",
)
```

## Resource Limits

```python
s.expect_max_steps(10)
s.expect_max_duration(5000)   # milliseconds
s.expect_max_tokens(10000)
```

## Behavioral Quality

```python
# Side effects (AgentRewardBench)
s.expect_no_side_effects(
    allowed_tools=["search", "book"],
    mutating_tools=["cancel", "delete"],
)

# Repetition detection
s.expect_no_repetition(max_rate=0.0)
```

## Milestone (Partial Credit)

```python
s.expect_milestone("Search done", tool="search", weight=1)
s.expect_milestone("Booked", tool="book", weight=2)
# score = weighted milestone achievement
```

## LLM Judge

```python
def my_judge(criteria, trace_text):
    # Call any LLM
    return (True, "Agent was professional")

s.expect_llm_judge("Agent is empathetic", judge_fn=my_judge)
```

## Custom Expectations

```python
from agentgate.expectations import Expectation
from agentgate.types import AgentTrace, ExpectationResult

class ExpectCustom(Expectation):
    def check(self, trace: AgentTrace) -> ExpectationResult:
        if some_condition(trace):
            return ExpectationResult(True, "custom_check")
        return ExpectationResult(False, "custom_check", "Failed because...")

s.expectations.append(ExpectCustom())
```
