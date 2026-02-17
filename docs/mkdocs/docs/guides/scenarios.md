# Writing Scenarios

## Basic Scenario

```python
s = Scenario("Check booking", input="Check status of booking BK001")
s.expect_tool_call("check_booking")
s.expect_no_tool_call("cancel_booking")
```

## Tool Ordering

```python
# Relative ordering (recommended — resilient to non-determinism)
s.expect_tool_call("search", before="book")
s.expect_tool_call("book", after="search")

# Subsequence order
s.expect_tool_order(["search", "select", "book"])

# Exact position (fragile — avoid unless necessary)
s.expect_tool_call("search", order=1)
```

## Call Counts

```python
s.expect_tool_call("retry_api", min_times=1, max_times=3)
s.expect_tool_call("search", times=1)  # exactly once
```

## Argument Matching

```python
s.expect_tool_call("search_flights", with_args={"destination": "Tokyo"})
```

## Error Recovery

```python
s.on_tool_failure("search_flights", expect="retry")
s.on_tool_failure("payment", expect="human_handoff")
```

## Milestones (Partial Credit)

```python
s.expect_milestone("Found flights", tool="search_flights", weight=1)
s.expect_milestone("Selected option", tool="select_flight", weight=1)
s.expect_milestone("Booking confirmed", tool="book_flight", weight=2)

result = s.check(trace)
print(result.score)  # 0.25 if only search completed
```

## Policy Adherence

```python
s.expect_policy("PII Protection",
    forbidden_tools=["export_raw_data"],
    forbidden_outputs=["SSN", "credit card"],
    required_tools=["redact_pii"],
)
```

## Resource Limits

```python
s = Scenario("Fast check", input="...", timeout_seconds=10, max_steps=5)
s.expect_max_tokens(10000)
s.expect_max_duration(5000)  # 5 seconds
```

## Composing Expectations

Mix and match freely:

```python
s = Scenario("Full booking workflow", input="Book SFO→NRT next Friday")
s.expect_tool_call("search_flights", before="book_flight")
s.expect_tool_call("book_flight", with_args={"destination": "NRT"})
s.expect_no_tool_call("cancel_booking")
s.expect_no_tool_call("delete_user")
s.expect_no_side_effects(
    allowed_tools=["search_flights", "book_flight"],
    mutating_tools=["cancel_booking"],
)
s.expect_no_repetition()
s.expect_no_error()
s.expect_max_steps(10)
s.expect_output(contains="confirmed")
s.expect_milestone("Search done", tool="search_flights", weight=1)
s.expect_milestone("Booked", tool="book_flight", weight=2)
```
