# Metrics API

## Trajectory Metrics

```python
from agentgate import node_f1, edge_f1, tool_edit_distance

trace = adapter.run("Book a flight")
expected = ["search_flights", "book_flight"]

node_f1(trace, expected)            # 1.0 — all expected tools called
edge_f1(trace, expected)            # 1.0 — correct ordering
tool_edit_distance(trace, expected)  # 0.0 — exact match
```

## Side Effects & Repetition

```python
from agentgate import side_effect_rate, repetition_rate

side_effect_rate(trace, allowed=["search", "book"], mutating=["cancel"])
repetition_rate(trace)  # fraction of consecutive duplicate steps
```

## SABER Deviation

```python
from agentgate import decisive_deviation_score

score = decisive_deviation_score(
    trace, expected_tools=["search", "book"],
    mutating_tools=["book", "cancel"],
)
# Mutating deviations weighted 5x
```

## Trajectory Analysis

```python
from agentgate import step_credit, critical_steps, trajectory_redundancy, trajectory_efficiency

credits = step_credit(trace, expected)          # [1.0, -0.5, 0.5, 1.0]
critical = critical_steps(trace, expected)       # [2, 5]
redundancy = trajectory_redundancy(trace, expected)  # 0.1
efficiency = trajectory_efficiency(trace, expected)  # 0.85
```

## Confidence Calibration

```python
from agentgate import trajectory_confidence, detect_handoff_errors

conf = trajectory_confidence(trace)  # 0.0-1.0
errors = detect_handoff_errors(trace)
# [HandoffError("data_gap", step=3, detail="empty input")]
```

## Reproducibility

```python
from agentgate import variance_report, reproducibility_score

results = suite.run(agent, runs=10)
report = variance_report(results.results)
# VarianceReport(consistency=0.8, pass_rate=0.8)

score = reproducibility_score(results.results)  # 0.0-1.0
```

## Cost

```python
from agentgate import token_cost, cost_score, pareto_frontier, AgentResult

cost = token_cost(trace)  # $0.0045
efficiency = cost_score(trace, max_tokens=10000)  # 0.85

results = [
    AgentResult("GPT-4o", success_rate=0.92, cost=45.0),
    AgentResult("DeepSeek", success_rate=0.78, cost=1.5),
]
frontier = pareto_frontier(results)
```

## Memory Evaluation

```python
from agentgate import MemoryProbe, memory_consistency_suite

probes = [MemoryProbe(
    fact="User birthday is March 5",
    queries=["When is my birthday?"],
    expected_answers=["march 5"],
    competency="AR",
)]
suite = memory_consistency_suite(probes)
```

## Multi-Agent

```python
from agentgate import MultiAgentTrace, collaboration_quality, coordination_efficiency

mat = MultiAgentTrace(
    traces={"planner": t1, "executor": t2},
    coordination_protocol="star",
    total_messages=5,
)
quality = collaboration_quality(mat)       # 0.0-1.0
efficiency = coordination_efficiency(mat)  # 0.0-1.0
```

## Silent Failure Detection

```python
from agentgate import full_silent_failure_scan

report = full_silent_failure_scan(
    trace,
    expected_tools=["search", "book"],
    max_tool_repeats=2,
    required_output_keywords=["confirmation"],
)
print(report.failure_types)  # ['drift', 'cycles']
```
