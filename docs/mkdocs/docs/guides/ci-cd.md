# CI/CD Integration

## GitHub Actions

```yaml
name: Agent E2E Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.12"
    - run: pip install agentgate[all]
    - run: pytest tests/ -q

  # Optional: real agent tests (costs money)
  integration:
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
    - run: pip install agentgate[all]
    - run: pytest tests/test_real_agent.py -v
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## pytest Markers

```python
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "real_agent: tests that call real LLM APIs (costs money)",
]
```

```python
@pytest.mark.real_agent
def test_booking_flow(real_adapter):
    ...
```

```bash
# Run only mock tests (fast, free)
pytest tests/ --ignore=tests/test_real_agent.py

# Run everything including real API tests
pytest tests/ -m "real_agent or not real_agent"
```

## Regression Detection

```python
from agentgate import diff_results

baseline = suite.run(agent_v1)
current = suite.run(agent_v2)
report = diff_results(baseline, current)
print(report)

# Fail CI on regressions
assert "REGRESSION" not in report, report
```

## Capability Graduation

```python
from agentgate import EvalSuiteManager

mgr = EvalSuiteManager()
mgr.add_regression("core_booking", booking_scenario)
mgr.add_capability("multilingual", new_scenario)

result = mgr.run(agent, trials=5)
# Regression must pass. Capability is aspirational.
# When capability hits 95%+, it graduates to regression.
mgr.graduate(threshold=0.95)
```
