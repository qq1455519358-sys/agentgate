# Contributing to AgentGate

Thanks for your interest! Here's how to get started.

## Setup

```bash
git clone https://github.com/qq1455519358-sys/agentgate.git
cd agentgate
pip install -e ".[dev]"
pytest tests/ -q
```

## Running Tests

```bash
# All mock tests (fast, no API keys needed)
pytest tests/ --ignore=tests/test_real_agent.py -q

# Real API tests (requires DEEPSEEK_API_KEY)
pytest tests/test_real_agent.py -v
```

## Code Style

We use `ruff` for linting:

```bash
ruff check agentgate/ tests/
ruff format agentgate/ tests/
```

## What We Need Help With

- **New adapters** — Bedrock, Vertex AI, LlamaIndex, Haystack
- **New adversarial scenarios** — More OWASP patterns, domain-specific attacks
- **Benchmarks** — Run AgentGate against public agent benchmarks
- **Documentation** — Guides, tutorials, real-world examples
- **Bug reports** — If your agent breaks AgentGate, we want to know

## PR Process

1. Fork the repo
2. Create a branch (`git checkout -b feat/my-feature`)
3. Write tests for your changes
4. Run `pytest tests/ -q` — all must pass
5. Submit PR with a clear description

## Design Principles

- **Zero dependencies** in core — don't add `numpy` or `torch` to the base package
- **Every metric cites its paper** — no made-up metrics
- **Mock-first** — all tests should work without API keys
- **Backward compatible** — don't break existing tests

## License

By contributing, you agree your contributions are licensed under Apache 2.0.
