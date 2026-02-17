# Installation

## Basic Install

```bash
pip install agentgate
```

Zero runtime dependencies. Works with Python 3.10+.

## Optional Extras

```bash
# LLM-as-Judge support (requires OpenAI SDK)
pip install agentgate[llm]

# LangGraph adapter
pip install agentgate[langgraph]

# CLI tools
pip install agentgate[cli]

# Everything
pip install agentgate[all]
```

## From Source

```bash
git clone https://github.com/qq1455519358-sys/agentgate.git
cd agentgate
pip install -e ".[dev]"
pytest tests/ -q
```

## Verify

```python
import agentgate
print(agentgate.__version__)  # 0.3.0
print(len(agentgate.__all__))  # 108 exports
```
