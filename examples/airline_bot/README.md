# Airline Bot Example

E2E behavioral tests for a LangGraph-based airline booking agent.

## Run

```
pip install agentgate
cd examples/airline_bot
pytest -v              # mock mode, no API keys
```

For real agent tests:

```
pip install agentgate[langgraph]
export DEEPSEEK_API_KEY=...
python test_booking.py
python test_adversarial.py
```
