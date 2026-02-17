"""Multi-step E2E test with real LangGraph agent."""
import sys, time, os
from pathlib import Path

# Load API key
for line in Path(r"C:\Users\24638\AlphaAgent\.env").read_text().splitlines():
    if line.startswith("DEEPSEEK_API_KEY="):
        os.environ["OPENAI_API_KEY"] = line.split("=",1)[1].strip('"')
    elif line.startswith("DEEPSEEK_API_BASE="):
        os.environ["OPENAI_BASE_URL"] = line.split("=",1)[1].strip('"')

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.real_langgraph_agent import app
from agentgate.adapters.langgraph import LangGraphAdapter
from agentgate.discover import Discovery

adapter = LangGraphAdapter(app)
discovery = Discovery(adapter)

multi_step_inputs = [
    "Search for flights from SFO to NRT on March 15th, then book the cheapest one for John Smith",
    "Check booking BK001 and cancel it if it is confirmed",
    "I want to fly from SFO to NRT on March 15. What is the baggage policy for the airline with the cheapest flight?",
]

print("📹 Recording multi-step scenarios (3 inputs × 3 runs)...\n")
for inp in multi_step_inputs:
    print(f'  "{inp[:65]}..."')
    start = time.time()
    discovery.record(inp, runs=3, on_progress=lambda i,n: print(f"    Run {i}/{n}", end=" "))
    print(f"({time.time()-start:.1f}s)")

print("\n📊 Analyzing multi-step patterns...")
report = discovery.analyze()
print(report.summary())

print("\n🔧 Generating scenarios...")
scenarios = discovery.generate_scenarios(min_frequency=0.6)
print(f"Generated {len(scenarios)} scenarios")
discovery.save_scenarios("generated/airline_multistep.py", min_frequency=0.6)

print("\n🧪 Running E2E tests (2 runs each, 50% pass threshold)...")
suite = discovery.to_suite(min_frequency=0.6)
result = suite.run(adapter, runs=2, min_pass_rate=0.5)
print(result.summary())

if result.passed:
    print("🚀 DEPLOY: All passed")
else:
    failed = [r for r in result.results if not r.passed]
    print(f"🚫 BLOCKED: {len(failed)} scenario(s) failed")
