"""
🔥 AgentGate + Real LangGraph Agent — Complete Discovery → Test Workflow

This is NOT a mock. This runs a real LLM-powered LangGraph agent with tool calling,
then uses AgentGate Discovery to:
1. Record the agent's behavior across multiple runs
2. Extract behavioral patterns
3. Auto-generate E2E scenarios
4. Run those scenarios as tests

Uses DeepSeek API (OpenAI-compatible) to keep costs minimal (~$0.001/run).
"""

import os
import sys
import time
from typing import Annotated, Literal
from pathlib import Path

# ============================================================
# Setup API Key
# ============================================================

def load_api_key():
    """Load DeepSeek API key from AlphaAgent .env"""
    env_path = Path(r"C:\Users\24638\AlphaAgent\.env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                key = line.split("=", 1)[1].strip('"').strip("'")
                os.environ["OPENAI_API_KEY"] = key
            elif line.startswith("DEEPSEEK_API_BASE="):
                base = line.split("=", 1)[1].strip('"').strip("'")
                os.environ["OPENAI_BASE_URL"] = base

load_api_key()

# ============================================================
# Build a real LangGraph ReAct agent with tools
# ============================================================

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode


# Define tools for our airline support agent
@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for available flights between two cities on a given date."""
    # Simulated database
    flights = {
        ("SFO", "NRT"): [
            {"flight": "NH101", "price": 850, "departure": "10:00", "airline": "ANA"},
            {"flight": "JL002", "price": 920, "departure": "14:30", "airline": "JAL"},
        ],
        ("LAX", "LHR"): [
            {"flight": "BA280", "price": 1200, "departure": "16:00", "airline": "British Airways"},
        ],
    }
    key = (origin.upper(), destination.upper())
    if key in flights:
        return f"Found flights: {flights[key]}"
    return f"No flights found from {origin} to {destination} on {date}"


@tool
def check_booking(booking_id: str) -> str:
    """Check the status of an existing booking."""
    bookings = {
        "BK001": {"status": "confirmed", "flight": "NH101", "passenger": "John"},
        "BK002": {"status": "cancelled", "flight": "BA280", "passenger": "Jane"},
    }
    if booking_id in bookings:
        return f"Booking {booking_id}: {bookings[booking_id]}"
    return f"Booking {booking_id} not found"


@tool
def book_flight(flight_id: str, passenger_name: str) -> str:
    """Book a specific flight for a passenger."""
    return f"Successfully booked flight {flight_id} for {passenger_name}. Booking ID: BK{hash(flight_id) % 1000:03d}"


@tool
def cancel_booking(booking_id: str) -> str:
    """Cancel an existing booking."""
    return f"Booking {booking_id} has been cancelled. Refund will be processed in 3-5 business days."


@tool 
def get_baggage_policy(airline: str) -> str:
    """Get baggage allowance policy for an airline."""
    policies = {
        "ANA": "2 bags x 23kg each included",
        "JAL": "2 bags x 23kg each included",
        "British Airways": "1 bag x 23kg included, additional bag $75",
    }
    return policies.get(airline, f"Baggage policy for {airline} not found")


# Build the LangGraph agent
tools = [search_flights, check_booking, book_flight, cancel_booking, get_baggage_policy]

model = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.1,  # Low temp for more consistent behavior
).bind_tools(tools)


def agent_node(state: MessagesState):
    """The main agent node that decides what to do."""
    system = SystemMessage(content="""You are a helpful airline support agent. You can:
- Search for flights (search_flights)
- Check booking status (check_booking)
- Book flights (book_flight)
- Cancel bookings (cancel_booking)
- Get baggage policies (get_baggage_policy)

Always search before booking. Always check a booking before cancelling.
Be concise and helpful. Use tools when needed.""")
    
    messages = [system] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    """Decide whether to call tools or finish."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "__end__"


# Build graph
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
app = graph.compile()


# ============================================================
# AgentGate Discovery workflow
# ============================================================

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentgate.discover import Discovery
from agentgate.adapters.langgraph import LangGraphAdapter

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  AgentGate Discovery — Real LangGraph Agent                  ║
║  Using DeepSeek API (OpenAI-compatible)                      ║
╠══════════════════════════════════════════════════════════════╣
║  Step 1: Record real agent behavior                          ║
║  Step 2: Extract behavioral patterns                         ║
║  Step 3: Auto-generate E2E scenarios                         ║
║  Step 4: Run scenarios as tests                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Create adapter
    adapter = LangGraphAdapter(app)

    # Step 1: Discovery — Record behavior
    print("📹 Step 1: Recording agent behavior (3 inputs × 3 runs each)...\n")
    discovery = Discovery(adapter)

    inputs = [
        "Search for flights from SFO to NRT on March 15th",
        "Check the status of booking BK001",
        "What's the baggage policy for ANA?",
    ]

    for input_text in inputs:
        print(f"  Recording: \"{input_text}\"")
        start = time.time()
        discovery.record(input_text, runs=3, on_progress=lambda i, n: print(f"    Run {i}/{n}...", end=" "))
        elapsed = time.time() - start
        print(f"({elapsed:.1f}s)")

    # Step 2: Analyze patterns
    print("\n📊 Step 2: Analyzing patterns...")
    report = discovery.analyze()
    print(report.summary())

    # Step 3: Generate scenarios
    print("\n\n🔧 Step 3: Auto-generating E2E scenarios...")
    scenarios = discovery.generate_scenarios(min_frequency=0.6)
    print(f"   Generated {len(scenarios)} scenarios")

    # Save for human review
    out_path = discovery.save_scenarios("generated/airline_agent.py", min_frequency=0.6)
    print(f"   Saved to: {out_path}")
    print("   ⚠️  Review the file and add safety guardrails before using in CI!")

    # Step 4: Run the generated scenarios as tests
    print("\n\n🧪 Step 4: Running generated scenarios against the agent...")
    suite = discovery.to_suite(min_frequency=0.6)
    result = suite.run(adapter, runs=2, min_pass_rate=0.5)
    print(result.summary())

    # CI/CD gate
    print("\n" + "=" * 60)
    if result.passed:
        print("🚀 DEPLOY: All E2E scenarios passed")
    else:
        print(f"🚫 BLOCKED: {len([r for r in result.results if not r.passed])} scenario(s) failed")

    # Save traces for future mock replay
    discovery.save_traces("traces/airline_agent/")
    print(f"\n💾 Traces saved to traces/airline_agent/ (for mock replay in CI)")

    return 0 if result.passed else 1


if __name__ == "__main__":
    exit(main())
