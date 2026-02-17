"""
🔬 AgentGate Demo: Research Assistant Agent

Tests an AI research assistant across 4 scenarios:
  A. Simple factual question (happy path: search → extract → synthesize)
  B. Empty search results (should retry with different keywords)
  C. Multi-step deep research (iterative search → analyze → compare)
  D. Contradictory sources (should flag uncertainty, not cherry-pick)

This demonstrates AgentGate testing:
  - Multi-step research workflows
  - Retry/recovery when tools return empty results
  - Information synthesis quality
  - Intellectual honesty (uncertainty acknowledgment)

Run: python examples/demo_research_assistant.py
"""

from agentgate.scenario import (
    Scenario, ScenarioSuite, AgentTrace, AgentStep, StepKind,
    CallableAgentAdapter,
)

# ============================================================
# Simulated Research Assistant Agent
# ============================================================

def simulate_research_agent(input_text: str) -> AgentTrace:
    """
    Simulates a research assistant agent that:
    - Searches the web for information
    - Extracts and synthesizes findings
    - Handles empty results, multi-step research, and contradictions
    """
    text = input_text.lower()
    steps = []

    # --- Scenario A: Simple factual question (happy path) ---
    if "population" in text or "capital" in text:
        steps = [
            AgentStep(
                kind=StepKind.LLM_CALL, name="agent",
                output="Let me research that for you.",
            ),
            AgentStep(
                kind=StepKind.TOOL_CALL, name="search_web",
                input={"query": "Japan population 2024", "num_results": 5},
                output={"results": [
                    {"title": "Japan Demographics 2024", "url": "https://worlddata.info/japan",
                     "snippet": "Japan's population in 2024 is approximately 123.3 million."},
                    {"title": "Japan Census Bureau", "url": "https://stat.go.jp",
                     "snippet": "Official census data: 123,294,513 as of Oct 2024."},
                ]},
            ),
            AgentStep(
                kind=StepKind.TOOL_CALL, name="extract_info",
                input={"urls": ["https://worlddata.info/japan", "https://stat.go.jp"]},
                output={"key_facts": [
                    "Population: ~123.3 million (2024)",
                    "Declining at 0.5% per year",
                    "Median age: 49.1 years",
                ]},
            ),
            AgentStep(
                kind=StepKind.TOOL_CALL, name="synthesize",
                input={"facts": ["123.3 million", "declining 0.5%/yr", "median age 49.1"]},
                output={"summary": "Japan's population is 123.3M and declining.",
                        "confidence": 0.95, "sources": 2},
            ),
            AgentStep(
                kind=StepKind.STATE_CHANGE, name="research_complete",
                output=True,
            ),
        ]
        output = (
            "Japan's population in 2024 is approximately 123.3 million people, "
            "according to both the World Data Index and Japan's Census Bureau. "
            "The population is declining at about 0.5% per year, with a median age "
            "of 49.1 years. [Sources: worlddata.info, stat.go.jp]"
        )

    # --- Scenario B: Empty search results → should retry with different keywords ---
    elif "obscure" in text or "niche" in text:
        steps = [
            AgentStep(
                kind=StepKind.LLM_CALL, name="agent",
                output="Let me search for that information.",
            ),
            # First search returns nothing
            AgentStep(
                kind=StepKind.TOOL_CALL, name="search_web",
                input={"query": "Voynich manuscript paragraph 42 decoded 2024"},
                output={"results": []},
            ),
            # ⚠️ BUG: Agent gives up immediately instead of retrying
            # A good agent would try broader keywords like "Voynich manuscript latest research"
            AgentStep(
                kind=StepKind.LLM_CALL, name="agent",
                output="I couldn't find any results.",
            ),
            AgentStep(
                kind=StepKind.STATE_CHANGE, name="research_complete",
                output=True,
            ),
        ]
        output = (
            "I wasn't able to find information on that topic. "
            "No results matched your query."
        )

    # --- Scenario C: Multi-step deep research ---
    elif "compare" in text or "deep research" in text:
        steps = [
            AgentStep(
                kind=StepKind.LLM_CALL, name="agent",
                output="This requires multi-step research. Let me dig into it.",
            ),
            # Step 1: Initial broad search
            AgentStep(
                kind=StepKind.TOOL_CALL, name="search_web",
                input={"query": "Rust vs Go performance benchmarks 2024"},
                output={"results": [
                    {"title": "Rust vs Go Benchmarks", "url": "https://bench.dev/rust-go",
                     "snippet": "Rust outperforms Go in CPU-bound tasks by 2-5x."},
                ]},
            ),
            # Step 2: Analyze initial findings
            AgentStep(
                kind=StepKind.TOOL_CALL, name="analyze",
                input={"topic": "Rust vs Go performance", "initial_findings": "Rust 2-5x faster CPU"},
                output={"gaps": ["memory usage comparison missing",
                         "concurrency benchmarks needed",
                         "developer productivity not covered"],
                        "next_queries": ["Rust vs Go memory usage", "Rust vs Go concurrency"]},
            ),
            # Step 3: Deeper targeted search
            AgentStep(
                kind=StepKind.TOOL_CALL, name="search_deeper",
                input={"queries": ["Rust vs Go memory usage 2024",
                                   "Rust vs Go concurrency performance"]},
                output={"results": [
                    {"topic": "memory", "finding": "Rust uses 30-50% less memory than Go due to no GC"},
                    {"topic": "concurrency", "finding": "Go goroutines easier; Rust async slightly faster"},
                ]},
            ),
            # Step 4: Compare and conclude
            AgentStep(
                kind=StepKind.TOOL_CALL, name="compare",
                input={"dimensions": ["cpu", "memory", "concurrency", "productivity"]},
                output={"comparison": {
                    "cpu": {"winner": "Rust", "margin": "2-5x"},
                    "memory": {"winner": "Rust", "margin": "30-50% less"},
                    "concurrency": {"winner": "tie", "note": "Go easier, Rust faster"},
                    "productivity": {"winner": "Go", "note": "faster to write"},
                }},
            ),
            AgentStep(
                kind=StepKind.STATE_CHANGE, name="research_complete",
                output=True,
            ),
        ]
        output = (
            "Based on my research across multiple sources:\n\n"
            "**Rust vs Go Performance (2024):**\n"
            "• CPU: Rust wins by 2-5x in compute-heavy tasks\n"
            "• Memory: Rust uses 30-50% less (no garbage collector)\n"
            "• Concurrency: Tie — Go's goroutines are simpler, Rust's async is slightly faster\n"
            "• Developer Productivity: Go wins — faster development cycles\n\n"
            "Bottom line: Rust for performance-critical systems, Go for rapid development."
        )

    # --- Scenario D: Contradictory sources → should flag uncertainty ---
    elif "contradictory" in text or "controversial" in text:
        # ⚠️ BUG: Agent cherry-picks one source and presents it as fact
        steps = [
            AgentStep(
                kind=StepKind.LLM_CALL, name="agent",
                output="Let me research that question.",
            ),
            AgentStep(
                kind=StepKind.TOOL_CALL, name="search_web",
                input={"query": "coffee health effects 2024 study"},
                output={"results": [
                    {"title": "Coffee Extends Lifespan", "url": "https://health-journal.com/a1",
                     "snippet": "Study of 500K people shows 3+ cups/day reduces mortality 15%."},
                    {"title": "Coffee Linked to Heart Risk", "url": "https://cardio-research.org/b2",
                     "snippet": "New meta-analysis: >4 cups/day increases cardiovascular risk 20%."},
                    {"title": "Coffee Effects Are Genetic", "url": "https://genetics-today.com/c3",
                     "snippet": "CYP1A2 gene variant determines if coffee is beneficial or harmful."},
                ]},
            ),
            AgentStep(
                kind=StepKind.TOOL_CALL, name="extract_info",
                input={"urls": ["https://health-journal.com/a1",
                                "https://cardio-research.org/b2",
                                "https://genetics-today.com/c3"]},
                output={"findings": [
                    {"claim": "Coffee reduces mortality 15%", "source": "health-journal.com"},
                    {"claim": "Coffee increases cardiovascular risk 20%", "source": "cardio-research.org"},
                    {"claim": "Effects depend on CYP1A2 genotype", "source": "genetics-today.com"},
                ]},
            ),
            # ⚠️ BUG: Synthesize only uses the positive study!
            AgentStep(
                kind=StepKind.TOOL_CALL, name="synthesize",
                input={"facts": ["Coffee reduces mortality 15%"]},  # cherry-picked!
                output={"summary": "Coffee is beneficial — drink 3+ cups daily.",
                        "confidence": 0.90, "sources": 1},  # only 1 source used
            ),
            AgentStep(
                kind=StepKind.STATE_CHANGE, name="research_complete",
                output=True,
            ),
        ]
        output = (
            "Research shows that drinking 3 or more cups of coffee per day reduces "
            "mortality by 15%. Coffee is generally beneficial for health."
        )

    else:
        steps = [
            AgentStep(
                kind=StepKind.LLM_CALL, name="agent",
                output="What would you like me to research?",
            ),
        ]
        output = "I can help you research any topic. What question would you like me to investigate?"

    return AgentTrace(
        input=input_text,
        output=output,
        steps=steps,
        total_duration_ms=2400.0,
    )


# ============================================================
# Define E2E Scenarios
# ============================================================

def build_research_suite() -> ScenarioSuite:
    suite = ScenarioSuite("research-assistant-agent")

    # ----------------------------------------------------------
    # Scenario A: Factual question (happy path)
    # Tests: search → extract → synthesize pipeline, output with sources
    # ----------------------------------------------------------
    factual = Scenario(
        "Factual question (search → extract → synthesize)",
        input="What is the population of Japan in 2024?",
    )
    factual.expect_tool_call("search_web", order=1)
    factual.expect_tool_call("extract_info", order=2)
    factual.expect_tool_call("synthesize", order=3)
    factual.expect_tool_order(["search_web", "extract_info", "synthesize"])
    factual.expect_state("research_complete", value=True)
    factual.expect_output(contains="123")                    # the population figure
    factual.expect_output(contains="Sources")                 # should cite sources
    factual.expect_no_error()
    factual.expect_max_steps(8)
    suite.add(factual)

    # ----------------------------------------------------------
    # Scenario B: Empty search results → should retry
    # Tests: agent MUST try different keywords, not give up immediately
    # This will FAIL because our buggy agent doesn't retry
    # ----------------------------------------------------------
    empty_results = Scenario(
        "Empty search → should retry with different keywords",
        input="Research this obscure topic about Voynich manuscript decoding",
    )
    # Agent should search at least twice (retry with broader keywords)
    empty_results.expect_tool_call("search_web", times=2)
    # Output should NOT just say "no results" — should show effort
    empty_results.expect_output(not_contains="No results matched")
    suite.add(empty_results)

    # ----------------------------------------------------------
    # Scenario C: Multi-step deep research
    # Tests: iterative research with analysis gaps → deeper search → comparison
    # ----------------------------------------------------------
    deep = Scenario(
        "Multi-step deep research (search → analyze → compare)",
        input="Compare Rust vs Go for backend development — deep research",
    )
    deep.expect_tool_call("search_web", order=1)
    deep.expect_tool_call("analyze", order=2)
    deep.expect_tool_call("search_deeper", order=3)
    deep.expect_tool_call("compare", order=4)
    deep.expect_tool_order(["search_web", "analyze", "search_deeper", "compare"])
    deep.expect_state("research_complete", value=True)
    deep.expect_output(contains="Rust")
    deep.expect_output(contains="Go")
    deep.expect_max_steps(10)
    deep.expect_no_error()
    suite.add(deep)

    # ----------------------------------------------------------
    # Scenario D: Contradictory sources → must flag uncertainty
    # Tests: agent should NOT cherry-pick one source; should note contradictions
    # This will FAIL because our buggy agent only cites the positive study
    # ----------------------------------------------------------
    contradictory = Scenario(
        "Contradictory sources → should flag uncertainty",
        input="Is coffee healthy? I've heard contradictory things — controversial topic",
    )
    contradictory.expect_tool_call("search_web")
    contradictory.expect_tool_call("extract_info")
    contradictory.expect_tool_call("synthesize")
    # Output quality checks — a responsible agent should:
    contradictory.expect_output(contains="cardiovascular")   # mention the negative study too
    contradictory.expect_output(not_contains="generally beneficial")  # shouldn't give one-sided conclusion
    contradictory.expect_state("research_complete", value=True)
    suite.add(contradictory)

    return suite


# ============================================================
# Run the demo
# ============================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  AgentGate Demo: Research Assistant Agent                     ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  4 scenarios testing a research assistant agent:             ║
║                                                              ║
║  A. Factual question         — search → extract → synthesize ║
║  B. Empty search results     — should retry, not give up     ║
║  C. Multi-step research      — iterative deep investigation  ║
║  D. Contradictory sources    — should flag uncertainty        ║
║                                                              ║
║  Scenarios B and D are intentionally buggy:                  ║
║  B: Agent gives up after first empty search (no retry)       ║
║  D: Agent cherry-picks one source, ignores contradictions    ║
║                                                              ║
║  These are real-world research quality issues that only      ║
║  E2E behavioral testing can catch.                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    suite = build_research_suite()
    adapter = CallableAgentAdapter(
        simulate_research_agent,
        trace_extractor=lambda result: result,
    )
    result = suite.run(adapter)

    print(result.summary())

    # CI/CD gate
    print("\n" + "=" * 60)
    passed_count = sum(1 for r in result.results if r.passed)
    failed_count = len(result.results) - passed_count
    if result.passed:
        print("🚀 DEPLOY: All research scenarios passed")
    else:
        print(f"🚫 BLOCKED: {failed_count} scenario(s) failed")
        print("   Issues found:")
        for r in result.results:
            if not r.passed:
                print(f"   • {r.scenario_name}:")
                for e in r.failed_expectations:
                    print(f"     - {e.description}")
        print("\n   These are the kinds of bugs that slip past unit-level evals.")
        print("   A response can be 'fluent and helpful' while being dangerously wrong.")

    return 0 if result.passed else 1


if __name__ == "__main__":
    exit(main())
