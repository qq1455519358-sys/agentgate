# Changelog

## [0.3.1] - 2026-02-18

### Added
- **AnthropicAdapter** — Claude tool-use with full trace capture
- **CrewAIAdapter** — Wrap CrewAI Crew objects
- **AutoGenAdapter** — Wrap Microsoft AutoGen agents
- **FunctionAdapter** — Any function returning dict/string/trace → AgentTrace
- GitHub Pages docs deployment (MkDocs Material)
- 10 new adapter tests

### Changed
- Updated HN launch post with real API test results
- Repo metadata: topics, description, homepage

## [0.3.0] - 2026-02-17

### Added
- **OpenAIAdapter** + **AsyncOpenAIAdapter** — Full tool-use loop with trace capture
- Real API tests against DeepSeek V3 (6 tests, all passing)
- MkDocs Material documentation site
- Refactored `scenario.py` God Object into `types.py`, `expectations.py`
- 108 public API exports with full backward compatibility

### Core Modules (22)
- `scenario.py` — Scenario definition and behavioral assertions
- `types.py` — AgentTrace, AgentStep, StepKind, MockAgent
- `expectations.py` — 18 expectation types (tool_call, output, state, etc.)
- `runner.py` — TestSuite with statistical analysis (pass^k)
- `adversarial.py` — OWASP Agentic Top 10 scenario generators
- `confidence.py` — τ-bench pass^k, HTC calibration
- `metrics.py` — Node F1, Edge F1, tool edit distance
- `trace.py` — Trace recording and replay
- `report.py` — Human-readable + JSON reporting
- `kpi_trap.py` — Outcome-driven constraint violation detection
- `silent_failures.py` — IBM Research silent failure taxonomy
- `noise.py` — AgentNoiseBench robustness testing
- `memory_eval.py` — MemoryAgentBench memory evaluation
- `tool_selection.py` — ToolCert/CATS adversarial tool injection
- `multi_agent.py` — MultiAgentBench collaboration metrics
- `saber.py` — SABER decisive deviation scoring
- `state_diff.py` — Agent-Diff outcome verification
- `graph_analysis.py` — WebGraphEval trajectory analysis
- `cost.py` — Token cost and Pareto frontier
- `policy.py` — CuP policy adherence
- `regression.py` — Cross-version regression detection
- `reproducibility.py` — EvalSuiteManager capability management

### Research
- 24 papers cited and implemented
- Full reference list in README

## [0.2.0] - 2026-02-17

Initial development release.
