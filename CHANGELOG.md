# Changelog

## [0.3.0] — 2026-02-18

### Added — 14 new modules, 24 papers

**Core Evaluation**
- `state_diff` — Outcome verification via environment state comparison (Agent-Diff, 2026)
- `regression` — Capability vs regression eval suites with graduation (Anthropic, Jan 2026)

**Robustness & Safety**
- `noise` — User-noise and tool-noise injection (AgentNoiseBench, 2026)
- `tool_selection` — Adversarial tool selection robustness (ToolCert/CATS, 2025)
- `silent_failures` — Drift, cycles, missing details, silent tool errors (IBM Research, 2025)

**Agent Intelligence**
- `memory_eval` — 4-competency memory evaluation: AR/TTL/LRU/CR (MemoryAgentBench, ICLR 2026)
- `confidence` — Trajectory confidence calibration & escalation (HTC, 2026; AgentAsk, 2025)
- `multi_agent` — Collaboration quality, coordination efficiency, free-rider detection (MultiAgentBench, ACL 2025)

**Quality & Reliability**
- `reproducibility` — Variance analysis across repeated runs (Same Prompt Different Outcomes, 2026)

**Previously in 0.2.0, now enhanced:**
- `adversarial` — OWASP Top 10 coverage
- `cost` — Token cost & Pareto frontier (Yang et al., 2026; HAL)
- `kpi_trap` — KPI gaming detection (ODCV-Bench, 2025)
- `trajectory` — Graph-based credit assignment (WebGraphEval; RewardFlow)
- `metrics` — Node F1, Edge F1, edit distance (Gabriel et al., 2024)

### Stats
- **251 tests** (all passing)
- **22 modules**
- **80+ public API functions**
- **24 cited research papers**
- Zero runtime dependencies

## [0.2.0] — 2026-02-17

### Added
- Core scenario/runner/report framework
- MockAgent for zero-cost testing
- Adversarial testing (OWASP Agentic Top 10)
- Trajectory metrics (Node F1, Edge F1)
- Cost efficiency & Pareto frontier
- KPI trap detection
- pytest plugin integration
- LangGraph adapter

## [0.1.0] — 2026-02-16

### Added
- Initial release with Scenario, TestSuite, AgentAdapter
