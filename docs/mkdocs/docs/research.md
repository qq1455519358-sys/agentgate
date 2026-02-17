# Research Foundation

AgentGate implements evaluation techniques from **24 published papers** spanning ICLR, NeurIPS, ACL, and top AI labs.

## Papers by Module

### Core Framework

| Paper | Venue | Module | Key Contribution |
|-------|-------|--------|-----------------|
| [A Hitchhiker's Guide to Agent Evaluation](https://iclr-blogposts.github.io/2026/blog/2026/agent-evaluation/) | ICLR 2026 Blog | Core | 8-dimension evaluation framework |
| [Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) | Anthropic 2026 | `regression` | Capability vs regression management |

### Metrics

| Paper | Venue | Module | Key Contribution |
|-------|-------|--------|-----------------|
| [τ-bench](https://arxiv.org/abs/2406.12045) | ICLR 2025 | `scenario` | pass@k / pass^k consistency metrics |
| [Advancing Agentic Systems](https://arxiv.org/abs/2410.22457) | NeurIPS 2024 | `metrics` | Node F1, Edge F1, tool edit distance |
| [SABER](https://arxiv.org/abs/2512.07850) | ICLR 2026 | `metrics` | Decisive deviation scoring |

### Safety & Adversarial

| Paper | Venue | Module | Key Contribution |
|-------|-------|--------|-----------------|
| [OWASP Top 10 Agentic](https://genai.owasp.org/) | OWASP 2025 | `adversarial` | 10 agentic attack categories |
| [AgentHarm](https://arxiv.org/abs/2410.09024) | ICLR 2025 | `adversarial` | 440 malicious agent tasks |
| [ASB](https://arxiv.org/abs/2410.02644) | ICLR 2025 | `adversarial` | 400+ tools, 27 attack methods |
| [ST-WebAgentBench](https://arxiv.org/abs/2410.06703) | 2024 | `scenario` | Completion under Policy (CuP) |

### Robustness

| Paper | Venue | Module | Key Contribution |
|-------|-------|--------|-----------------|
| [AgentNoiseBench](https://arxiv.org/abs/2602.11348) | 2026 | `noise` | User-noise + tool-noise taxonomy |
| [ToolCert/CATS](https://arxiv.org/abs/2510.03992) | 2025 | `tool_selection` | Adversarial tool injection |
| [ODCV-Bench](https://arxiv.org/abs/2512.20798) | ICML 2026 sub. | `kpi_trap` | KPI gaming detection |

### Quality & Efficiency

| Paper | Venue | Module | Key Contribution |
|-------|-------|--------|-----------------|
| [AgentRewardBench](https://arxiv.org/abs/2504.08942) | NeurIPS 2025 | `metrics` | Side effects + repetition |
| [Toward Efficient Agents](https://arxiv.org/abs/2601.14192) | 2026 | `cost` | Cost-effectiveness metrics |
| [HAL](https://arxiv.org/abs/2510.11977) | Princeton 2025 | `cost` | Pareto frontier, 21K rollouts |

### Trajectory Analysis

| Paper | Venue | Module | Key Contribution |
|-------|-------|--------|-----------------|
| [WebGraphEval](https://arxiv.org/abs/2510.19205) | NeurIPS 2025 | `trajectory` | Graph-based evaluation |
| [RewardFlow](https://openreview.net/forum?id=5oGJbM5u86) | ICLR 2026 | `trajectory` | Credit assignment |
| [Agent-Diff](https://arxiv.org/abs/2602.11224) | 2026 | `state_diff` | State diff verification |

### Multi-Agent & Memory

| Paper | Venue | Module | Key Contribution |
|-------|-------|--------|-----------------|
| [MultiAgentBench](https://arxiv.org/abs/2503.01935) | ACL 2025 | `multi_agent` | Collaboration quality |
| [ValueFlow](https://arxiv.org/abs/2602.08567) | 2026 | `multi_agent` | Free-rider detection |
| [MemoryAgentBench](https://arxiv.org/abs/2507.05257) | ICLR 2026 | `memory_eval` | 4-competency memory eval |

### Confidence & Reliability

| Paper | Venue | Module | Key Contribution |
|-------|-------|--------|-----------------|
| [HTC](https://arxiv.org/abs/2601.15778) | 2026 | `confidence` | Trajectory confidence calibration |
| [AgentAsk](https://arxiv.org/abs/2510.07593) | 2025 | `confidence` | Handoff error taxonomy |
| [Same Prompt Different Outcomes](https://arxiv.org/abs/2602.14349) | 2026 | `reproducibility` | Variance analysis |
| [HumanAgencyBench](https://arxiv.org/) | 2025 | `confidence` | Escalation appropriateness |

### Failure Detection

| Paper | Venue | Module | Key Contribution |
|-------|-------|--------|-----------------|
| [Silent Failures in Multi-Agentic AI](https://arxiv.org/abs/2511.04032) | IBM 2025 | `silent_failures` | Drift, cycles, missing details |
