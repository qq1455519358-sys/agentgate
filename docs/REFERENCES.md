# References

AgentGate implements evaluation techniques from 24 research papers. Each module cites its source in the docstring.

## Core Methodology

1. **τ-bench** — Yao et al. (2024). "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains." ICLR 2025. [arXiv:2406.12045](https://arxiv.org/abs/2406.12045)
   - Module: `confidence.py` (pass^k metric)

2. **Tool F1 / Structural Similarity** — Gabriel et al. (2024). NeurIPS 2024 Workshop. [arXiv:2410.22457](https://arxiv.org/abs/2410.22457)
   - Module: `metrics.py` (node_f1, edge_f1)

3. **TRACE** — Kim et al. (2025). "TRACE: Multi-dimensional Trajectory Evaluation via Evidence Bank." [arXiv:2510.02837](https://arxiv.org/abs/2510.02837)
   - Module: `trace.py`, `graph_analysis.py`

## Security & Adversarial

4. **AgentHarm** — Andriushchenko et al. (2024). "AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents." ICLR 2025. [arXiv:2410.09024](https://arxiv.org/abs/2410.09024)
   - Module: `adversarial.py`

5. **ASB** — Zhang et al. (2024). "Agent Security Bench." ICLR 2025. [arXiv:2410.02644](https://arxiv.org/abs/2410.02644)
   - Module: `adversarial.py`

6. **OWASP Top 10 for Agentic Applications** (2025). [Report](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)
   - Module: `adversarial.py`

7. **ToolCert/CATS** (2025). "Adversarial Tool Selection." [arXiv:2510.03992](https://arxiv.org/abs/2510.03992)
   - Module: `tool_selection.py`

## Trajectory Analysis

8. **WebGraphEval** — Qian et al. (2025). NeurIPS 2025. [arXiv:2510.19205](https://arxiv.org/abs/2510.19205)
   - Module: `graph_analysis.py`

9. **RewardFlow** — ICLR 2026. [OpenReview](https://openreview.net/forum?id=5oGJbM5u86)
   - Module: `graph_analysis.py`

10. **SABER** — Cuadron et al. (2025). ICLR 2026. [arXiv:2512.07850](https://arxiv.org/abs/2512.07850)
    - Module: `saber.py`

11. **AgentRewardBench** — Lù et al. (2025). McGill/Mila. NeurIPS 2025. [arXiv:2504.08942](https://arxiv.org/abs/2504.08942)
    - Module: `side_effects.py`

## Robustness & Reliability

12. **AgentNoiseBench** (2026). [arXiv:2602.11348](https://arxiv.org/abs/2602.11348)
    - Module: `noise.py`

13. **Silent Failures** — IBM Research (2025). [arXiv:2511.04032](https://arxiv.org/abs/2511.04032)
    - Module: `silent_failures.py`

14. **HTC Calibration** (2025). [arXiv:2601.15778](https://arxiv.org/abs/2601.15778)
    - Module: `confidence.py`

## Agent Capabilities

15. **MemoryAgentBench** — ICLR 2026. [arXiv:2507.05257](https://arxiv.org/abs/2507.05257)
    - Module: `memory_eval.py`

16. **MultiAgentBench** (2025). ACL 2025. [arXiv:2503.01935](https://arxiv.org/abs/2503.01935)
    - Module: `multi_agent.py`

17. **ValueFlow** (2026). [arXiv:2602.08567](https://arxiv.org/abs/2602.08567)
    - Module: `multi_agent.py`

## Evaluation Frameworks

18. **HAL** — Princeton (2025). Holistic Agent Leaderboard. [arXiv:2510.11977](https://arxiv.org/abs/2510.11977)
    - Module: `cost.py`

19. **ODCV-Bench** — Li et al. (2025). Under review ICML 2026. [arXiv:2512.20798](https://arxiv.org/abs/2512.20798)
    - Module: `kpi_trap.py`

20. **ST-WebAgentBench** — Shlomov et al. (2024). [arXiv:2410.06703](https://arxiv.org/abs/2410.06703)
    - Module: `policy.py`

21. **Agent-Diff** (2026). [arXiv:2602.11224](https://arxiv.org/abs/2602.11224)
    - Module: `state_diff.py`

## Industry Guides

22. **Anthropic** (2026). "Demystifying Evals for AI Agents." [Blog](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
    - Module: `reproducibility.py`

23. **ICLR 2026 Blog** — "A Hitchhiker's Guide to Agent Evaluation." [Blog](https://iclr-blogposts.github.io/2026/blog/2026/agent-evaluation/)
    - Architecture: overall design

24. **Yang et al. (2026)** — "Toward Efficient Agents." [arXiv:2601.14192](https://arxiv.org/abs/2601.14192)
    - Module: `cost.py`
