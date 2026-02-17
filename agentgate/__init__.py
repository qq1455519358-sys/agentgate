"""
AgentGate — E2E behavioral testing for AI agents.

Your agent passed all unit tests. Then it deleted production data.
DeepEval tests what your LLM says. AgentGate tests what your agent does.
"""

__version__ = "0.3.0"

# Core types (canonical source: agentgate.types)
from agentgate.types import (
    StepKind,
    AgentStep,
    AgentTrace,
    ExpectationResult,
    SingleRunResult,
    ScenarioResult,
    SuiteResult,
)

# Expectations (canonical source: agentgate.expectations)
from agentgate.expectations import (
    Expectation,
    ExpectToolCall,
    ExpectNoToolCall,
    ExpectToolOrder,
    ExpectState,
    ExpectNoError,
    ExpectOnToolFailure,
    ExpectOutput,
    ExpectMaxSteps,
    ExpectMaxDuration,
    ExpectNoSideEffects,
    ExpectNoRepetition,
    ExpectMaxTokens,
    ExpectMilestone,
    ExpectLLMJudge,
)

# Scenario & Suite (still in scenario.py but thinner now)
from agentgate.scenario import (
    Scenario,
    ScenarioSuite,
    AgentAdapter,
    CallableAgentAdapter,
)

from agentgate.runner import TestSuite
from agentgate.mock import MockAgent
from agentgate.trace import TraceRecorder

from agentgate.discovery import Discovery
from agentgate.report import diff_results
from agentgate.metrics import (
    node_f1, edge_f1, tool_edit_distance,
    side_effect_rate, repetition_rate, decisive_deviation_score,
)
from agentgate.cost import (
    TokenUsage, token_cost, cost_score,
    AgentResult, pareto_frontier, is_pareto_optimal,
)
from agentgate.kpi_trap import kpi_trap, kpi_trap_suite
from agentgate.state_diff import StateDiff, ExpectStateDiff, expect_state_diff
from agentgate.regression import EvalSuiteManager, EvalResult
from agentgate.noise import (
    UserNoise, ToolNoise, NoisyScenario,
    apply_user_noise, apply_tool_noise,
    ExpectNoiseRobustness, noise_robustness_suite,
)
from agentgate.confidence import (
    trajectory_confidence,
    ExpectCalibratedConfidence, ExpectAppropriateEscalation,
    HandoffError, detect_handoff_errors,
    expect_calibrated_confidence, expect_appropriate_escalation,
)
from agentgate.reproducibility import (
    VarianceReport, variance_report, reproducibility_score,
    ExpectReproducible, expect_reproducible,
)
from agentgate.silent_failures import (
    SilentFailureReport, detect_drift, detect_cycles,
    detect_missing_details, detect_silent_tool_errors,
    full_silent_failure_scan,
    ExpectNoSilentFailures, expect_no_silent_failures,
)
from agentgate.tool_selection import (
    AdversarialTool, ToolSelectionTest,
    ExpectCorrectToolSelection, ExpectNoToolConfusion,
    expect_correct_tool_selection, expect_no_tool_confusion,
    tool_selection_robustness_suite,
)
from agentgate.multi_agent import (
    AgentRole, MultiAgentTrace,
    collaboration_quality, coordination_efficiency,
    ExpectCollaborationQuality, ExpectNoFreeRiders,
    expect_collaboration_quality, expect_no_free_riders,
)
from agentgate.memory_eval import (
    MemoryProbe, ExpectAccurateRetrieval, ExpectConflictResolution,
    ExpectMemoryConsistency, expect_accurate_retrieval,
    expect_conflict_resolution, expect_memory_consistency,
    memory_consistency_suite,
)
from agentgate.trajectory import (
    step_credit, critical_steps, trajectory_redundancy, trajectory_efficiency,
)
from agentgate.adversarial import (
    prompt_injection,
    privilege_escalation,
    tool_misuse,
    data_exfiltration,
    owasp_agentic_suite,
)

__all__ = [
    # Primary API
    "Scenario", "TestSuite", "MockAgent", "TraceRecorder", "Discovery",
    # Types
    "AgentTrace", "AgentStep", "StepKind",
    "ExpectationResult", "ScenarioResult", "SingleRunResult", "SuiteResult",
    # Adapters
    "AgentAdapter", "CallableAgentAdapter",
    # Expectations
    "Expectation",
    "ExpectToolCall", "ExpectNoToolCall", "ExpectToolOrder",
    "ExpectState", "ExpectNoError", "ExpectOnToolFailure", "ExpectOutput",
    "ExpectMaxSteps", "ExpectMaxDuration",
    "ExpectNoSideEffects", "ExpectNoRepetition", "ExpectMaxTokens",
    "ExpectMilestone", "ExpectLLMJudge",
    # Regression
    "diff_results",
    # Metrics
    "node_f1", "edge_f1", "tool_edit_distance",
    "side_effect_rate", "repetition_rate", "decisive_deviation_score",
    # Cost
    "TokenUsage", "token_cost", "cost_score",
    "AgentResult", "pareto_frontier", "is_pareto_optimal",
    # KPI trap
    "kpi_trap", "kpi_trap_suite",
    # Trajectory
    "step_credit", "critical_steps", "trajectory_redundancy", "trajectory_efficiency",
    # State diff
    "StateDiff", "ExpectStateDiff", "expect_state_diff",
    # Regression management
    "EvalSuiteManager", "EvalResult",
    # Noise
    "UserNoise", "ToolNoise", "NoisyScenario",
    "apply_user_noise", "apply_tool_noise",
    "ExpectNoiseRobustness", "noise_robustness_suite",
    # Memory
    "MemoryProbe", "ExpectAccurateRetrieval", "ExpectConflictResolution",
    "ExpectMemoryConsistency", "expect_accurate_retrieval",
    "expect_conflict_resolution", "expect_memory_consistency",
    "memory_consistency_suite",
    # Multi-agent
    "AgentRole", "MultiAgentTrace",
    "collaboration_quality", "coordination_efficiency",
    "ExpectCollaborationQuality", "ExpectNoFreeRiders",
    "expect_collaboration_quality", "expect_no_free_riders",
    # Silent failures
    "SilentFailureReport", "detect_drift", "detect_cycles",
    "detect_missing_details", "detect_silent_tool_errors",
    "full_silent_failure_scan",
    "ExpectNoSilentFailures", "expect_no_silent_failures",
    # Tool selection
    "AdversarialTool", "ToolSelectionTest",
    "ExpectCorrectToolSelection", "ExpectNoToolConfusion",
    "expect_correct_tool_selection", "expect_no_tool_confusion",
    "tool_selection_robustness_suite",
    # Confidence
    "trajectory_confidence",
    "ExpectCalibratedConfidence", "ExpectAppropriateEscalation",
    "HandoffError", "detect_handoff_errors",
    "expect_calibrated_confidence", "expect_appropriate_escalation",
    # Reproducibility
    "VarianceReport", "variance_report", "reproducibility_score",
    "ExpectReproducible", "expect_reproducible",
    # Adversarial
    "prompt_injection", "privilege_escalation",
    "tool_misuse", "data_exfiltration", "owasp_agentic_suite",
]
