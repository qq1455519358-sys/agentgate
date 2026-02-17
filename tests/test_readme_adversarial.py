"""Verify README adversarial code block works."""
from agentgate import (
    TestSuite, MockAgent, AgentTrace, AgentStep, StepKind,
    prompt_injection, privilege_escalation,
)


def test_readme_adversarial():
    """README adversarial example should work with a safe mock."""
    mock = MockAgent(default_trace=AgentTrace(
        input="", output="I cannot do that.",
        steps=[AgentStep(kind=StepKind.LLM_CALL, name="agent", output="Refused.")],
    ))

    suite = TestSuite("security-gate")
    for s in prompt_injection(dangerous_tools=["cancel_booking", "delete_user"]):
        suite.add(s)
    for s in privilege_escalation(dangerous_tools=["admin_panel"]):
        suite.add(s)

    result = suite.run(mock)
    assert result.passed, result.summary()
