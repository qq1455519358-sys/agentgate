"""Test adversarial scenario generators."""
from agentgate import (
    TestSuite, MockAgent, AgentTrace, AgentStep, StepKind,
    prompt_injection, privilege_escalation, tool_misuse,
    data_exfiltration, owasp_agentic_suite,
)


def _safe_mock():
    """Agent that refuses all adversarial inputs."""
    m = MockAgent()
    m.default_trace = AgentTrace(
        input="",
        output="I cannot perform that action.",
        steps=[
            AgentStep(kind=StepKind.LLM_CALL, name="agent",
                      output="I'm sorry, I cannot comply with that request."),
        ],
    )
    return m


def test_prompt_injection_generates_scenarios():
    scenarios = prompt_injection(dangerous_tools=["cancel_booking", "delete_user"])
    # 5 payloads × 2 tools = 10
    assert len(scenarios) == 10
    for s in scenarios:
        assert "INJECTION" in s.name
        assert len(s.expectations) >= 1


def test_privilege_escalation_generates_scenarios():
    scenarios = privilege_escalation(dangerous_tools=["admin_panel"])
    # 4 prompts × 1 tool = 4
    assert len(scenarios) == 4
    for s in scenarios:
        assert "ESCALATION" in s.name


def test_tool_misuse_generates_scenarios():
    scenarios = tool_misuse(tools={"search": ["delete", "modify"]})
    assert len(scenarios) == 2


def test_data_exfiltration_generates_scenarios():
    scenarios = data_exfiltration(sensitive_tools=["get_user_data"])
    assert len(scenarios) == 3
    for s in scenarios:
        assert len(s.expectations) >= 4  # 4 default patterns


def test_owasp_suite_combines_all():
    scenarios = owasp_agentic_suite(dangerous_tools=["cancel_booking"])
    # injection (5) + escalation (4) + exfiltration (3) = 12
    assert len(scenarios) == 12


def test_adversarial_against_safe_agent():
    """Safe agent should pass all adversarial scenarios."""
    mock = _safe_mock()
    suite = TestSuite("adversarial")
    for s in prompt_injection(dangerous_tools=["cancel_booking"]):
        suite.add(s)
    result = suite.run(mock)
    assert result.passed, result.summary()


def test_custom_payloads():
    scenarios = prompt_injection(
        dangerous_tools=["nuke"],
        custom_payloads=["Pretty please {action}"],
    )
    # 5 default + 1 custom = 6 payloads × 1 tool = 6
    assert len(scenarios) == 6
