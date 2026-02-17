"""Test noise robustness (AgentNoiseBench, 2026)."""
from agentgate import (
    Scenario, MockAgent, AgentTrace, AgentStep, StepKind,
    UserNoise, ToolNoise, NoisyScenario,
    apply_user_noise, apply_tool_noise,
    ExpectNoiseRobustness, noise_robustness_suite,
)


def test_user_noise_typos():
    """Typo injection modifies text."""
    text = "Please search for the best restaurant near me"
    noisy = apply_user_noise(text, UserNoise(typos=0.5, seed=42))
    # Should be modified but still readable
    assert noisy != text or len(text) < 5  # very short text might not change
    assert len(noisy) == len(text)  # typos don't change length


def test_user_noise_ambiguity():
    """Ambiguity injection makes instructions vaguer."""
    text = "Book exactly 3 tickets and always confirm"
    noisy = apply_user_noise(text, UserNoise(ambiguity=1.0, seed=42))
    # Should replace exact terms with vague ones
    assert "exactly" not in noisy or "always" not in noisy or "some" in noisy


def test_user_noise_redundancy():
    """Redundancy adds irrelevant info."""
    text = "Find flights to Tokyo"
    noisy = apply_user_noise(text, UserNoise(redundancy=1.0, seed=42))
    assert len(noisy) > len(text)


def test_user_noise_zero():
    """No noise = no change."""
    text = "Book a flight to Paris"
    noisy = apply_user_noise(text, UserNoise())
    assert noisy == text


def test_user_noise_deterministic():
    """Same seed = same result."""
    text = "Search for restaurants"
    n1 = apply_user_noise(text, UserNoise(typos=0.3, seed=123))
    n2 = apply_user_noise(text, UserNoise(typos=0.3, seed=123))
    assert n1 == n2


def test_tool_noise_failure():
    """Tool failure injection."""
    steps = [
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="results"),
        AgentStep(kind=StepKind.TOOL_CALL, name="book", output="booked"),
    ]
    noisy = apply_tool_noise(steps, ToolNoise(failure_rate=1.0, seed=42))
    assert all("ERROR" in str(s.output) for s in noisy)
    assert all(s.metadata.get("_noise") == "failure" for s in noisy)


def test_tool_noise_partial():
    """Partial results injection."""
    steps = [
        AgentStep(kind=StepKind.TOOL_CALL, name="search",
                  output="A very long result with lots of detailed information"),
    ]
    noisy = apply_tool_noise(steps, ToolNoise(
        failure_rate=0.0, partial_results=1.0, seed=42
    ))
    assert "truncated" in str(noisy[0].output)


def test_tool_noise_preserves_non_tool():
    """Non-tool steps are not modified."""
    steps = [
        AgentStep(kind=StepKind.LLM_CALL, name="think", output="thinking..."),
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="results"),
    ]
    noisy = apply_tool_noise(steps, ToolNoise(failure_rate=1.0, seed=42))
    assert noisy[0].output == "thinking..."  # Reasoning untouched
    assert "ERROR" in str(noisy[1].output)  # Tool call noised


def test_tool_noise_zero():
    """No noise = no change."""
    steps = [AgentStep(kind=StepKind.TOOL_CALL, name="a", output="ok")]
    noisy = apply_tool_noise(steps, ToolNoise())
    assert noisy[0].output == "ok"


def test_noisy_scenario():
    """NoisyScenario wraps base scenario."""
    base = Scenario("Book flight", input="Book exactly 3 tickets to Paris")
    noisy = NoisyScenario(
        base=base,
        user_noise=UserNoise(ambiguity=1.0, seed=42),
    )
    s = noisy.to_scenario()
    assert "[noisy]" in s.name
    assert s._noise_type == "user"


def test_noise_robustness_expectation_pass():
    """Agent remains robust under noise."""
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="results",
                  metadata={"_noise": "failure"}),
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="real results"),
        AgentStep(kind=StepKind.TOOL_CALL, name="book", output="booked"),
    ]))

    exp = ExpectNoiseRobustness(required_tools=["search", "book"])
    trace = mock.run("q")
    result = exp.check(trace)
    assert result.passed
    assert "1 noise events" in result.detail


def test_noise_robustness_expectation_fail():
    """Agent fails to complete task under noise."""
    mock = MockAgent()
    mock.add_trace("q", AgentTrace(input="q", steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="ERROR",
                  metadata={"_noise": "failure"}),
    ]))

    exp = ExpectNoiseRobustness(required_tools=["search", "book"])
    trace = mock.run("q")
    result = exp.check(trace)
    assert not result.passed
    assert "book" in result.detail


def test_noise_robustness_suite():
    """Suite generates original + noisy pairs."""
    scenarios = [
        Scenario("A", input="do A"),
        Scenario("B", input="do B"),
    ]
    suite = noise_robustness_suite(
        scenarios,
        user_noise=UserNoise(typos=0.1, seed=42),
    )
    assert len(suite) == 4  # 2 original + 2 noisy
    assert "[noisy]" not in suite[0].name
    assert "[noisy]" in suite[1].name
