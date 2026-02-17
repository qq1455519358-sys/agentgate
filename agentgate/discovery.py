"""
AgentGate Discover — Record agent behavior, extract patterns, generate scenarios.

This is the "录制 → 标注" step that comes BEFORE testing.
When you face a new agent and don't know what to expect, Discover:

1. Runs the agent N times with your inputs
2. Extracts behavioral patterns (common tool sequences, states, failures)
3. Auto-generates Scenario drafts from observed patterns
4. You review and annotate which patterns are "golden"

Usage:
    from agentgate.discover import Discovery

    # Step 1: Record
    discovery = Discovery(my_agent_adapter)
    discovery.record("Book a flight to Tokyo", runs=10)
    discovery.record("Cancel my booking", runs=10)

    # Step 2: Analyze patterns
    report = discovery.analyze()
    print(report.summary())  # shows common patterns, outliers, failure modes

    # Step 3: Generate scenario drafts
    scenarios = discovery.generate_scenarios(min_frequency=0.8)
    
    # Step 4: Save for human review
    discovery.save_scenarios("scenarios/booking_agent.py")
    # → editable Python file with auto-generated expectations
    
    # Step 5: After human review, run as tests
    suite = discovery.to_suite()
    result = suite.run(my_agent)
"""

from __future__ import annotations

import time
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from agentgate.scenario import (
    AgentAdapter, AgentTrace, AgentStep, StepKind,
    Scenario, ScenarioSuite, CallableAgentAdapter,
)


# ---------------------------------------------------------------------------
# Pattern types
# ---------------------------------------------------------------------------

@dataclass
class ToolPattern:
    """A tool call pattern observed across runs."""
    tool_name: str
    frequency: float          # 0.0-1.0, fraction of runs where this tool was called
    avg_position: float       # average position in tool call sequence
    position_std: float       # standard deviation of position
    common_args: dict[str, Any] = field(default_factory=dict)
    error_rate: float = 0.0   # fraction of calls that errored
    
    @property
    def is_consistent(self) -> bool:
        """Tool appears in >80% of runs at roughly the same position."""
        return self.frequency >= 0.8 and self.position_std < 1.5


@dataclass
class SequencePattern:
    """A tool call sequence pattern."""
    sequence: list[str]       # ordered tool names
    frequency: float          # fraction of runs with this exact sequence
    is_dominant: bool = False # most common sequence


@dataclass 
class StatePattern:
    """A state transition pattern."""
    state_key: str
    frequency: float
    common_value: Any = None
    avg_step: float = 0.0     # average step where state is reached


@dataclass
class FailurePattern:
    """A failure pattern observed across runs."""
    tool_name: str
    error_message: str
    frequency: float
    recovery_action: Optional[str] = None  # what the agent did after failure


@dataclass
class InputAnalysis:
    """Complete analysis of one input across N runs."""
    input_text: str
    total_runs: int
    traces: list[AgentTrace]
    
    # Extracted patterns
    tool_patterns: list[ToolPattern] = field(default_factory=list)
    sequence_patterns: list[SequencePattern] = field(default_factory=list)
    state_patterns: list[StatePattern] = field(default_factory=list)
    failure_patterns: list[FailurePattern] = field(default_factory=list)
    
    # Aggregate stats
    avg_steps: float = 0.0
    avg_duration_ms: float = 0.0
    success_rate: float = 0.0  # runs with no errors
    
    @property
    def dominant_sequence(self) -> Optional[SequencePattern]:
        dominant = [s for s in self.sequence_patterns if s.is_dominant]
        return dominant[0] if dominant else None
    
    @property
    def always_called(self) -> list[str]:
        """Tools called in every single run."""
        return [t.tool_name for t in self.tool_patterns if t.frequency >= 1.0]
    
    @property
    def never_called_tools(self) -> set[str]:
        """All tools available but never called (for safety guardrails)."""
        called = {t.tool_name for t in self.tool_patterns}
        return set()  # can't know available tools from traces alone
    
    def summary(self) -> str:
        lines = [
            f"\n📊 Input: \"{self.input_text}\"",
            f"   Runs: {self.total_runs} | Avg steps: {self.avg_steps:.1f} | "
            f"Avg duration: {self.avg_duration_ms:.0f}ms | Success rate: {self.success_rate:.0%}",
        ]
        
        # Tool patterns
        if self.tool_patterns:
            lines.append(f"\n   🔧 Tool Patterns:")
            for tp in sorted(self.tool_patterns, key=lambda t: t.avg_position):
                consistency = "✅ consistent" if tp.is_consistent else "⚠️ variable"
                error_info = f" (errors: {tp.error_rate:.0%})" if tp.error_rate > 0 else ""
                lines.append(
                    f"      {tp.tool_name}: {tp.frequency:.0%} of runs, "
                    f"avg position {tp.avg_position:.1f}±{tp.position_std:.1f} "
                    f"[{consistency}]{error_info}"
                )
        
        # Sequence patterns
        if self.sequence_patterns:
            lines.append(f"\n   📋 Sequences (top 3):")
            for sp in self.sequence_patterns[:3]:
                marker = "⭐" if sp.is_dominant else "  "
                lines.append(f"      {marker} {' → '.join(sp.sequence)} ({sp.frequency:.0%})")
        
        # State patterns
        if self.state_patterns:
            lines.append(f"\n   📌 States Reached:")
            for st in self.state_patterns:
                lines.append(f"      {st.state_key} = {st.common_value} ({st.frequency:.0%}, step ~{st.avg_step:.0f})")
        
        # Failure patterns
        if self.failure_patterns:
            lines.append(f"\n   ⚠️ Failure Patterns:")
            for fp in self.failure_patterns:
                recovery = f" → {fp.recovery_action}" if fp.recovery_action else ""
                lines.append(f"      {fp.tool_name}: \"{fp.error_message}\" ({fp.frequency:.0%}){recovery}")
        
        return "\n".join(lines)


@dataclass
class DiscoveryReport:
    """Complete discovery report across all inputs."""
    analyses: list[InputAnalysis]
    total_runs: int
    total_duration_ms: float
    
    def summary(self) -> str:
        lines = [
            "=" * 60,
            "🔍 AgentGate Discovery Report",
            f"   Total inputs: {len(self.analyses)} | Total runs: {self.total_runs} | "
            f"Duration: {self.total_duration_ms/1000:.1f}s",
            "=" * 60,
        ]
        for analysis in self.analyses:
            lines.append(analysis.summary())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Discovery engine
# ---------------------------------------------------------------------------

class Discovery:
    """
    Record, analyze, and generate scenarios from agent behavior.
    
    The bridge between "I have an agent" and "I have E2E tests."
    """

    def __init__(self, agent: AgentAdapter | callable):
        if callable(agent) and not isinstance(agent, AgentAdapter):
            agent = CallableAgentAdapter(agent)
        self.agent = agent
        self._recordings: dict[str, list[AgentTrace]] = defaultdict(list)

    def record(self, input_text: str, *, runs: int = 5,
               on_progress: Optional[callable] = None) -> list[AgentTrace]:
        """
        Run the agent N times with the same input and record traces.
        
        Args:
            input_text: Input to send to the agent
            runs: Number of times to run (default 5)
            on_progress: Optional callback(run_number, total_runs)
        
        Returns:
            List of recorded traces
        """
        traces = []
        for i in range(runs):
            if on_progress:
                on_progress(i + 1, runs)
            trace = self.agent.run(input_text)
            traces.append(trace)
            self._recordings[input_text].append(trace)
        return traces

    def record_batch(self, inputs: list[str], *, runs_per_input: int = 5,
                     on_progress: Optional[callable] = None) -> dict[str, list[AgentTrace]]:
        """Record multiple inputs in batch."""
        results = {}
        for idx, input_text in enumerate(inputs):
            if on_progress:
                on_progress(idx + 1, len(inputs), input_text)
            results[input_text] = self.record(input_text, runs=runs_per_input)
        return results

    def analyze(self) -> DiscoveryReport:
        """Analyze all recorded traces and extract patterns."""
        analyses = []
        total_runs = 0
        total_duration = 0.0
        
        for input_text, traces in self._recordings.items():
            analysis = self._analyze_input(input_text, traces)
            analyses.append(analysis)
            total_runs += len(traces)
            total_duration += sum(t.total_duration_ms for t in traces)
        
        return DiscoveryReport(
            analyses=analyses,
            total_runs=total_runs,
            total_duration_ms=total_duration,
        )

    def generate_scenarios(self, *, min_frequency: float = 0.8,
                           include_safety: bool = True) -> list[Scenario]:
        """
        Auto-generate Scenario objects from observed patterns.
        
        Args:
            min_frequency: Minimum frequency to include a pattern (0.0-1.0)
            include_safety: Include expect_no_tool_call for never-seen tools
        
        Returns:
            List of Scenario objects with auto-generated expectations
        """
        report = self.analyze()
        scenarios = []
        
        for analysis in report.analyses:
            scenario = Scenario(
                f"auto_{analysis.input_text[:40].replace(' ', '_')}",
                input=analysis.input_text,
            )
            
            # Add tool call expectations for consistent patterns
            for tp in analysis.tool_patterns:
                if tp.frequency >= min_frequency:
                    kwargs = {}
                    if tp.is_consistent and len(analysis.tool_patterns) > 1:
                        # Find relative ordering from dominant sequence
                        dominant = analysis.dominant_sequence
                        if dominant and tp.tool_name in dominant.sequence:
                            idx = dominant.sequence.index(tp.tool_name)
                            # Add before/after constraints
                            if idx > 0:
                                kwargs["after"] = dominant.sequence[idx - 1]
                            if idx < len(dominant.sequence) - 1:
                                kwargs["before"] = dominant.sequence[idx + 1]
                    
                    scenario.expect_tool_call(tp.tool_name, **kwargs)
            
            # Add state expectations
            for sp in analysis.state_patterns:
                if sp.frequency >= min_frequency:
                    scenario.expect_state(sp.state_key, value=sp.common_value)
            
            # Add failure recovery expectations
            for fp in analysis.failure_patterns:
                if fp.recovery_action and fp.frequency >= 0.3:
                    scenario.on_tool_failure(fp.tool_name, expect=fp.recovery_action)
            
            # Add step limit (2x avg as safety margin)
            if analysis.avg_steps > 0:
                scenario.expect_max_steps(int(analysis.avg_steps * 2.5))
            
            # Add no-error expectation if success rate is high
            if analysis.success_rate >= 0.9:
                scenario.expect_no_error()
            
            scenarios.append(scenario)
        
        return scenarios

    def save_scenarios(self, path: str | Path, *, min_frequency: float = 0.8) -> Path:
        """
        Generate scenarios and save as an editable Python file.
        
        The generated file is meant to be reviewed and adjusted by a human:
        - Remove expectations that are too strict
        - Add domain-specific expectations
        - Adjust thresholds
        """
        scenarios = self.generate_scenarios(min_frequency=min_frequency)
        report = self.analyze()
        
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        lines = [
            '"""',
            'Auto-generated AgentGate scenarios.',
            '',
            '⚠️  REVIEW BEFORE USING — These expectations were extracted from',
            f'observed behavior across {sum(a.total_runs for a in report.analyses)} runs.',
            '',
            'TODO for human reviewer:',
            '  1. Remove overly strict expectations',
            '  2. Add domain-specific safety guardrails (expect_no_tool_call)',
            '  3. Adjust thresholds based on your agent\'s requirements',
            '  4. Add expect_output() assertions for critical content',
            '"""',
            '',
            'from agentgate import Scenario, ScenarioSuite',
            '',
            '',
        ]
        
        suite_name = p.stem.replace("_", "-")
        lines.append(f'suite = ScenarioSuite("{suite_name}")')
        lines.append('')
        
        for i, (scenario, analysis) in enumerate(zip(scenarios, report.analyses)):
            lines.append(f'# {"=" * 56}')
            lines.append(f'# Scenario {i+1}: {analysis.input_text}')
            lines.append(f'# Based on {analysis.total_runs} recorded runs')
            if analysis.dominant_sequence:
                seq = " → ".join(analysis.dominant_sequence.sequence)
                lines.append(f'# Dominant path: {seq} ({analysis.dominant_sequence.frequency:.0%})')
            lines.append(f'# {"=" * 56}')
            lines.append('')
            
            var_name = f"scenario_{i+1}"
            lines.append(f'{var_name} = Scenario(')
            lines.append(f'    "{scenario.name}",')
            lines.append(f'    input="{scenario.input}",')
            lines.append(f')')
            
            # Write expectations with comments
            for exp in scenario.expectations:
                exp_type = type(exp).__name__
                if exp_type == "ExpectToolCall":
                    freq_info = ""
                    for tp in analysis.tool_patterns:
                        if tp.tool_name == exp.tool_name:
                            freq_info = f"  # seen in {tp.frequency:.0%} of runs"
                    
                    args = [f'"{exp.tool_name}"']
                    if hasattr(exp, 'before') and exp.before:
                        args.append(f'before="{exp.before}"')
                    if hasattr(exp, 'after') and exp.after:
                        args.append(f'after="{exp.after}"')
                    
                    lines.append(f'{var_name}.expect_tool_call({", ".join(args)}){freq_info}')
                
                elif exp_type == "ExpectState":
                    lines.append(f'{var_name}.expect_state("{exp.state_key}", value={exp.value!r})')
                
                elif exp_type == "ExpectOnToolFailure":
                    lines.append(f'{var_name}.on_tool_failure("{exp.tool_name}", expect="{exp.expect_behavior}")')
                
                elif exp_type == "ExpectMaxSteps":
                    lines.append(f'{var_name}.expect_max_steps({exp.max_steps})  # ~{analysis.avg_steps:.0f} avg observed')
                
                elif exp_type == "ExpectNoError":
                    lines.append(f'{var_name}.expect_no_error()  # {analysis.success_rate:.0%} success rate observed')
            
            lines.append(f'suite.add({var_name})')
            lines.append('')
            lines.append('')
        
        # Add runner boilerplate
        lines.extend([
            '# Run the suite',
            'if __name__ == "__main__":',
            '    # Replace with your actual agent adapter',
            '    # from agentgate.adapters.langgraph import LangGraphAdapter',
            '    # agent = LangGraphAdapter(your_graph)',
            '    #',
            '    # For statistical confidence with non-deterministic agents:',
            '    # result = suite.run(agent, runs=5, min_pass_rate=0.8)',
            '    #',
            '    # result = suite.run(agent)',
            '    # print(result.summary())',
            '    # assert result.passed, f"E2E tests failed: {result.pass_rate:.0%}"',
            '    print("Edit this file, connect your agent, and run!")',
        ])
        
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def to_suite(self, *, min_frequency: float = 0.8) -> ScenarioSuite:
        """Convert discovered patterns directly to a runnable ScenarioSuite."""
        scenarios = self.generate_scenarios(min_frequency=min_frequency)
        suite = ScenarioSuite("discovered")
        for s in scenarios:
            suite.add(s)
        return suite

    def save_traces(self, directory: str | Path) -> None:
        """Save all recorded traces for later analysis."""
        from agentgate.mock import MockAgent
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        for input_text, traces in self._recordings.items():
            for i, trace in enumerate(traces):
                safe = input_text[:30].replace(" ", "_").replace("/", "_")
                filename = f"{safe}_run{i+1:02d}.json"
                MockAgent.save_trace(trace, dir_path / filename)

    # -----------------------------------------------------------------------
    # Internal analysis
    # -----------------------------------------------------------------------

    def _analyze_input(self, input_text: str, traces: list[AgentTrace]) -> InputAnalysis:
        """Analyze all traces for a single input."""
        n = len(traces)
        
        analysis = InputAnalysis(
            input_text=input_text,
            total_runs=n,
            traces=traces,
        )
        
        # Aggregate stats
        analysis.avg_steps = sum(len(t.steps) for t in traces) / n if n else 0
        analysis.avg_duration_ms = sum(t.total_duration_ms for t in traces) / n if n else 0
        analysis.success_rate = sum(1 for t in traces if not t.errors) / n if n else 0
        
        # Tool patterns
        analysis.tool_patterns = self._extract_tool_patterns(traces)
        
        # Sequence patterns
        analysis.sequence_patterns = self._extract_sequence_patterns(traces)
        
        # State patterns
        analysis.state_patterns = self._extract_state_patterns(traces)
        
        # Failure patterns
        analysis.failure_patterns = self._extract_failure_patterns(traces)
        
        return analysis

    def _extract_tool_patterns(self, traces: list[AgentTrace]) -> list[ToolPattern]:
        """Extract tool call frequency and position patterns."""
        n = len(traces)
        tool_data: dict[str, list] = defaultdict(lambda: {"positions": [], "errors": 0, "count": 0})
        
        for trace in traces:
            tool_names = trace.tool_names
            seen = set()
            for i, name in enumerate(tool_names):
                tool_data[name]["positions"].append(i + 1)
                tool_data[name]["count"] += 1
                seen.add(name)
            
            # Count runs where tool appeared
            for name in seen:
                tool_data[name].setdefault("run_count", 0)
                tool_data[name]["run_count"] = tool_data[name].get("run_count", 0) + 1
            
            # Count errors
            for step in trace.tool_calls:
                if step.error:
                    tool_data[step.name]["errors"] += 1
        
        patterns = []
        for name, data in tool_data.items():
            positions = data["positions"]
            run_count = data.get("run_count", 0)
            total_calls = data["count"]
            
            import statistics
            avg_pos = statistics.mean(positions) if positions else 0
            std_pos = statistics.stdev(positions) if len(positions) > 1 else 0
            
            patterns.append(ToolPattern(
                tool_name=name,
                frequency=run_count / n if n else 0,
                avg_position=avg_pos,
                position_std=std_pos,
                error_rate=data["errors"] / total_calls if total_calls else 0,
            ))
        
        return sorted(patterns, key=lambda p: p.avg_position)

    def _extract_sequence_patterns(self, traces: list[AgentTrace]) -> list[SequencePattern]:
        """Extract common tool call sequences."""
        n = len(traces)
        sequence_counter: Counter = Counter()
        
        for trace in traces:
            seq = tuple(trace.tool_names)
            sequence_counter[seq] += 1
        
        patterns = []
        most_common_count = 0
        for seq, count in sequence_counter.most_common():
            freq = count / n
            is_most = count > most_common_count if most_common_count == 0 else False
            if most_common_count == 0:
                most_common_count = count
                is_most = True
            
            patterns.append(SequencePattern(
                sequence=list(seq),
                frequency=freq,
                is_dominant=is_most,
            ))
        
        return patterns

    def _extract_state_patterns(self, traces: list[AgentTrace]) -> list[StatePattern]:
        """Extract common state transitions."""
        n = len(traces)
        state_data: dict[str, dict] = defaultdict(lambda: {"count": 0, "values": [], "steps": []})
        
        for trace in traces:
            for i, step in enumerate(trace.steps):
                if step.kind == StepKind.STATE_CHANGE:
                    state_data[step.name]["count"] += 1
                    state_data[step.name]["values"].append(step.output)
                    state_data[step.name]["steps"].append(i + 1)
        
        patterns = []
        for key, data in state_data.items():
            values = data["values"]
            most_common_val = Counter(str(v) for v in values).most_common(1)
            
            import statistics
            patterns.append(StatePattern(
                state_key=key,
                frequency=data["count"] / n if n else 0,
                common_value=values[0] if values else None,
                avg_step=statistics.mean(data["steps"]) if data["steps"] else 0,
            ))
        
        return patterns

    def _extract_failure_patterns(self, traces: list[AgentTrace]) -> list[FailurePattern]:
        """Extract failure and recovery patterns."""
        n = len(traces)
        failure_data: dict[str, dict] = defaultdict(lambda: {"count": 0, "messages": [], "recoveries": []})
        
        for trace in traces:
            for i, step in enumerate(trace.steps):
                if step.error and step.kind == StepKind.TOOL_CALL:
                    failure_data[step.name]["count"] += 1
                    failure_data[step.name]["messages"].append(step.error)
                    
                    # Look at what happened next
                    if i + 1 < len(trace.steps):
                        next_step = trace.steps[i + 1]
                        failure_data[step.name]["recoveries"].append(next_step.name)
        
        patterns = []
        for tool_name, data in failure_data.items():
            messages = Counter(data["messages"]).most_common(1)
            recoveries = Counter(data["recoveries"]).most_common(1)
            
            patterns.append(FailurePattern(
                tool_name=tool_name,
                error_message=messages[0][0] if messages else "unknown",
                frequency=data["count"] / n if n else 0,
                recovery_action=recoveries[0][0] if recoveries else None,
            ))
        
        return patterns
