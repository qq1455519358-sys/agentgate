#!/usr/bin/env python3
"""
Verify AgentGate implementation against ICLR 2026 paper verbatim.

Source: "A Hitchhiker's Guide to Agent Evaluation"
URL: https://iclr-blogposts.github.io/2026/blog/2026/agent-evaluation/

For each of the 8 evaluation dimensions in the paper, this script:
  1. Quotes the paper's exact text
  2. Runs AgentGate code
  3. Asserts correctness with numeric checks
  4. Reports PASS/FAIL

Run: python _verify_against_paper.py
"""
from math import comb

from agentgate import (
    Scenario, TestSuite, MockAgent, AgentTrace, AgentStep, StepKind,
    ExpectMilestone, ExpectLLMJudge,
)
from agentgate.metrics import node_f1, edge_f1, tool_edit_distance


passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name}: {detail}")
        failed += 1


# ═══════════════════════════════════════════════════════════════════
# 1. PRIMARY METRICS — Success rate
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1/8 PRIMARY METRICS")
print('Paper: "Most agent benchmarks report success rates or task')
print('       completion percentages as the main metric"')
print("=" * 60)

mock = MockAgent()
mock.add_trace("t1", AgentTrace(input="t1", steps=[
    AgentStep(kind=StepKind.TOOL_CALL, name="a", output="ok"),
]))
mock.add_trace("t2", AgentTrace(input="t2", steps=[
    AgentStep(kind=StepKind.TOOL_CALL, name="b", output="fail"),
]))

s1 = Scenario("Pass", input="t1")
s1.expect_tool_call("a")
s2 = Scenario("Fail", input="t2")
s2.expect_tool_call("a")  # will fail — agent used "b"

suite = TestSuite("primary")
suite.add(s1)
suite.add(s2)
result = suite.run(mock)

check("pass_rate = passed/total", result.pass_rate == 0.5,
      f"got {result.pass_rate}")
check("pass_rate is float 0-1", 0.0 <= result.pass_rate <= 1.0)
check("suite.passed reflects all()", result.passed == False)

# ═══════════════════════════════════════════════════════════════════
# 2. CONSISTENCY — pass@k and pass^k
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2/8 CONSISTENCY METRICS")
print('Paper: "pass@k = Success in one of k attempts"')
print('       "pass^k = Success in all k attempts"')
print('       "pass^k_i = C(c_i, k) / C(n_i, k)"  (τ-bench)')
print("=" * 60)

# Create mock that succeeds 3 out of 4 times
counter = {"n": 0}
mock2 = MockAgent()

def flaky_run(input_text):
    counter["n"] += 1
    if counter["n"] % 4 != 0:  # 3/4 succeed
        return AgentTrace(input=input_text, steps=[
            AgentStep(kind=StepKind.TOOL_CALL, name="check", output="ok"),
        ])
    else:
        return AgentTrace(input=input_text, steps=[
            AgentStep(kind=StepKind.TOOL_CALL, name="fail", output="nope"),
        ])

mock2.run = flaky_run
s = Scenario("Flaky", input="test")
s.expect_tool_call("check")
s.expect_no_tool_call("fail")
suite2 = TestSuite("consistency")
suite2.add(s)
result2 = suite2.run(mock2, runs=4, min_pass_rate=0.0)

# τ-bench uses C(c,k)/C(n,k) for both pass@k and pass^k
# pass@1 = C(c,1)/C(n,1) = c/n (simple success rate)
check("pass_at_k (= pass@1 = c/n)", result2.pass_at_k is not None)
if result2.pass_at_k is not None:
    # c=3, n=4: pass@1 = 3/4 = 0.75
    check("pass@1 = c/n = 3/4 = 0.75",
          abs(result2.pass_at_k - 0.75) < 0.01,
          f"got {result2.pass_at_k}")

# Verify pass^k formula: C(c,k)/C(n,k) series
series_pow = result2.pass_power_k_series()
check("pass_power_k_series returns dict", series_pow is not None)
if series_pow:
    # pass^1 = C(3,1)/C(4,1) = 3/4 = 0.75
    check("pass^1 = C(c,1)/C(n,1) = C(3,1)/C(4,1) = 3/4",
          abs(series_pow[1] - 0.75) < 0.01,
          f"got {series_pow.get(1)}")
    # pass^2 = C(3,2)/C(4,2) = 3/6 = 0.5
    check("pass^2 = C(c,2)/C(n,2) = C(3,2)/C(4,2) = 3/6",
          abs(series_pow[2] - 0.5) < 0.01,
          f"got {series_pow.get(2)}")
    # pass^3 = C(3,3)/C(4,3) = 1/4 = 0.25
    check("pass^3 = C(3,3)/C(4,3) = 1/4 = 0.25",
          abs(series_pow[3] - 0.25) < 0.01,
          f"got {series_pow.get(3)}")
    # pass^4 = C(3,4)/C(4,4) = 0/1 = 0.0  (can't choose 4 from 3)
    check("pass^4 = C(3,4)/C(4,4) = 0 (not all passed)",
          series_pow[4] == 0.0,
          f"got {series_pow.get(4)}")
    # Monotonically non-increasing
    vals = list(series_pow.values())
    check("pass^k non-increasing as k increases",
          all(vals[i] >= vals[i+1] for i in range(len(vals)-1)),
          f"series: {series_pow}")

# Paper quote verification: "GPT-4's success drops from ~61% (pass@1) to ~25% for pass^8"
check("Paper fact: pass^k(max) <= pass^1 always holds",
      series_pow[max(series_pow)] <= series_pow[1])


# ═══════════════════════════════════════════════════════════════════
# 3. POLICY ADHERENCE (CuP)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3/8 POLICY ADHERENCE")
print('Paper: "Completion under Policy (CuP), which gives credit only')
print('       if no policy is violated"')
print('       Ref: ST-WebAgentBench')
print("=" * 60)

# Agent that completes task but violates policy
mock3 = MockAgent()
mock3.add_trace("violate", AgentTrace(
    input="violate",
    steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="complete_task", output="done"),
        AgentStep(kind=StepKind.TOOL_CALL, name="export_all_data", output="leaked"),
    ],
    output="Task done (but data leaked)",
))

s3 = Scenario("CuP test", input="violate")
s3.expect_tool_call("complete_task")  # task completed ✅
s3.expect_policy("Data Protection", forbidden_tools=["export_all_data"])

suite3 = TestSuite("cup")
suite3.add(s3)
result3 = suite3.run(mock3)

check("CuP: task completed but policy violated → FAIL",
      result3.passed == False)
check("CuP: zero credit for policy violation (paper: 'credit only if no policy violated')",
      result3.results[0].passed == False)

# Agent that completes AND follows policy
mock3b = MockAgent()
mock3b.add_trace("safe", AgentTrace(
    input="safe",
    steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="complete_task", output="done"),
    ],
    output="Task done safely",
))

s3b = Scenario("CuP safe", input="safe")
s3b.expect_tool_call("complete_task")
s3b.expect_policy("Data Protection", forbidden_tools=["export_all_data"])

suite3b = TestSuite("cup-safe")
suite3b.add(s3b)
result3b = suite3b.run(mock3b)

check("CuP: task completed + policy followed → PASS",
      result3b.passed == True)


# ═══════════════════════════════════════════════════════════════════
# 4. ADVERSARIAL SAFETY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4/8 ADVERSARIAL SAFETY")
print('Paper: "CoSafe benchmark feeds agents adversarial prompts...and')
print('       measures the rate of unsafe completions"')
print("=" * 60)

from agentgate.adversarial import (
    prompt_injection, privilege_escalation,
    tool_misuse, data_exfiltration, owasp_agentic_suite,
)

inj = prompt_injection(dangerous_tools=["rm_rf", "drop_table"])
check("prompt_injection creates scenarios", isinstance(inj, list) and len(inj) > 0)
check("prompt_injection scenarios have expectations",
      all(len(s.expectations) > 0 for s in inj))

priv = privilege_escalation(dangerous_tools=["admin_panel"])
check("privilege_escalation creates scenarios", isinstance(priv, list) and len(priv) > 0)

mis = tool_misuse(tools={"eval": ["execute arbitrary code", "run shell"]})
check("tool_misuse creates scenarios", isinstance(mis, list) and len(mis) > 0)

exf = data_exfiltration(sensitive_tools=["upload_to_external"])
check("data_exfiltration creates scenarios", isinstance(exf, list) and len(exf) > 0)

owasp = owasp_agentic_suite(dangerous_tools=["rm_rf"])
check("owasp_agentic_suite returns list of scenarios", len(owasp) >= 4)

# Safe agent should pass
safe_mock = MockAgent()
for sc in owasp:
    safe_mock.add_trace(sc.input, AgentTrace(
        input=sc.input,
        steps=[
            AgentStep(kind=StepKind.TOOL_CALL, name="safe_respond", output="I cannot do that"),
        ],
        output="I cannot assist with that request.",
    ))
suite4 = TestSuite("adversarial")
for sc in owasp:
    suite4.add(sc)
result4 = suite4.run(safe_mock)
check("Safe agent passes adversarial suite", result4.passed)


# ═══════════════════════════════════════════════════════════════════
# 5. TOOL-CALL ANALYSIS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5/8 TOOL-CALL ANALYSIS")
print('Paper Table:')
print('  "Node F1: Correct tools chosen (graph-based)"')
print('  "Edge F1: Correct ordering of tools"')
print('  "Normalized Edit Distance: Similarity to reference trajectory"')
print("=" * 60)

trace5 = AgentTrace(
    input="book flight",
    steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="ok"),
        AgentStep(kind=StepKind.TOOL_CALL, name="book", output="ok"),
        AgentStep(kind=StepKind.TOOL_CALL, name="confirm", output="ok"),
    ],
)
expected = ["search", "book", "confirm"]

# Perfect match
nf1 = node_f1(trace5, expected)
check("node_f1 perfect match = 1.0", nf1 == 1.0, f"got {nf1}")

ef1 = edge_f1(trace5, expected)
check("edge_f1 perfect match = 1.0", ef1 == 1.0, f"got {ef1}")

ed = tool_edit_distance(trace5, expected)
check("edit_distance perfect match = 0.0", ed == 0.0, f"got {ed}")

# Partial match: agent called wrong tools
trace5b = AgentTrace(
    input="book flight",
    steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="ok"),
        AgentStep(kind=StepKind.TOOL_CALL, name="random", output="oops"),
    ],
)

nf1b = node_f1(trace5b, expected)
check("node_f1 partial: 0 < score < 1", 0 < nf1b < 1, f"got {nf1b}")
# precision = 1/2 (search is correct), recall = 1/3 (only 1 of 3 found)
# F1 = 2*(0.5*0.333)/(0.5+0.333) = 0.4
check("node_f1 math: P=1/2, R=1/3, F1=2*P*R/(P+R) = 0.4",
      abs(nf1b - 0.4) < 0.01, f"got {nf1b}")

edb = tool_edit_distance(trace5b, expected)
check("edit_distance partial > 0", edb > 0, f"got {edb}")
check("edit_distance normalized to [0,1]", 0 <= edb <= 1, f"got {edb}")


# ═══════════════════════════════════════════════════════════════════
# 6. EFFICIENCY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6/8 EFFICIENCY")
print('Paper Auxiliary measures: "Latency, token cost, number of steps"')
print("=" * 60)

mock6 = MockAgent()
mock6.add_trace("slow", AgentTrace(
    input="slow",
    steps=[AgentStep(kind=StepKind.TOOL_CALL, name=f"step{i}", output="ok")
           for i in range(10)],
    total_duration_ms=5000,
))

# Step limit
s6a = Scenario("Step limit", input="slow")
s6a.expect_max_steps(5)
suite6a = TestSuite("eff-steps")
suite6a.add(s6a)
r6a = suite6a.run(mock6)
check("expect_max_steps(5) fails for 10-step agent",
      r6a.passed == False)

s6b = Scenario("Step OK", input="slow")
s6b.expect_max_steps(15)
suite6b = TestSuite("eff-steps-ok")
suite6b.add(s6b)
r6b = suite6b.run(mock6)
check("expect_max_steps(15) passes for 10-step agent",
      r6b.passed == True)

# Duration limit
s6c = Scenario("Duration limit", input="slow")
s6c.expect_max_duration(1000)  # 1s limit but agent took 5s
suite6c = TestSuite("eff-dur")
suite6c.add(s6c)
r6c = suite6c.run(mock6)
check("expect_max_duration(1000ms) fails for 5000ms agent",
      r6c.passed == False)


# ═══════════════════════════════════════════════════════════════════
# 7. MILESTONES (PARTIAL CREDIT)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7/8 MILESTONES AND SUBGOALS")
print('Paper: "By breaking a task into milestones, evaluators can')
print('       compute metrics like: Fraction of subtasks achieved,')
print('       Milestone-based accuracy, Progress score (even for')
print('       failed tasks)"')
print('       Ref: TheAgentCompany, WebCanvas')
print("=" * 60)

# Agent completes 1 of 3 milestones
mock7 = MockAgent()
mock7.add_trace("partial", AgentTrace(
    input="partial",
    steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="ok"),
        # stops here — didn't select or book
    ],
))

s7 = Scenario("Flight booking", input="partial")
s7.expect_milestone("Searched flights", tool="search", weight=1.0)
s7.expect_milestone("Selected flight", tool="select", weight=1.0)
s7.expect_milestone("Booked flight", tool="book", weight=2.0)

suite7 = TestSuite("milestones")
suite7.add(s7)
r7 = suite7.run(mock7)

# Paper says "Fraction of subtasks achieved"
check("Fraction of subtasks: 1/3 milestones reached",
      sum(1 for e in r7.results[0].expectations if e.passed) == 1)

# Paper says "Progress score (even for failed tasks)"
check("Progress score > 0 even though task failed",
      r7.results[0].score > 0)
check("Progress score < 1 (task incomplete)",
      r7.results[0].score < 1.0)

# Weighted: 1/(1+1+2) = 0.25
check("Weighted score = 1/4 = 0.25 (weight-proportional)",
      abs(r7.results[0].score - 0.25) < 0.01,
      f"got {r7.results[0].score}")

# Full completion
mock7b = MockAgent()
mock7b.add_trace("full", AgentTrace(
    input="full",
    steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search", output="ok"),
        AgentStep(kind=StepKind.TOOL_CALL, name="select", output="ok"),
        AgentStep(kind=StepKind.TOOL_CALL, name="book", output="ok"),
    ],
))
s7b = Scenario("Full booking", input="full")
s7b.expect_milestone("Searched", tool="search", weight=1.0)
s7b.expect_milestone("Selected", tool="select", weight=1.0)
s7b.expect_milestone("Booked", tool="book", weight=2.0)
suite7b = TestSuite("full")
suite7b.add(s7b)
r7b = suite7b.run(mock7b)
check("Full completion → score = 1.0", r7b.results[0].score == 1.0)

# average_score across suite
mock7c = MockAgent()
mock7c.add_trace("good", AgentTrace(input="good", steps=[
    AgentStep(kind=StepKind.TOOL_CALL, name="a", output="ok"),
    AgentStep(kind=StepKind.TOOL_CALL, name="b", output="ok"),
]))
mock7c.add_trace("bad", AgentTrace(input="bad", steps=[
    AgentStep(kind=StepKind.TOOL_CALL, name="a", output="ok"),
]))
sg = Scenario("Good", input="good")
sg.expect_milestone("A", tool="a")
sg.expect_milestone("B", tool="b")
sb = Scenario("Bad", input="bad")
sb.expect_milestone("A", tool="a")
sb.expect_milestone("B", tool="b")
suite7c = TestSuite("avg")
suite7c.add(sg)
suite7c.add(sb)
r7c = suite7c.run(mock7c)
check("average_score across scenarios = (1.0+0.5)/2 = 0.75",
      abs(r7c.average_score - 0.75) < 0.01,
      f"got {r7c.average_score}")


# ═══════════════════════════════════════════════════════════════════
# 8. AGENT-AS-A-JUDGE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("8/8 AGENT-AS-A-JUDGE")
print('Paper: "The LLM-as-a-Judge paradigm employs a large model to')
print('       score or critique an agent\'s multi-step output"')
print('       Ref: Zhuge et al. (2024) Agent-as-a-Judge')
print("=" * 60)

# Judge that checks if agent searched before booking
def order_judge(criteria, trace_text):
    lines = trace_text.split("\n")
    search_idx = None
    book_idx = None
    for i, line in enumerate(lines):
        if "search" in line.lower() and "tool_call" in line:
            search_idx = i
        if "book" in line.lower() and "tool_call" in line:
            book_idx = i
    if search_idx is not None and book_idx is not None and search_idx < book_idx:
        return (True, "Agent searched before booking")
    return (False, "Agent did not search before booking")

mock8 = MockAgent()
mock8.add_trace("correct", AgentTrace(
    input="correct",
    steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="search_flights", output="3 results"),
        AgentStep(kind=StepKind.TOOL_CALL, name="book_flight", output="confirmed"),
    ],
    output="Flight booked!",
))

s8 = Scenario("Order check", input="correct")
s8.expect_llm_judge("Agent must search before booking", judge_fn=order_judge)
suite8 = TestSuite("judge")
suite8.add(s8)
r8 = suite8.run(mock8)
check("LLM judge passes when criteria met", r8.passed)

# Judge with wrong order
mock8b = MockAgent()
mock8b.add_trace("wrong", AgentTrace(
    input="wrong",
    steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="book_flight", output="confirmed"),
        AgentStep(kind=StepKind.TOOL_CALL, name="search_flights", output="3 results"),
    ],
))

s8b = Scenario("Wrong order", input="wrong")
s8b.expect_llm_judge("Agent must search before booking", judge_fn=order_judge)
suite8b = TestSuite("judge-fail")
suite8b.add(s8b)
r8b = suite8b.run(mock8b)
check("LLM judge fails when criteria violated", r8b.passed == False)
check("Judge provides reasoning on failure",
      len(r8b.results[0].failed_expectations) > 0
      and "search" in r8b.results[0].failed_expectations[0].detail.lower())

# Dict return format
def dict_judge(criteria, trace_text):
    return {"passed": True, "reasoning": "All criteria met"}

mock8c = MockAgent()
mock8c.add_trace("q", AgentTrace(input="q", steps=[]))
s8c = Scenario("Dict judge", input="q")
s8c.expect_llm_judge("check", judge_fn=dict_judge)
suite8c = TestSuite("dict")
suite8c.add(s8c)
r8c = suite8c.run(mock8c)
check("Judge with dict return format works", r8c.passed)

# Error handling
def broken_judge(criteria, trace_text):
    raise ConnectionError("LLM API timeout")

mock8d = MockAgent()
mock8d.add_trace("q", AgentTrace(input="q", steps=[]))
s8d = Scenario("Error judge", input="q")
s8d.expect_llm_judge("check", judge_fn=broken_judge)
suite8d = TestSuite("error")
suite8d.add(s8d)
r8d = suite8d.run(mock8d)
check("Judge error → graceful failure (no crash)",
      r8d.passed == False)
check("Judge error contains exception info",
      "timeout" in r8d.results[0].failed_expectations[0].detail.lower())

# trace_to_text includes all components
trace8e = AgentTrace(
    input="test input",
    steps=[
        AgentStep(kind=StepKind.TOOL_CALL, name="tool_a",
                  input={"param": "value"}, output="result"),
    ],
    output="final answer",
)
text = ExpectLLMJudge._trace_to_text(trace8e)
check("trace_to_text includes input", "test input" in text)
check("trace_to_text includes tool calls", "tool_a" in text)
check("trace_to_text includes tool args", "param=value" in text)
check("trace_to_text includes output", "final answer" in text)


# ═══════════════════════════════════════════════════════════════════
# MATHEMATICAL FORMULA VERIFICATION
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FORMULA VERIFICATION")
print("Verify τ-bench formulas with known values")
print("=" * 60)

# τ-bench formula: pass^k_i = C(c_i, k) / C(n_i, k)
# Example from paper: GPT-4 airline pass^1=0.420, pass^4=0.200
# If n=8, c≈5: pass^1 = C(5,1)/C(8,1) = 5/8 = 0.625 (close-ish)
# The exact numbers depend on n and c per scenario, but the formula should work

# Manual calculation: n=6, c=4
n, c = 6, 4
manual_series = {}
for k in range(1, n+1):
    manual_series[k] = comb(c, k) / comb(n, k) if comb(n, k) > 0 else 0.0

check(f"Manual C(4,1)/C(6,1) = {comb(4,1)}/{comb(6,1)} = {comb(4,1)/comb(6,1):.4f}",
      abs(manual_series[1] - 4/6) < 0.001)
check(f"Manual C(4,2)/C(6,2) = {comb(4,2)}/{comb(6,2)} = {comb(4,2)/comb(6,2):.4f}",
      abs(manual_series[2] - 6/15) < 0.001)
check(f"Manual C(4,3)/C(6,3) = {comb(4,3)}/{comb(6,3)} = {comb(4,3)/comb(6,3):.4f}",
      abs(manual_series[3] - 4/20) < 0.001)
check(f"Manual C(4,4)/C(6,4) = {comb(4,4)}/{comb(6,4)} = {comb(4,4)/comb(6,4):.4f}",
      abs(manual_series[4] - 1/15) < 0.001)
check(f"Manual C(4,5)/C(6,5) = 0 (can't choose 5 from 4)",
      manual_series[5] == 0.0)
check(f"Manual C(4,6)/C(6,6) = 0 (can't choose 6 from 4)",
      manual_series[6] == 0.0)
check("Series is monotonically non-increasing",
      all(manual_series[k] >= manual_series[k+1] for k in range(1, n)))

# Now verify AgentGate computes the same
counter2 = {"n": 0}
mock_formula = MockAgent()
def formula_run(input_text):
    counter2["n"] += 1
    if counter2["n"] <= 4:  # c=4 out of n=6
        return AgentTrace(input=input_text, steps=[
            AgentStep(kind=StepKind.TOOL_CALL, name="correct", output="ok"),
        ])
    else:
        return AgentTrace(input=input_text, steps=[
            AgentStep(kind=StepKind.TOOL_CALL, name="wrong", output="fail"),
        ])
mock_formula.run = formula_run

sf = Scenario("Formula", input="test")
sf.expect_tool_call("correct")
sf.expect_no_tool_call("wrong")
suite_f = TestSuite("formula")
suite_f.add(sf)
rf = suite_f.run(mock_formula, runs=6, min_pass_rate=0.0)
computed = rf.pass_power_k_series()

check("AgentGate pass^k matches manual for k=1",
      abs(computed[1] - manual_series[1]) < 0.001,
      f"computed={computed[1]}, expected={manual_series[1]}")
check("AgentGate pass^k matches manual for k=2",
      abs(computed[2] - manual_series[2]) < 0.001,
      f"computed={computed[2]}, expected={manual_series[2]}")
check("AgentGate pass^k matches manual for k=3",
      abs(computed[3] - manual_series[3]) < 0.001,
      f"computed={computed[3]}, expected={manual_series[3]}")
check("AgentGate pass^k matches manual for k=6 (= 0.0)",
      computed[6] == 0.0,
      f"computed={computed[6]}")


# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
total = passed + failed
print(f"VERIFICATION COMPLETE: {passed}/{total} checks passed")
if failed == 0:
    print("🎉 ALL CHECKS PASSED — Implementation matches paper!")
else:
    print(f"⚠️  {failed} checks FAILED")
print("=" * 60)

exit(0 if failed == 0 else 1)
